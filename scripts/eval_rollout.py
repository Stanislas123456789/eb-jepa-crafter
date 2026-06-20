#!/usr/bin/env python3
"""
Rollout evaluation and dreaming visualization for a trained JEPA world model on Crafter.

Part A: Computes per-horizon MSE between autoregressive predictions and ground-truth
        encoded states, along with a copy-baseline comparison.
Part B: Creates side-by-side "dreaming" visualizations showing actual frames,
        per-step latent MSE, and (optionally) probe prediction comparisons.

Usage:
    python scripts/eval_rollout.py \
        --checkpoint_path checkpoints/latest.pth.tar \
        --data_dir data/crafter_trajectories \
        --output_dir eval_results \
        --num_batches 50 \
        --batch_size 16
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn

# Ensure the project root is on sys.path so imports resolve
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from eb_jepa.action_encoders import ActionEmbeddingEncoder
from eb_jepa.architectures import (
    ImpalaEncoder,
    InverseDynamicsModel,
    RNNPredictor,
)
from eb_jepa.jepa import JEPA
from eb_jepa.losses import SquareLossSeq, VC_IDM_Sim_Regularizer
from eb_jepa.datasets.crafter.crafter_dataset import (
    CrafterTrajDataset,
    CrafterSlicedDataset,
)
from eb_jepa.datasets.crafter.normalizer import CrafterNormalizer


# ---------------------------------------------------------------------------
# Model construction (mirrors examples/ac_video_jepa/main.py for Crafter)
# ---------------------------------------------------------------------------

def build_jepa_model(
    dobs: int = 3,
    henc: int = 32,
    dstc: int = 32,
    img_size: int = 64,
    num_actions: int = 17,
    d_action_emb: int = 32,
    device: torch.device = torch.device("cpu"),
) -> JEPA:
    """Reconstruct the JEPA model architecture used for Crafter training."""

    encoder = ImpalaEncoder(
        width=1,
        stack_sizes=(16, henc, dstc),
        num_blocks=2,
        dropout_rate=None,
        layer_norm=False,
        input_channels=dobs,
        final_ln=True,
        mlp_output_dim=512,
        input_shape=(dobs, img_size, img_size),
    )

    aencoder = ActionEmbeddingEncoder(num_actions, d_action_emb)

    predictor = RNNPredictor(
        hidden_size=encoder.mlp_output_dim,
        action_dim=d_action_emb,
        final_ln=encoder.final_ln,
    )

    # Regularizer and predcost are not used during eval but JEPA.__init__ requires them.
    # We construct them so the state dict keys match the checkpoint.

    # Determine spatial dims by running a dummy forward
    with torch.no_grad():
        dummy = torch.zeros(1, dobs, 1, img_size, img_size)
        enc_out = encoder(dummy)
        _, f, _, h, w = enc_out.shape

    idm = InverseDynamicsModel(
        state_dim=h * w * f,
        hidden_dim=256,
        action_dim=num_actions,
    )

    regularizer = VC_IDM_Sim_Regularizer(
        cov_coeff=8,
        std_coeff=16,
        sim_coeff_t=12,
        idm_coeff=1,
        idm=idm,
        first_t_only=False,
        projector=None,
        spatial_as_samples=False,
        idm_after_proj=False,
        sim_t_after_proj=False,
        discrete=True,
    )

    ploss = SquareLossSeq()

    jepa = JEPA(encoder, aencoder, predictor, regularizer, ploss)
    return jepa.to(device)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(jepa: JEPA, path: str, device: torch.device):
    """Load a training checkpoint into the JEPA model."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    # Handle compiled-model key prefix
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    jepa.load_state_dict(state_dict, strict=True)
    epoch = checkpoint.get("epoch", "?")
    step = checkpoint.get("step", "?")
    print(f"Loaded checkpoint from {path}  (epoch={epoch}, step={step})")
    return checkpoint


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_val_loader(data_dir: str, batch_size: int, sample_length: int = 17):
    """Build a validation DataLoader from Crafter trajectory .npz files."""
    traj_dataset = CrafterTrajDataset(data_dir=data_dir, sample_length=sample_length)

    # Reproduce the same train/val split as init_data in utils.py
    num_episodes = len(traj_dataset)
    train_fraction = 0.9
    num_train = int(train_fraction * num_episodes)

    indices = torch.randperm(num_episodes, generator=torch.Generator().manual_seed(42))
    val_indices = indices[num_train:].tolist()

    # Subset wrapper (same as _SubsetTrajDataset in utils.py)
    class _SubsetTrajDataset:
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
            self.action_dim = dataset.action_dim
            self.proprio_dim = dataset.proprio_dim
            self.state_dim = dataset.state_dim

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]

        def get_seq_length(self, idx):
            return self.dataset.get_seq_length(self.indices[idx])

    # Train subset for normalizer computation
    train_indices = indices[:num_train].tolist()
    train_traj = _SubsetTrajDataset(traj_dataset, train_indices)
    val_traj = _SubsetTrajDataset(traj_dataset, val_indices)

    # Compute normalizer on train data (matches training)
    train_dset = CrafterSlicedDataset(
        train_traj, sample_length=sample_length, num_stats_samples=5000
    )
    val_dset = CrafterSlicedDataset(
        val_traj, sample_length=sample_length, normalizer=train_dset.normalizer
    )

    loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=min(batch_size, len(val_dset)) if len(val_dset) > 0 else 1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    print(f"Validation set: {len(val_dset)} slices, {len(loader)} batches (bs={batch_size})")
    return loader, val_dset


# ---------------------------------------------------------------------------
# Part A: Rollout MSE evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_rollout_mse(jepa, loader, num_batches, device):
    """Compute per-horizon MSE between predicted and ground-truth encoded states."""
    jepa.eval()

    all_mse = []       # list of [T-1] tensors (mean over B and D)
    all_copy_mse = []  # copy-baseline MSE

    for batch_idx, (obs, actions, probe_labels, _, _) in enumerate(loader):
        if batch_idx >= num_batches:
            break

        obs = obs.to(device)       # [B, C, T, H, W]
        actions = actions.to(device)  # [B, T]
        B, C, T, H, W = obs.shape

        # 1. Encode full trajectory (ground truth)
        gt_encoded = jepa.encode(obs)  # [B, 512, T, 1, 1]

        # 2. Encode only first frame
        obs_init = obs[:, :, 0:1]  # [B, C, 1, H, W]

        # 3. Autoregressive unroll
        nsteps = T - 1
        predicted, _ = jepa.unroll(
            obs_init,
            actions,
            nsteps=nsteps,
            unroll_mode="autoregressive",
            compute_loss=False,
        )
        # predicted shape: [B, 512, T, 1, 1]  (1 context + nsteps predictions)

        # 4. Per-timestep MSE for predicted vs ground truth (skip first frame = context)
        # predicted[:,:,1:] are the predicted future states
        # gt_encoded[:,:,1:] are the actual encoded future states
        mse_per_step = ((gt_encoded[:, :, 1:] - predicted[:, :, 1:]) ** 2).mean(
            dim=(1, 3, 4)
        )  # [B, T-1]
        all_mse.append(mse_per_step.cpu())

        # 5. Copy baseline: repeat first encoded state
        first_encoded = gt_encoded[:, :, 0:1].expand_as(gt_encoded[:, :, 1:])
        copy_mse_per_step = ((gt_encoded[:, :, 1:] - first_encoded) ** 2).mean(
            dim=(1, 3, 4)
        )  # [B, T-1]
        all_copy_mse.append(copy_mse_per_step.cpu())

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed batch {batch_idx + 1}/{min(num_batches, len(loader))}")

    # Concatenate over all samples
    all_mse = torch.cat(all_mse, dim=0)           # [N_total, T-1]
    all_copy_mse = torch.cat(all_copy_mse, dim=0)  # [N_total, T-1]

    mean_mse = all_mse.mean(dim=0).numpy()          # [T-1]
    std_mse = all_mse.std(dim=0).numpy()             # [T-1]
    mean_copy_mse = all_copy_mse.mean(dim=0).numpy()
    std_copy_mse = all_copy_mse.std(dim=0).numpy()

    n_samples = all_mse.shape[0]

    return {
        "mean_mse_per_step": mean_mse,
        "std_mse_per_step": std_mse,
        "mean_copy_baseline_per_step": mean_copy_mse,
        "std_copy_baseline_per_step": std_copy_mse,
        "n_samples": n_samples,
    }


# ---------------------------------------------------------------------------
# Part A: Plotting
# ---------------------------------------------------------------------------

def plot_mse_vs_horizon(results, output_path):
    """Plot MSE vs horizon step with copy baseline."""
    steps = np.arange(1, len(results["mean_mse_per_step"]) + 1)
    mean_mse = results["mean_mse_per_step"]
    std_mse = results["std_mse_per_step"]
    mean_copy = results["mean_copy_baseline_per_step"]
    std_copy = results["std_copy_baseline_per_step"]
    n = results["n_samples"]
    se_mse = std_mse / np.sqrt(n)
    se_copy = std_copy / np.sqrt(n)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps, mean_mse, "o-", color="#2563eb", linewidth=2, markersize=5,
            label="JEPA Autoregressive")
    ax.fill_between(steps, mean_mse - se_mse, mean_mse + se_mse,
                     alpha=0.2, color="#2563eb")

    ax.plot(steps, mean_copy, "s--", color="#dc2626", linewidth=2, markersize=5,
            label="Copy Baseline (repeat t=0)")
    ax.fill_between(steps, mean_copy - se_copy, mean_copy + se_copy,
                     alpha=0.2, color="#dc2626")

    ax.set_xlabel("Horizon Step", fontsize=13)
    ax.set_ylabel("MSE (latent space)", fontsize=13)
    ax.set_title(f"Rollout MSE vs. Horizon  (n={n} trajectories)", fontsize=14)
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(steps)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved MSE-vs-horizon plot to {output_path}")


# ---------------------------------------------------------------------------
# Part B: Dreaming visualization
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_dreaming_visualizations(
    jepa, loader, output_dir, device, num_samples=5, probe_path=None
):
    """Create side-by-side dreaming visualizations for individual trajectories."""
    jepa.eval()

    # Optionally load a probe
    probe = None
    if probe_path and os.path.exists(probe_path):
        try:
            probe_ckpt = torch.load(probe_path, map_location=device, weights_only=False)
            # Build a simple linear probe head: 512 -> 16 (Crafter probe labels)
            probe_head = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 16),
            ).to(device)
            # Try loading state dict
            probe_sd = probe_ckpt.get("head_state_dict", probe_ckpt.get("model_state_dict", probe_ckpt))
            if isinstance(probe_sd, dict):
                probe_head.load_state_dict(probe_sd, strict=False)
            probe = probe_head
            probe.eval()
            print(f"Loaded probe from {probe_path}")
        except Exception as e:
            print(f"Warning: could not load probe from {probe_path}: {e}")
            probe = None

    # Get the dataset normalizer for unnormalizing frames
    normalizer = loader.dataset.normalizer

    sample_count = 0
    for batch_idx, (obs, actions, probe_labels, _, _) in enumerate(loader):
        if sample_count >= num_samples:
            break

        obs = obs.to(device)
        actions = actions.to(device)
        probe_labels = probe_labels.to(device)  # [B, K, T]
        B, C, T, H, W = obs.shape

        # Encode full trajectory
        gt_encoded = jepa.encode(obs)  # [B, 512, T, 1, 1]

        # Autoregressive unroll from first frame
        obs_init = obs[:, :, 0:1]
        nsteps = T - 1
        predicted, _ = jepa.unroll(
            obs_init,
            actions,
            nsteps=nsteps,
            unroll_mode="autoregressive",
            compute_loss=False,
        )

        # Per-timestep MSE
        mse_per_step = ((gt_encoded[:, :, 1:] - predicted[:, :, 1:]) ** 2).mean(
            dim=(1, 3, 4)
        )  # [B, T-1]

        # Probe predictions (if probe available)
        gt_probe_pred = None
        pred_probe_pred = None
        if probe is not None:
            # Apply probe to ground-truth embeddings
            gt_flat = gt_encoded.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # [B, T, 512]
            gt_flat = gt_flat.reshape(B * T, 512)
            gt_probe_pred = probe(gt_flat).reshape(B, T, -1)  # [B, T, K]

            # Apply probe to predicted embeddings
            pred_flat = predicted.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # [B, T, 512]
            pred_flat = pred_flat.reshape(B * T, 512)
            pred_probe_pred = probe(pred_flat).reshape(B, T, -1)  # [B, T, K]

        # Process individual samples in the batch
        for i in range(min(B, num_samples - sample_count)):
            _create_dreaming_figure(
                obs_sample=obs[i].cpu(),
                mse_sample=mse_per_step[i].cpu().numpy(),
                gt_probe=gt_probe_pred[i].cpu().numpy() if gt_probe_pred is not None else None,
                pred_probe=pred_probe_pred[i].cpu().numpy() if pred_probe_pred is not None else None,
                probe_labels_sample=probe_labels[i].cpu().numpy(),  # [K, T]
                normalizer=normalizer,
                sample_idx=sample_count,
                output_dir=output_dir,
            )
            sample_count += 1
            if sample_count >= num_samples:
                break


def _create_dreaming_figure(
    obs_sample,
    mse_sample,
    gt_probe,
    pred_probe,
    probe_labels_sample,
    normalizer,
    sample_idx,
    output_dir,
):
    """Create and save a dreaming visualization for a single trajectory.

    Layout:
        Row 1: Actual observation frames
        Row 2: Per-step latent MSE bar chart
        Row 3: Probe predictions comparison (if available)
    """
    # obs_sample: [C, T, H, W], normalized
    C, T, H, W = obs_sample.shape

    # Unnormalize observations for display
    obs_unnorm = normalizer.unnormalize_state(obs_sample)  # [C, T, H, W]
    obs_unnorm = obs_unnorm.clamp(0, 1)

    has_probe = gt_probe is not None and pred_probe is not None
    num_rows = 3 if has_probe else 2

    # Select a subset of timesteps to display (show every frame if T <= 17, else subsample)
    max_frames = min(T, 17)
    frame_indices = np.linspace(0, T - 1, max_frames, dtype=int)

    fig = plt.figure(figsize=(max(2.0 * max_frames, 14), 3.5 * num_rows))
    gs = gridspec.GridSpec(num_rows, 1, height_ratios=[1, 0.8] + ([1] if has_probe else []),
                           hspace=0.35)

    # --- Row 1: Observation frames ---
    gs_frames = gridspec.GridSpecFromSubplotSpec(1, max_frames, subplot_spec=gs[0],
                                                  wspace=0.05)
    for j, t_idx in enumerate(frame_indices):
        ax = fig.add_subplot(gs_frames[0, j])
        frame = obs_unnorm[:, t_idx].permute(1, 2, 0).numpy()  # [H, W, C]
        ax.imshow(frame)
        ax.set_title(f"t={t_idx}", fontsize=8)
        ax.axis("off")

    # --- Row 2: MSE bar chart ---
    ax_mse = fig.add_subplot(gs[1])
    horizon_steps = np.arange(1, len(mse_sample) + 1)
    colors = plt.cm.YlOrRd(mse_sample / (mse_sample.max() + 1e-8))
    ax_mse.bar(horizon_steps, mse_sample, color=colors, edgecolor="gray", linewidth=0.5)
    ax_mse.set_xlabel("Horizon Step", fontsize=10)
    ax_mse.set_ylabel("Latent MSE", fontsize=10)
    ax_mse.set_title("Predicted vs. Actual Latent MSE", fontsize=11)
    ax_mse.set_xticks(horizon_steps)
    ax_mse.grid(axis="y", alpha=0.3)

    # --- Row 3: Probe comparison (if available) ---
    if has_probe:
        ax_probe = fig.add_subplot(gs[2])

        # probe_labels_sample: [K, T]
        # Show a few key probes: health (idx 0) and a couple others
        K = probe_labels_sample.shape[0]
        probe_names = _get_crafter_probe_names(K)

        # Pick up to 4 interesting probes to display
        display_indices = list(range(min(4, K)))
        ts = np.arange(T)

        for pi, pidx in enumerate(display_indices):
            label = probe_names[pidx]
            # Ground-truth labels
            ax_probe.plot(ts, probe_labels_sample[pidx], "-",
                          color=f"C{pi}", linewidth=1.5, alpha=0.5,
                          label=f"{label} (true)")
            # Probe applied to GT embeddings
            ax_probe.plot(ts, gt_probe[:, pidx], "o",
                          color=f"C{pi}", markersize=3,
                          label=f"{label} (enc)")
            # Probe applied to predicted embeddings
            ax_probe.plot(ts, pred_probe[:, pidx], "x",
                          color=f"C{pi}", markersize=4,
                          label=f"{label} (pred)")

        ax_probe.set_xlabel("Timestep", fontsize=10)
        ax_probe.set_ylabel("Probe Value", fontsize=10)
        ax_probe.set_title("Probe Predictions: True vs Encoded vs Predicted", fontsize=11)
        ax_probe.legend(fontsize=7, ncol=3, loc="upper right")
        ax_probe.grid(True, alpha=0.3)

    fig.suptitle(f"Dreaming Visualization  --  Sample {sample_idx}", fontsize=13, y=1.01)

    out_path = os.path.join(output_dir, f"dreaming_sample_{sample_idx}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved dreaming visualization to {out_path}")


def _get_crafter_probe_names(K):
    """Return human-readable names for Crafter probe labels (16 dimensions).

    Order matches eval_probe.py and the data collection script.
    """
    names = [
        # Vitals (0-3)
        "health",
        "food",
        "drink",
        "energy",
        # Resources (4-9)
        "sapling",
        "wood",
        "stone",
        "coal",
        "iron",
        "diamond",
        # Tools (10-15)
        "wood_pickaxe",
        "stone_pickaxe",
        "iron_pickaxe",
        "wood_sword",
        "stone_sword",
        "iron_sword",
    ]
    if K <= len(names):
        return names[:K]
    return names + [f"probe_{i}" for i in range(len(names), K)]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results):
    """Print a summary table of rollout evaluation results."""
    mean_mse = results["mean_mse_per_step"]
    std_mse = results["std_mse_per_step"]
    mean_copy = results["mean_copy_baseline_per_step"]

    print("\n" + "=" * 65)
    print("  ROLLOUT EVALUATION SUMMARY")
    print("=" * 65)
    print(f"  Total trajectories evaluated: {results['n_samples']}")
    print(f"  Horizon steps: {len(mean_mse)}")
    print("-" * 65)
    print(f"  {'Step':>4}  {'JEPA MSE':>12}  {'Std':>10}  {'Copy MSE':>12}  {'Ratio':>8}")
    print("-" * 65)
    for i in range(len(mean_mse)):
        ratio = mean_mse[i] / (mean_copy[i] + 1e-8)
        print(f"  {i+1:>4}  {mean_mse[i]:>12.6f}  {std_mse[i]:>10.6f}  "
              f"{mean_copy[i]:>12.6f}  {ratio:>8.4f}")
    print("-" * 65)
    print(f"  Mean JEPA MSE (all steps): {mean_mse.mean():.6f}")
    print(f"  Mean Copy MSE (all steps): {mean_copy.mean():.6f}")
    print(f"  Overall ratio (JEPA/Copy):  {mean_mse.mean() / (mean_copy.mean() + 1e-8):.4f}")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate JEPA rollout quality on Crafter trajectories."
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to a trained JEPA checkpoint (.pth.tar)."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to directory with Crafter trajectory .npz files."
    )
    parser.add_argument(
        "--output_dir", type=str, default="eval_results",
        help="Directory to save results and figures."
    )
    parser.add_argument(
        "--num_batches", type=int, default=50,
        help="Number of validation batches to evaluate."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--probe_path", type=str, default=None,
        help="(Optional) Path to a trained probe checkpoint for dreaming visualization."
    )
    parser.add_argument(
        "--num_dream_samples", type=int, default=5,
        help="Number of dreaming visualization samples to generate."
    )
    parser.add_argument(
        "--sample_length", type=int, default=17,
        help="Trajectory length (must match training config)."
    )
    # Model architecture args (defaults match crafter.yaml)
    parser.add_argument("--dobs", type=int, default=3)
    parser.add_argument("--henc", type=int, default=32)
    parser.add_argument("--dstc", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--num_actions", type=int, default=17)
    parser.add_argument("--d_action_emb", type=int, default=32)

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device: {device}")
    print(f"Output directory: {args.output_dir}")

    # Build model
    print("\nBuilding JEPA model...")
    jepa = build_jepa_model(
        dobs=args.dobs,
        henc=args.henc,
        dstc=args.dstc,
        img_size=args.img_size,
        num_actions=args.num_actions,
        d_action_emb=args.d_action_emb,
        device=device,
    )

    # Load checkpoint
    print("Loading checkpoint...")
    load_checkpoint(jepa, args.checkpoint_path, device)
    jepa.eval()

    # Build data loader
    print("\nBuilding validation data loader...")
    val_loader, val_dset = build_val_loader(
        args.data_dir, args.batch_size, sample_length=args.sample_length
    )

    # ---- Part A: Rollout MSE evaluation ----
    print("\n--- Part A: Rollout MSE Evaluation ---")
    effective_batches = min(args.num_batches, len(val_loader))
    print(f"Evaluating {effective_batches} batches...")

    results = evaluate_rollout_mse(jepa, val_loader, args.num_batches, device)

    print_summary(results)

    # Plot
    plot_path = os.path.join(args.output_dir, "rollout_mse_vs_horizon.png")
    plot_mse_vs_horizon(results, plot_path)

    # Save results JSON
    json_path = os.path.join(args.output_dir, "rollout_results.json")
    json_results = {
        "mean_mse_per_step": results["mean_mse_per_step"].tolist(),
        "std_mse_per_step": results["std_mse_per_step"].tolist(),
        "mean_copy_baseline_per_step": results["mean_copy_baseline_per_step"].tolist(),
        "std_copy_baseline_per_step": results["std_copy_baseline_per_step"].tolist(),
        "n_samples": int(results["n_samples"]),
        "overall_mean_mse": float(results["mean_mse_per_step"].mean()),
        "overall_mean_copy_mse": float(results["mean_copy_baseline_per_step"].mean()),
        "overall_ratio": float(
            results["mean_mse_per_step"].mean()
            / (results["mean_copy_baseline_per_step"].mean() + 1e-8)
        ),
    }
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved results JSON to {json_path}")

    # ---- Part B: Dreaming visualizations ----
    print(f"\n--- Part B: Dreaming Visualizations ({args.num_dream_samples} samples) ---")

    # Rebuild loader to iterate from the start for dreaming
    dream_loader, _ = build_val_loader(
        args.data_dir, args.batch_size, sample_length=args.sample_length
    )
    generate_dreaming_visualizations(
        jepa,
        dream_loader,
        args.output_dir,
        device,
        num_samples=args.num_dream_samples,
        probe_path=args.probe_path,
    )

    print(f"\nAll done. Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
