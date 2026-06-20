#!/usr/bin/env python3
"""
Visualize how a JEPA world model learns to predict over training epochs.

Takes a FIXED test trajectory and, for each training epoch checkpoint (e-0 through e-11),
computes autoregressive rollout predictions vs ground-truth encodings. Produces:

  1. learning_progress.gif  -- animated GIF (one frame per epoch) showing actual frames,
     per-step MSE bars (shrinking over epochs), and a text summary.
  2. learning_curve.png     -- mean rollout MSE vs training epoch.
  3. epoch_XX.png           -- individual PNG frames for each epoch.

Usage:
    python scripts/visualize_learning.py \
        --checkpoint_dir checkpoints/.../idm1_seed1000/ \
        --data_dir data/crafter_trajectories \
        --output_dir eval_results/learning_viz
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path
# ---------------------------------------------------------------------------
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
from eb_jepa.datasets.utils import _SubsetTrajDataset

# ---------------------------------------------------------------------------
# Probe label names (matches eval_probe.py)
# ---------------------------------------------------------------------------
LABEL_NAMES = [
    "health", "food", "drink", "energy",
    "sapling", "wood", "stone", "coal", "iron", "diamond",
    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "wood_sword", "stone_sword", "iron_sword",
]

# ---------------------------------------------------------------------------
# Model construction (mirrors eval_rollout.py / eval_probe.py)
# ---------------------------------------------------------------------------

def build_jepa_model(device: torch.device = torch.device("cpu")) -> JEPA:
    """Reconstruct the JEPA model architecture used for Crafter training."""
    dobs = 3
    henc = 32
    dstc = 32
    img_size = 64
    num_actions = 17
    d_action_emb = 32

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

    # Determine spatial dims for IDM
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


def load_checkpoint(jepa: JEPA, path: str, device: torch.device):
    """Load a training checkpoint into the JEPA model."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    jepa.load_state_dict(state_dict, strict=True)
    epoch = checkpoint.get("epoch", "?")
    step = checkpoint.get("step", "?")
    print(f"  Loaded {path}  (epoch={epoch}, step={step})")
    return checkpoint


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def build_normalizer_and_loader(data_dir: str, sample_length: int = 17):
    """Build train/val split and return the normalizer + val dataset."""
    traj_dataset = CrafterTrajDataset(data_dir=data_dir, sample_length=sample_length)

    num_episodes = len(traj_dataset)
    num_train = int(0.9 * num_episodes)
    indices = torch.randperm(num_episodes, generator=torch.Generator().manual_seed(42))
    train_indices = indices[:num_train].tolist()
    val_indices = indices[num_train:].tolist()

    train_traj = _SubsetTrajDataset(traj_dataset, train_indices)
    val_traj = _SubsetTrajDataset(traj_dataset, val_indices)

    # Build train sliced dataset to compute normalizer
    train_dset = CrafterSlicedDataset(
        train_traj, sample_length=sample_length, num_stats_samples=5000
    )
    normalizer = train_dset.normalizer

    return traj_dataset, val_traj, normalizer


def prepare_episode(traj_dataset, episode_idx: int, normalizer, sample_length: int,
                    device: torch.device):
    """Prepare a single episode for evaluation.

    Returns:
        obs_tensor:   [1, 3, T, 64, 64] normalized
        actions_tensor: [1, T] long
        obs_raw:      [T, 64, 64, 3] uint8 for display
        probe_labels: [T, 16] float32
    """
    episode = traj_dataset[episode_idx]
    T_total = episode["observations"].shape[0]
    T = min(sample_length, T_total)

    obs_raw = episode["observations"][:T]        # [T, 64, 64, 3] uint8
    actions = episode["actions"][:T]              # [T]
    probe_labels = episode["probe_labels"][:T]    # [T, 16]

    # Convert to tensor: [T, H, W, C] -> [C, T, H, W] float32 [0,1]
    obs_float = torch.from_numpy(obs_raw.copy()).float() / 255.0
    obs_float = obs_float.permute(3, 0, 1, 2)  # [C, T, H, W]
    obs_float = normalizer.normalize_state(obs_float)  # normalized
    obs_tensor = obs_float.unsqueeze(0).to(device)  # [1, C, T, H, W]

    actions_tensor = torch.from_numpy(actions.copy()).long().unsqueeze(0).to(device)  # [1, T]

    return obs_tensor, actions_tensor, obs_raw, probe_labels


# ---------------------------------------------------------------------------
# Core evaluation: compute MSE for one checkpoint on one or more episodes
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_epoch(ckpt_path, episodes_data, device):
    """Load a checkpoint and compute per-step MSE for each episode.

    Args:
        ckpt_path: path to the .pth.tar checkpoint
        episodes_data: list of (obs_tensor, actions_tensor) tuples
        device: torch device

    Returns:
        dict with 'mse_per_step' (list of arrays), 'mean_mse_per_step', 'mean_mse'
    """
    jepa = build_jepa_model(device)
    load_checkpoint(jepa, ckpt_path, device)
    jepa.eval()

    all_mse = []
    for obs_tensor, actions_tensor in episodes_data:
        B, C, T, H, W = obs_tensor.shape
        nsteps = T - 1

        # Ground truth encoding
        gt_encoded = jepa.encode(obs_tensor)  # [1, 512, T, 1, 1]

        # Predicted encoding from first frame + actions
        obs_init = obs_tensor[:, :, 0:1]  # [1, C, 1, H, W]
        predicted, _ = jepa.unroll(
            obs_init,
            actions_tensor,
            nsteps=nsteps,
            unroll_mode="autoregressive",
            compute_loss=False,
        )  # [1, 512, T, 1, 1]

        # Per-timestep MSE (skip first frame = context)
        mse_per_step = ((gt_encoded[:, :, 1:] - predicted[:, :, 1:]) ** 2).mean(
            dim=(1, 3, 4)
        ).squeeze(0)  # [T-1]
        all_mse.append(mse_per_step.cpu().numpy())

    # Free memory
    del jepa
    if device.type == "cuda":
        torch.cuda.empty_cache()

    mean_mse_per_step = np.mean(all_mse, axis=0)  # [T-1]

    return {
        "mse_per_step_list": all_mse,          # list of [T-1] arrays
        "mean_mse_per_step": mean_mse_per_step,  # [T-1]
        "mean_mse": float(mean_mse_per_step.mean()),
    }


# ---------------------------------------------------------------------------
# Visualization: create one GIF frame for a given epoch
# ---------------------------------------------------------------------------

def create_epoch_frame(obs_raw, result, epoch, global_max_mse):
    """Create one frame of the GIF for a given epoch.

    Args:
        obs_raw:        [T, 64, 64, 3] uint8 raw observation frames
        result:         dict from evaluate_epoch (uses 'mean_mse_per_step', 'mean_mse')
        epoch:          epoch number
        global_max_mse: fixed y-axis upper bound across all epochs

    Returns:
        matplotlib Figure
    """
    mse = result["mean_mse_per_step"]
    mean_mse = result["mean_mse"]
    T = obs_raw.shape[0]

    fig = plt.figure(figsize=(16, 10), facecolor="white")
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.8, 1.5, 0.7], hspace=0.35)

    # --- Row 1: Actual Crafter frames ---
    show_indices = [i for i in [0, 2, 4, 6, 8, 10, 12, 14, 16] if i < T]
    n_frames = len(show_indices)
    gs_frames = gridspec.GridSpecFromSubplotSpec(1, n_frames, subplot_spec=gs[0], wspace=0.05)

    for j, t_idx in enumerate(show_indices):
        ax = fig.add_subplot(gs_frames[0, j])
        ax.imshow(obs_raw[t_idx])
        ax.set_title(f"t={t_idx}", fontsize=9, fontweight="bold")
        ax.axis("off")

    # Add epoch label above the frames
    fig.text(0.5, 0.97, f"Epoch {epoch}/11 -- World Model Prediction Quality",
             fontsize=16, fontweight="bold", ha="center", va="top",
             color="#1a1a2e")

    # --- Row 2: Per-step MSE bars ---
    ax_mse = fig.add_subplot(gs[1])
    horizon_steps = np.arange(1, len(mse) + 1)

    # Color bars: green for low, red for high (relative to global max)
    norm_vals = mse / max(global_max_mse, 1e-8)
    norm_vals = np.clip(norm_vals, 0, 1)
    colors = plt.cm.RdYlGn_r(norm_vals)

    bars = ax_mse.bar(horizon_steps, mse, color=colors, edgecolor="gray",
                      linewidth=0.5, width=0.7)
    ax_mse.set_xlabel("Prediction Horizon (steps ahead)", fontsize=12)
    ax_mse.set_ylabel("Latent MSE", fontsize=12)
    ax_mse.set_title(
        f"Predicted vs Actual Latent Error  (mean = {mean_mse:.4f})",
        fontsize=13, fontweight="bold"
    )
    ax_mse.set_ylim(0, global_max_mse * 1.15)  # FIXED y-axis across all epochs
    ax_mse.set_xticks(horizon_steps)
    ax_mse.grid(axis="y", alpha=0.3)

    # Add value labels on top of bars
    for bar, val in zip(bars, mse):
        if val > global_max_mse * 0.05:
            ax_mse.text(bar.get_x() + bar.get_width() / 2., val + global_max_mse * 0.02,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7, color="gray")

    # --- Row 3: Summary text ---
    ax_text = fig.add_subplot(gs[2])
    ax_text.axis("off")

    # Progress bar visualization
    if len(mse) > 0:
        summary_text = (
            f"Epoch {epoch}/11   |   "
            f"Mean MSE: {mean_mse:.4f}   |   "
            f"Step-1 MSE: {mse[0]:.4f}   |   "
            f"Step-{len(mse)} MSE: {mse[-1]:.4f}"
        )
    else:
        summary_text = f"Epoch {epoch}/11   |   No data"

    # Draw a progress bar for epoch
    ax_text.text(0.5, 0.7, summary_text, transform=ax_text.transAxes,
                 fontsize=14, ha="center", va="center", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f4fd",
                           edgecolor="#2563eb", alpha=0.9))

    # Epoch progress indicator
    progress = (epoch + 1) / 12
    ax_text.barh(0.2, progress, height=0.15, color="#2563eb", alpha=0.7,
                 transform=ax_text.transAxes)
    ax_text.barh(0.2, 1.0, height=0.15, color="#e0e0e0", alpha=0.3,
                 transform=ax_text.transAxes)
    ax_text.text(0.5, 0.2, f"Training Progress: {progress*100:.0f}%",
                 transform=ax_text.transAxes, fontsize=10, ha="center", va="center")

    plt.subplots_adjust(top=0.93)
    return fig


# ---------------------------------------------------------------------------
# Summary learning curve plot
# ---------------------------------------------------------------------------

def plot_learning_curve(results_per_epoch, output_path):
    """Plot mean rollout MSE vs training epoch."""
    epochs = [r["epoch"] for r in results_per_epoch]
    mean_mses = [r["mean_mse"] for r in results_per_epoch]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Main line
    ax.plot(epochs, mean_mses, "o-", color="#2563eb", linewidth=2.5,
            markersize=10, markerfacecolor="white", markeredgewidth=2,
            markeredgecolor="#2563eb", zorder=5)

    # Fill under curve
    ax.fill_between(epochs, mean_mses, alpha=0.1, color="#2563eb")

    # Annotations for first and last
    ax.annotate(f"{mean_mses[0]:.4f}", (epochs[0], mean_mses[0]),
                textcoords="offset points", xytext=(15, 10),
                fontsize=11, color="#dc2626", fontweight="bold")
    ax.annotate(f"{mean_mses[-1]:.4f}", (epochs[-1], mean_mses[-1]),
                textcoords="offset points", xytext=(-40, 10),
                fontsize=11, color="#059669", fontweight="bold")

    # Improvement arrow
    if len(mean_mses) >= 2:
        improvement = (mean_mses[0] - mean_mses[-1]) / mean_mses[0] * 100
        ax.text(0.95, 0.95,
                f"Improvement: {improvement:.1f}%\n"
                f"({mean_mses[0]:.4f} -> {mean_mses[-1]:.4f})",
                transform=ax.transAxes, fontsize=11, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#ecfdf5",
                          edgecolor="#059669", alpha=0.9))

    ax.set_xlabel("Training Epoch", fontsize=14)
    ax.set_ylabel("Mean Rollout MSE (latent space)", fontsize=14)
    ax.set_title("World Model Learns to Predict: Rollout Error vs Training Epoch",
                 fontsize=16, fontweight="bold")
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved learning curve to {output_path}")


# ---------------------------------------------------------------------------
# Per-horizon heatmap across epochs
# ---------------------------------------------------------------------------

def plot_horizon_heatmap(results_per_epoch, output_path):
    """Plot a heatmap of MSE[epoch, horizon_step]."""
    epochs = [r["epoch"] for r in results_per_epoch]
    matrix = np.array([r["mean_mse_per_step"] for r in results_per_epoch])
    # matrix shape: [num_epochs, T-1]

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Prediction Horizon Step", fontsize=13)
    ax.set_ylabel("Training Epoch", fontsize=13)
    ax.set_title("Prediction Error Heatmap: Epoch x Horizon", fontsize=15, fontweight="bold")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(range(1, matrix.shape[1] + 1))
    ax.set_yticks(range(len(epochs)))
    ax.set_yticklabels(epochs)
    plt.colorbar(im, ax=ax, label="Latent MSE")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved horizon heatmap to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize JEPA world model learning over training epochs."
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True,
        help="Directory containing e-0.pth.tar through e-11.pth.tar checkpoints."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to directory with Crafter trajectory .npz files."
    )
    parser.add_argument(
        "--output_dir", type=str, default="eval_results/learning_viz",
        help="Where to save GIF, plots, and per-epoch PNGs."
    )
    parser.add_argument(
        "--episode_idx", type=int, default=0,
        help="Which episode to visualize in the GIF (default: 0)."
    )
    parser.add_argument(
        "--num_samples", type=int, default=5,
        help="How many test trajectories to average over for the summary plot (default: 5)."
    )
    parser.add_argument(
        "--sample_length", type=int, default=17,
        help="Trajectory length to use (default: 17)."
    )
    parser.add_argument(
        "--max_epoch", type=int, default=11,
        help="Maximum epoch index to look for (default: 11, meaning e-0 through e-11)."
    )
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # ------------------------------------------------------------------
    # 1. Discover available checkpoints
    # ------------------------------------------------------------------
    print("\n--- Discovering checkpoints ---")
    available_epochs = []
    for epoch in range(args.max_epoch + 1):
        ckpt_path = os.path.join(args.checkpoint_dir, f"e-{epoch}.pth.tar")
        if os.path.exists(ckpt_path):
            available_epochs.append(epoch)
            print(f"  Found: e-{epoch}.pth.tar")
        else:
            print(f"  Missing: e-{epoch}.pth.tar (skipping)")

    if len(available_epochs) == 0:
        print("ERROR: No epoch checkpoints found. Check --checkpoint_dir.")
        sys.exit(1)

    print(f"\nWill evaluate {len(available_epochs)} epoch(s): {available_epochs}")

    # ------------------------------------------------------------------
    # 2. Load data
    # ------------------------------------------------------------------
    print("\n--- Loading data ---")
    traj_dataset, val_traj, normalizer = build_normalizer_and_loader(
        args.data_dir, sample_length=args.sample_length
    )

    # Prepare the display episode (for GIF frames)
    print(f"\nPreparing display episode (idx={args.episode_idx})...")
    obs_tensor_display, actions_tensor_display, obs_raw_display, probe_labels_display = \
        prepare_episode(traj_dataset, args.episode_idx, normalizer,
                        args.sample_length, device)
    print(f"  obs shape: {obs_tensor_display.shape}, actions shape: {actions_tensor_display.shape}")

    # Prepare multiple episodes for averaging (from val set)
    num_val_episodes = len(val_traj)
    num_avg = min(args.num_samples, num_val_episodes)
    print(f"\nPreparing {num_avg} validation episodes for averaging...")
    avg_episodes_data = []
    for i in range(num_avg):
        obs_t, act_t, _, _ = prepare_episode(val_traj, i, normalizer,
                                              args.sample_length, device)
        avg_episodes_data.append((obs_t, act_t))

    # Also include the display episode in its own list
    display_episode_data = [(obs_tensor_display, actions_tensor_display)]

    # ------------------------------------------------------------------
    # 3. Evaluate each epoch
    # ------------------------------------------------------------------
    print("\n--- Evaluating each epoch ---")
    results_per_epoch = []

    for epoch in available_epochs:
        ckpt_path = os.path.join(args.checkpoint_dir, f"e-{epoch}.pth.tar")
        print(f"\nEpoch {epoch}:")

        # Evaluate on display episode (for GIF)
        display_result = evaluate_epoch(ckpt_path, display_episode_data, device)

        # Evaluate on multiple episodes (for averaged learning curve)
        avg_result = evaluate_epoch(ckpt_path, avg_episodes_data, device)

        results_per_epoch.append({
            "epoch": epoch,
            # For GIF: per-step MSE from display episode
            "display_mse_per_step": display_result["mean_mse_per_step"],
            "display_mean_mse": display_result["mean_mse"],
            # For learning curve: averaged over multiple episodes
            "mean_mse_per_step": avg_result["mean_mse_per_step"],
            "mean_mse": avg_result["mean_mse"],
        })

        print(f"  Display MSE: {display_result['mean_mse']:.4f}  |  "
              f"Avg MSE ({num_avg} episodes): {avg_result['mean_mse']:.4f}")

    # ------------------------------------------------------------------
    # 4. Create GIF frames
    # ------------------------------------------------------------------
    print("\n--- Creating GIF frames ---")

    # Compute global max MSE across all epochs for fixed y-axis
    global_max_mse = max(r["display_mean_mse"] for r in results_per_epoch)
    global_max_mse_per_step = max(
        r["display_mse_per_step"].max() for r in results_per_epoch
    )
    print(f"  Global max MSE (for fixed y-axis): {global_max_mse_per_step:.4f}")

    frames = []
    for result in results_per_epoch:
        epoch = result["epoch"]
        frame_result = {
            "mean_mse_per_step": result["display_mse_per_step"],
            "mean_mse": result["display_mean_mse"],
        }

        fig = create_epoch_frame(obs_raw_display, frame_result, epoch,
                                 global_max_mse_per_step)

        # Save individual frame as PNG
        frame_path = os.path.join(args.output_dir, f"epoch_{epoch:02d}.png")
        fig.savefig(frame_path, dpi=120, bbox_inches="tight", facecolor="white")
        print(f"  Saved {frame_path}")

        # Convert figure to RGB array for GIF
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        try:
            buf = fig.canvas.tostring_rgb()
        except AttributeError:
            buf = fig.canvas.buffer_rgba()
            img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))[:, :, :3]
        else:
            img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3))
        frames.append(img)

        plt.close(fig)

    # Save GIF
    try:
        import imageio
        gif_path = os.path.join(args.output_dir, "learning_progress.gif")
        imageio.mimsave(gif_path, frames, duration=1000, loop=0)
        print(f"\nSaved GIF to {gif_path}")
    except ImportError:
        print("\nWARNING: imageio not installed. Trying with PIL instead...")
        try:
            from PIL import Image
            gif_path = os.path.join(args.output_dir, "learning_progress.gif")
            pil_frames = [Image.fromarray(f) for f in frames]
            pil_frames[0].save(
                gif_path, save_all=True, append_images=pil_frames[1:],
                duration=1000, loop=0
            )
            print(f"Saved GIF to {gif_path}")
        except ImportError:
            print("ERROR: Neither imageio nor PIL available. GIF not saved.")
            print("Install with: pip install imageio  or  pip install Pillow")

    # ------------------------------------------------------------------
    # 5. Create summary plots
    # ------------------------------------------------------------------
    print("\n--- Creating summary plots ---")

    # Learning curve (averaged MSE vs epoch)
    curve_path = os.path.join(args.output_dir, "learning_curve.png")
    plot_learning_curve(results_per_epoch, curve_path)

    # Horizon heatmap
    heatmap_path = os.path.join(args.output_dir, "horizon_heatmap.png")
    plot_horizon_heatmap(results_per_epoch, heatmap_path)

    # ------------------------------------------------------------------
    # 6. Save numerical results as JSON
    # ------------------------------------------------------------------
    json_results = {
        "checkpoint_dir": args.checkpoint_dir,
        "data_dir": args.data_dir,
        "episode_idx": args.episode_idx,
        "num_avg_episodes": num_avg,
        "sample_length": args.sample_length,
        "epochs_evaluated": available_epochs,
        "per_epoch": [
            {
                "epoch": r["epoch"],
                "display_mean_mse": r["display_mean_mse"],
                "avg_mean_mse": r["mean_mse"],
                "display_mse_per_step": r["display_mse_per_step"].tolist(),
                "avg_mse_per_step": r["mean_mse_per_step"].tolist(),
            }
            for r in results_per_epoch
        ],
    }
    json_path = os.path.join(args.output_dir, "learning_results.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved results JSON to {json_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  LEARNING VISUALIZATION SUMMARY")
    print("=" * 65)
    if len(results_per_epoch) >= 2:
        first = results_per_epoch[0]
        last = results_per_epoch[-1]
        improvement = (first["mean_mse"] - last["mean_mse"]) / first["mean_mse"] * 100
        print(f"  Epochs evaluated:  {len(results_per_epoch)}")
        print(f"  First epoch MSE:   {first['mean_mse']:.4f} (epoch {first['epoch']})")
        print(f"  Last epoch MSE:    {last['mean_mse']:.4f} (epoch {last['epoch']})")
        print(f"  Improvement:       {improvement:.1f}%")
    else:
        print(f"  Only {len(results_per_epoch)} epoch(s) evaluated.")

    print(f"\n  Outputs:")
    print(f"    GIF:           {args.output_dir}/learning_progress.gif")
    print(f"    Learning curve:{args.output_dir}/learning_curve.png")
    print(f"    Heatmap:       {args.output_dir}/horizon_heatmap.png")
    print(f"    Per-epoch PNGs:{args.output_dir}/epoch_XX.png")
    print(f"    Results JSON:  {args.output_dir}/learning_results.json")
    print("=" * 65)


if __name__ == "__main__":
    main()
