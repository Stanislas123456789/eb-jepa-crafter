#!/usr/bin/env python3
"""
KILLER DEMO: JEPA World Model Prediction Overlay Visualization.

Shows actual Crafter game frames with the model's IMAGINED game state
(health, food, energy, wood) decoded from predicted latents via a linear probe.
Animates across training epochs to show the world model IMPROVING.

Outputs:
  1. overlay_demo.gif       -- Animated GIF across epochs with prediction overlays
  2. prediction_accuracy.png -- Probe predictions vs actuals for final model
  3. epoch_comparison.png   -- Side-by-side epoch rows showing improvement

Usage:
    python scripts/visualize_overlay.py \
        --checkpoint_dir checkpoints/.../idm1_seed1000/ \
        --data_dir data/crafter_trajectories \
        --output_dir eval_results/overlay_viz \
        --epochs_to_show 0,2,4,6,8,10,11
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

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
# Label config
# ---------------------------------------------------------------------------
LABEL_NAMES = [
    "health", "food", "drink", "energy",
    "sapling", "wood", "stone", "coal", "iron", "diamond",
    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "wood_sword", "stone_sword", "iron_sword",
]

# The 4 key features we visualize (index into the 16-dim probe output)
KEY_FEATURES = {
    "health": 0,
    "food": 1,
    "energy": 3,
    "wood": 5,
}

# Colors for each feature
FEATURE_COLORS = {
    "health": "#e74c3c",   # red
    "food": "#f39c12",     # orange
    "energy": "#3498db",   # blue
    "wood": "#8B4513",     # brown
}

FEATURE_ICONS = {
    "health": "\u2764",   # heart
    "food": "\u2615",     # food
    "energy": "\u26a1",   # lightning
    "wood": "\u2692",     # wood/hammer
}


# ---------------------------------------------------------------------------
# Model construction (same as visualize_learning.py)
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

def build_normalizer_and_dataset(data_dir: str, sample_length: int = 17):
    """Build trajectory dataset and normalizer."""
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

    return traj_dataset, train_traj, val_traj, normalizer


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
# Probe training
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_encoded_features(jepa: JEPA, traj_dataset, normalizer, device,
                             num_episodes=50, sample_length=17):
    """Encode episodes with the JEPA encoder and collect features + labels for probe training.

    Returns:
        features: [N, 512] tensor
        labels: [N, 16] tensor
    """
    jepa.eval()
    all_features = []
    all_labels = []

    num_available = len(traj_dataset)
    num_episodes = min(num_episodes, num_available)

    for ep_idx in range(num_episodes):
        episode = traj_dataset[ep_idx]
        T_total = episode["observations"].shape[0]
        T = min(sample_length, T_total)

        obs_raw = episode["observations"][:T]
        probe_labels = episode["probe_labels"][:T]  # [T, 16]

        # Normalize and encode
        obs_float = torch.from_numpy(obs_raw.copy()).float() / 255.0
        obs_float = obs_float.permute(3, 0, 1, 2)  # [C, T, H, W]
        obs_float = normalizer.normalize_state(obs_float)
        obs_tensor = obs_float.unsqueeze(0).to(device)  # [1, C, T, H, W]

        # Encode: [1, 512, T, 1, 1]
        enc = jepa.encode(obs_tensor)
        # Flatten to [T, 512]
        enc_flat = enc.squeeze(0).squeeze(-1).squeeze(-1).permute(1, 0)  # [T, 512]

        all_features.append(enc_flat.cpu())
        all_labels.append(torch.from_numpy(probe_labels.copy()).float())

    features = torch.cat(all_features, dim=0)  # [N, 512]
    labels = torch.cat(all_labels, dim=0)       # [N, 16]
    return features, labels


def train_linear_probe(features, labels, device, epochs=10, lr=1e-3, batch_size=256):
    """Train a linear probe on features -> labels.

    Returns the trained probe (nn.Linear(512, 16)).
    """
    feat_dim = features.shape[1]
    num_targets = labels.shape[1]
    probe = nn.Linear(feat_dim, num_targets).to(device)
    optimizer = AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    n = features.shape[0]
    n_batches = max(1, n // batch_size)

    for epoch in range(1, epochs + 1):
        probe.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0

        for i in range(n_batches):
            idx = perm[i * batch_size: (i + 1) * batch_size]
            x = features[idx].to(device)
            y = labels[idx].to(device)

            pred = probe(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 3 == 0 or epoch == 1:
            print(f"    Probe epoch {epoch}/{epochs} | Train MSE: {epoch_loss / n_batches:.6f}")

    probe.eval()
    return probe


# ---------------------------------------------------------------------------
# Core: Imagined predictions from JEPA
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_imagined_predictions(jepa, probe, obs_tensor, actions_tensor, device):
    """Run JEPA imagination and probe to get predicted game state at each timestep.

    Args:
        jepa: JEPA model
        probe: nn.Linear(512, 16) trained probe
        obs_tensor: [1, C, T, H, W] normalized observations
        actions_tensor: [1, T] actions

    Returns:
        predictions: [T, 16] numpy array of predicted probe labels
            Index 0 corresponds to encoding the first frame (not imagined).
            Indices 1..T-1 correspond to imagined latents.
    """
    jepa.eval()
    probe.eval()

    B, C, T, H, W = obs_tensor.shape
    nsteps = T - 1

    # 1. Encode first frame only
    obs_init = obs_tensor[:, :, 0:1]  # [1, C, 1, H, W]

    # 2. Unroll predictor autoregressively with real actions
    predicted_latents, _ = jepa.unroll(
        obs_init,
        actions_tensor,
        nsteps=nsteps,
        unroll_mode="autoregressive",
        compute_loss=False,
    )
    # predicted_latents: [1, 512, 1+nsteps, 1, 1] = [1, 512, T, 1, 1]

    # 3. Apply probe to each predicted latent
    # Flatten: [1, 512, T, 1, 1] -> [T, 512]
    latents = predicted_latents.squeeze(0).squeeze(-1).squeeze(-1).permute(1, 0)  # [T, 512]
    latents = latents.to(device)
    predictions = probe(latents)  # [T, 16]

    return predictions.cpu().numpy()


# ---------------------------------------------------------------------------
# Overlay visualization
# ---------------------------------------------------------------------------

def overlay_predictions_on_frame(frame, predicted_vals, actual_vals, timestep,
                                 scale=4, show_bars=True):
    """Create an annotated frame with prediction overlays using matplotlib.

    Args:
        frame: [H, W, 3] uint8 numpy array (64x64 Crafter frame)
        predicted_vals: dict {'health': 7.2, 'food': 5.1, ...}
        actual_vals: dict {'health': 7, 'food': 5, ...}
        timestep: int, for labeling
        scale: upscale factor (64 -> 256 at scale=4)
        show_bars: whether to show health/energy bars

    Returns:
        fig: matplotlib figure of the annotated frame
    """
    H, W = frame.shape[:2]
    fig_h = scale * H / 64  # inches
    fig_w = scale * W / 64

    fig, ax = plt.subplots(figsize=(fig_w + 0.8, fig_h + 1.6), facecolor="#1a1a2e")
    ax.imshow(frame, interpolation="nearest", aspect="equal")
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.axis("off")

    if show_bars:
        # Draw health bar at top of frame
        health_pred = predicted_vals.get("health", 0)
        health_max = 9.0
        bar_y = 1
        bar_h = 3
        bar_x = 2
        bar_w = W - 4

        # Background (dark)
        rect_bg = mpatches.FancyBboxPatch(
            (bar_x, bar_y), bar_w, bar_h,
            boxstyle="round,pad=0.3", facecolor="#333333", alpha=0.7,
            edgecolor="white", linewidth=0.5
        )
        ax.add_patch(rect_bg)

        # Health fill
        fill_w = max(0, min(bar_w, bar_w * (health_pred / health_max)))
        rect_fill = mpatches.FancyBboxPatch(
            (bar_x, bar_y), fill_w, bar_h,
            boxstyle="round,pad=0.3", facecolor="#e74c3c", alpha=0.85,
            edgecolor="none"
        )
        ax.add_patch(rect_fill)

    # Title with timestep
    ax.set_title(f"t={timestep}", fontsize=10, fontweight="bold",
                 color="white", pad=4)

    # Build prediction text below frame
    lines = []
    for feat_name in ["health", "food", "energy", "wood"]:
        pred_v = predicted_vals.get(feat_name, 0)
        act_v = actual_vals.get(feat_name, 0)
        diff = abs(pred_v - act_v)

        # Color code accuracy
        if diff <= 1.0:
            color = "#2ecc71"  # green = good
        elif diff <= 2.0:
            color = "#f39c12"  # orange = close
        else:
            color = "#e74c3c"  # red = wrong

        short = feat_name[0].upper()
        lines.append((f"{short}:{pred_v:.0f}/{act_v:.0f}", color))

    # Add text below the image
    text_y = -0.05
    for i, (txt, color) in enumerate(lines):
        x_pos = 0.02 + i * 0.25
        fig.text(x_pos + 0.05, 0.02, txt, fontsize=7, color=color,
                 fontweight="bold", transform=fig.transFigure,
                 fontfamily="monospace")

    plt.subplots_adjust(top=0.88, bottom=0.15, left=0.02, right=0.98)
    return fig


def create_frame_strip(obs_raw, predicted_all, probe_labels, epoch,
                       show_indices=None, total_epochs=11):
    """Create a strip of annotated frames for one epoch.

    Args:
        obs_raw: [T, 64, 64, 3] uint8 raw frames
        predicted_all: [T, 16] numpy predictions from imagined latent
        probe_labels: [T, 16] numpy ground truth
        epoch: epoch number
        show_indices: which timesteps to show (default: every 2nd up to 8 frames)
        total_epochs: total number of training epochs

    Returns:
        fig: matplotlib figure of the full strip
    """
    T = obs_raw.shape[0]
    if show_indices is None:
        show_indices = [i for i in range(0, T, 2)][:8]

    n_frames = len(show_indices)

    # Create figure: frames in a row with annotations below
    fig = plt.figure(figsize=(n_frames * 3.2, 5.5), facecolor="#0d1117")

    # Main title
    fig.suptitle(
        f"Epoch {epoch}/{total_epochs} -- World Model Predictions from Imagination",
        fontsize=16, fontweight="bold", color="white", y=0.97
    )

    # Subtitle explaining the viz
    fig.text(0.5, 0.92,
             "Pred/Actual  |  Green = accurate (+/-1)  Orange = close (+/-2)  Red = off",
             fontsize=9, color="#8b949e", ha="center", fontfamily="monospace")

    gs = gridspec.GridSpec(2, n_frames, height_ratios=[3, 1.2], hspace=0.05,
                           wspace=0.08, top=0.88, bottom=0.04, left=0.02, right=0.98)

    for j, t_idx in enumerate(show_indices):
        # --- Top row: Crafter frame with health bar overlay ---
        ax_img = fig.add_subplot(gs[0, j])
        ax_img.imshow(obs_raw[t_idx], interpolation="nearest")
        ax_img.axis("off")

        # Health bar overlay on frame
        H, W = obs_raw[t_idx].shape[:2]
        health_pred = predicted_all[t_idx, KEY_FEATURES["health"]]
        health_max = 9.0
        bar_y = 1
        bar_h = 4
        bar_x = 3
        bar_w = W - 6

        # Bar background
        rect_bg = plt.Rectangle((bar_x, bar_y), bar_w, bar_h,
                                facecolor="#333333", alpha=0.7,
                                edgecolor="white", linewidth=0.5)
        ax_img.add_patch(rect_bg)

        # Health fill
        fill_w = max(0, min(bar_w, bar_w * (health_pred / health_max)))
        fill_color = "#2ecc71" if abs(health_pred - probe_labels[t_idx, 0]) <= 1 else "#e74c3c"
        rect_fill = plt.Rectangle((bar_x, bar_y), fill_w, bar_h,
                                  facecolor=fill_color, alpha=0.85, edgecolor="none")
        ax_img.add_patch(rect_fill)

        # Timestep label
        label = "t=0\n(encoded)" if t_idx == 0 else f"t={t_idx}\n(imagined)"
        ax_img.set_title(label, fontsize=8, fontweight="bold",
                         color="#58a6ff" if t_idx > 0 else "#f0883e", pad=3)

        # --- Bottom row: Prediction comparison text ---
        ax_txt = fig.add_subplot(gs[1, j])
        ax_txt.set_xlim(0, 1)
        ax_txt.set_ylim(0, 1)
        ax_txt.axis("off")
        ax_txt.set_facecolor("#0d1117")

        y_pos = 0.85
        for feat_name, feat_idx in KEY_FEATURES.items():
            pred_v = predicted_all[t_idx, feat_idx]
            act_v = probe_labels[t_idx, feat_idx]
            diff = abs(pred_v - act_v)

            if diff <= 1.0:
                color = "#2ecc71"
                marker = "+"
            elif diff <= 2.0:
                color = "#f39c12"
                marker = "~"
            else:
                color = "#e74c3c"
                marker = "x"

            txt = f"{feat_name[:3].upper()}: {pred_v:4.1f}/{act_v:4.0f} {marker}"
            ax_txt.text(0.05, y_pos, txt, fontsize=7, color=color,
                        fontweight="bold", fontfamily="monospace",
                        transform=ax_txt.transAxes, verticalalignment="top")
            y_pos -= 0.25

    return fig


# ---------------------------------------------------------------------------
# GIF saving utility
# ---------------------------------------------------------------------------

def _save_gif(frames, output_path, duration=1500):
    """Save a list of RGB numpy arrays as an animated GIF."""
    try:
        import imageio
        imageio.mimsave(output_path, frames, duration=duration, loop=0)
        print(f"  Saved GIF to {output_path} ({len(frames)} frames)")
    except ImportError:
        try:
            from PIL import Image
            pil_frames = [Image.fromarray(f) for f in frames]
            pil_frames[0].save(
                output_path, save_all=True, append_images=pil_frames[1:],
                duration=duration, loop=0
            )
            print(f"  Saved GIF to {output_path} ({len(frames)} frames)")
        except ImportError:
            print("  ERROR: Neither imageio nor PIL available. GIF not saved.")
            print("  Install with: pip install imageio  or  pip install Pillow")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Module-level variables for data sharing between functions
_global_traj_dataset = None
_global_train_traj = None
_global_normalizer = None


def main():
    global _global_traj_dataset, _global_train_traj, _global_normalizer

    parser = argparse.ArgumentParser(
        description="JEPA World Model Prediction Overlay Visualization"
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
        "--output_dir", type=str, default="eval_results/overlay_viz",
        help="Where to save outputs."
    )
    parser.add_argument(
        "--episode_idx", type=int, default=0,
        help="Which episode to visualize (default: 0)."
    )
    parser.add_argument(
        "--sample_length", type=int, default=17,
        help="Trajectory length to use (default: 17)."
    )
    parser.add_argument(
        "--epochs_to_show", type=str, default="0,2,4,6,8,10,11",
        help="Comma-separated list of epochs for the GIF (default: 0,2,4,6,8,10,11)."
    )
    parser.add_argument(
        "--comparison_epochs", type=str, default=None,
        help="Comma-separated list of 3 epochs for side-by-side comparison "
             "(default: first, middle, last of epochs_to_show)."
    )
    args = parser.parse_args()

    # Parse epoch lists
    epochs_to_show = [int(e.strip()) for e in args.epochs_to_show.split(",")]

    if args.comparison_epochs:
        comparison_epochs = [int(e.strip()) for e in args.comparison_epochs.split(",")]
    else:
        # Pick 3 representative epochs: first, middle, last
        if len(epochs_to_show) >= 3:
            comparison_epochs = [
                epochs_to_show[0],
                epochs_to_show[len(epochs_to_show) // 2],
                epochs_to_show[-1],
            ]
        else:
            comparison_epochs = epochs_to_show[:3]

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
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n--- Loading data ---")
    traj_dataset, train_traj, val_traj, normalizer = build_normalizer_and_dataset(
        args.data_dir, sample_length=args.sample_length
    )

    # Set module-level globals for data sharing
    _global_traj_dataset = traj_dataset
    _global_train_traj = train_traj
    _global_normalizer = normalizer

    # Prepare the display episode
    print(f"\nPreparing display episode (idx={args.episode_idx})...")
    obs_tensor, actions_tensor, obs_raw, probe_labels = prepare_episode(
        traj_dataset, args.episode_idx, normalizer,
        args.sample_length, device
    )
    print(f"  obs shape: {obs_tensor.shape}, actions shape: {actions_tensor.shape}")
    print(f"  Raw frames: {obs_raw.shape}, probe labels: {probe_labels.shape}")

    # ------------------------------------------------------------------
    # 2. Discover available checkpoints
    # ------------------------------------------------------------------
    print("\n--- Discovering checkpoints ---")
    max_epoch_found = -1
    for epoch in range(20):
        ckpt_path = os.path.join(args.checkpoint_dir, f"e-{epoch}.pth.tar")
        if os.path.exists(ckpt_path):
            max_epoch_found = epoch
            print(f"  Found: e-{epoch}.pth.tar")

    if max_epoch_found < 0:
        print("ERROR: No epoch checkpoints found. Check --checkpoint_dir.")
        sys.exit(1)

    # Filter epochs_to_show to available ones
    available_epochs = []
    for epoch in epochs_to_show:
        ckpt_path = os.path.join(args.checkpoint_dir, f"e-{epoch}.pth.tar")
        if os.path.exists(ckpt_path):
            available_epochs.append(epoch)
    epochs_to_show = available_epochs

    print(f"\nWill generate visualizations for epochs: {epochs_to_show}")

    # ------------------------------------------------------------------
    # 3. Train probe ONCE on the best (final) epoch encoder
    # ------------------------------------------------------------------
    final_epoch = max(epochs_to_show)
    print(f"\n--- Training probe on epoch {final_epoch} encoder ---")

    ckpt_path = os.path.join(args.checkpoint_dir, f"e-{final_epoch}.pth.tar")
    jepa = build_jepa_model(device)
    load_checkpoint(jepa, ckpt_path, device)
    jepa.eval()

    features, labels = collect_encoded_features(
        jepa, train_traj, normalizer, device,
        num_episodes=80, sample_length=args.sample_length
    )
    print(f"  Collected {features.shape[0]} feature vectors for probe training")

    probe = train_linear_probe(features, labels, device, epochs=10, lr=1e-3)

    del jepa
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 4. Generate predictions for each epoch
    # ------------------------------------------------------------------
    print("\n--- Generating imagined predictions for each epoch ---")

    T = obs_raw.shape[0]
    show_indices = [i for i in range(0, T, 2)][:8]

    all_epoch_predictions = {}

    for epoch in epochs_to_show:
        print(f"\n  Epoch {epoch}:")
        ckpt_path = os.path.join(args.checkpoint_dir, f"e-{epoch}.pth.tar")

        jepa_ep = build_jepa_model(device)
        load_checkpoint(jepa_ep, ckpt_path, device)
        jepa_ep.eval()

        predicted_all = get_imagined_predictions(
            jepa_ep, probe, obs_tensor, actions_tensor, device
        )
        all_epoch_predictions[epoch] = predicted_all

        # Print sample predictions vs actuals
        print(f"    Sample predictions at t=4:")
        for feat_name, feat_idx in KEY_FEATURES.items():
            if 4 < T:
                pred_v = predicted_all[4, feat_idx]
                act_v = probe_labels[4, feat_idx]
                print(f"      {feat_name}: pred={pred_v:.1f}, actual={act_v:.0f}, "
                      f"diff={abs(pred_v - act_v):.1f}")

        del jepa_ep
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 5. Output 1: overlay_demo.gif
    # ------------------------------------------------------------------
    print("\n\n=== Output 1: overlay_demo.gif ===")

    gif_frames = []
    for epoch in epochs_to_show:
        predicted_all = all_epoch_predictions[epoch]
        fig = create_frame_strip(
            obs_raw, predicted_all, probe_labels, epoch,
            show_indices=show_indices, total_epochs=max(epochs_to_show)
        )

        fig.canvas.draw()
        w_px, h_px = fig.canvas.get_width_height()
        try:
            buf = fig.canvas.buffer_rgba()
            img = np.frombuffer(buf, dtype=np.uint8).reshape((h_px, w_px, 4))[:, :, :3].copy()
        except Exception:
            buf = fig.canvas.tostring_rgb()
            img = np.frombuffer(buf, dtype=np.uint8).reshape((h_px, w_px, 3)).copy()

        gif_frames.append(img)

        # Also save individual frame
        frame_path = os.path.join(args.output_dir, f"overlay_epoch_{epoch:02d}.png")
        fig.savefig(frame_path, dpi=120, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)

    gif_path = os.path.join(args.output_dir, "overlay_demo.gif")
    _save_gif(gif_frames, gif_path, duration=1500)

    # ------------------------------------------------------------------
    # 6. Output 2: prediction_accuracy.png
    # ------------------------------------------------------------------
    print("\n\n=== Output 2: prediction_accuracy.png ===")

    predicted_final = all_epoch_predictions[final_epoch]
    timesteps = np.arange(T)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor="white")
    fig.suptitle(
        f"World Model Imagination vs Reality (Epoch {final_epoch})",
        fontsize=16, fontweight="bold", y=0.98
    )

    for ax_idx, (feat_name, feat_idx) in enumerate(KEY_FEATURES.items()):
        ax = axes[ax_idx // 2, ax_idx % 2]

        actual = probe_labels[:, feat_idx]
        predicted = predicted_final[:, feat_idx]

        # Actual values (solid green)
        ax.plot(timesteps, actual, 'o-', color="#2ecc71", linewidth=2.5,
                markersize=6, markerfacecolor="white", markeredgewidth=1.5,
                markeredgecolor="#2ecc71", label="Actual (ground truth)", zorder=5)

        # Predicted values (dashed blue)
        ax.plot(timesteps, predicted, 's--', color="#3498db", linewidth=2,
                markersize=5, markerfacecolor="#3498db", markeredgewidth=1,
                markeredgecolor="#3498db", label="Predicted (imagined)", zorder=4)

        # Fill between
        ax.fill_between(timesteps, actual, predicted, alpha=0.15, color="#e74c3c")

        # Mark encoded vs imagined boundary
        ax.axvline(x=0.5, color="#f39c12", linestyle=":", alpha=0.7, linewidth=1.5)
        ax.text(0.02, 0.95, "Encoded", fontsize=8, color="#f39c12",
                transform=ax.transAxes, ha="left", va="top", fontweight="bold")
        ax.text(0.12, 0.95, "Imagined  -->", fontsize=8, color="#3498db",
                transform=ax.transAxes, ha="left", va="top", fontweight="bold")

        # Correlation and MAE
        if len(actual) > 1 and np.std(actual) > 0:
            corr = np.corrcoef(actual, predicted)[0, 1]
            mae = np.mean(np.abs(actual - predicted))
            ax.text(0.98, 0.05, f"Corr: {corr:.3f}\nMAE: {mae:.2f}",
                    fontsize=10, color="#333333", transform=ax.transAxes,
                    ha="right", va="bottom", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f4f8",
                              edgecolor="#d0d7de", alpha=0.9))

        ax.set_xlabel("Timestep", fontsize=11)
        ax.set_ylabel("Value", fontsize=11)
        ax.set_title(f"{feat_name.upper()}",
                     fontsize=14, fontweight="bold",
                     color=FEATURE_COLORS.get(feat_name, "#333333"))
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(timesteps)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    accuracy_path = os.path.join(args.output_dir, "prediction_accuracy.png")
    fig.savefig(accuracy_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved prediction accuracy plot to {accuracy_path}")

    # ------------------------------------------------------------------
    # 7. Output 3: epoch_comparison.png
    # ------------------------------------------------------------------
    print("\n\n=== Output 3: epoch_comparison.png ===")

    # Filter comparison epochs to available
    comparison_epochs_available = [e for e in comparison_epochs if e in all_epoch_predictions]
    if len(comparison_epochs_available) == 0:
        comparison_epochs_available = [epochs_to_show[0], epochs_to_show[-1]]
        if len(epochs_to_show) >= 3:
            comparison_epochs_available.insert(1, epochs_to_show[len(epochs_to_show) // 2])

    n_rows = len(comparison_epochs_available)
    n_frames = len(show_indices)

    fig = plt.figure(figsize=(n_frames * 3.0, n_rows * 4.5 + 1.5), facecolor="#0d1117")
    fig.suptitle(
        "World Model Improvement Over Training",
        fontsize=20, fontweight="bold", color="white", y=0.99
    )
    fig.text(0.5, 0.965,
             "Same trajectory, same probe -- only the encoder changes.  "
             "Green = accurate (+/-1)  Orange = close (+/-2)  Red = wrong",
             fontsize=10, color="#8b949e", ha="center", fontfamily="monospace")

    outer_gs = gridspec.GridSpec(n_rows, 1, hspace=0.35,
                                 top=0.94, bottom=0.02, left=0.06, right=0.98)

    for row_idx, epoch in enumerate(comparison_epochs_available):
        predicted_all = all_epoch_predictions[epoch]

        inner_gs = gridspec.GridSpecFromSubplotSpec(
            2, n_frames, subplot_spec=outer_gs[row_idx],
            height_ratios=[3, 1], hspace=0.05, wspace=0.08
        )

        # Compute accuracy for this epoch
        total_checks = 0
        accurate_checks = 0
        for t_idx in show_indices:
            for feat_name, feat_idx in KEY_FEATURES.items():
                diff = abs(predicted_all[t_idx, feat_idx] - probe_labels[t_idx, feat_idx])
                total_checks += 1
                if diff <= 1.0:
                    accurate_checks += 1
        accuracy_pct = 100 * accurate_checks / max(total_checks, 1)

        label_color = "#e74c3c" if accuracy_pct < 40 else ("#f39c12" if accuracy_pct < 70 else "#2ecc71")

        # Row label on the left
        pos = outer_gs[row_idx].get_position(fig)
        fig.text(0.02, (pos.y0 + pos.y1) / 2,
                 f"Epoch {epoch}\n{accuracy_pct:.0f}% acc",
                 fontsize=12, fontweight="bold", color=label_color,
                 verticalalignment="center", fontfamily="monospace",
                 rotation=0)

        for j, t_idx in enumerate(show_indices):
            # Image
            ax_img = fig.add_subplot(inner_gs[0, j])
            ax_img.imshow(obs_raw[t_idx], interpolation="nearest")
            ax_img.axis("off")

            # Health bar overlay
            H, W = obs_raw[t_idx].shape[:2]
            health_pred = predicted_all[t_idx, KEY_FEATURES["health"]]
            health_max = 9.0
            bar_h = 4
            bar_x = 3
            bar_w = W - 6

            rect_bg = plt.Rectangle((bar_x, 1), bar_w, bar_h,
                                    facecolor="#333333", alpha=0.7,
                                    edgecolor="white", linewidth=0.5)
            ax_img.add_patch(rect_bg)

            fill_w = max(0, min(bar_w, bar_w * (health_pred / health_max)))
            diff_h = abs(health_pred - probe_labels[t_idx, 0])
            fill_color = "#2ecc71" if diff_h <= 1 else ("#f39c12" if diff_h <= 2 else "#e74c3c")
            rect_fill = plt.Rectangle((bar_x, 1), fill_w, bar_h,
                                      facecolor=fill_color, alpha=0.85, edgecolor="none")
            ax_img.add_patch(rect_fill)

            # Title only on first row
            if row_idx == 0:
                label = "t=0 (enc)" if t_idx == 0 else f"t={t_idx} (img)"
                ax_img.set_title(label, fontsize=7, fontweight="bold",
                                 color="#58a6ff" if t_idx > 0 else "#f0883e", pad=2)

            # Annotation text
            ax_txt = fig.add_subplot(inner_gs[1, j])
            ax_txt.set_xlim(0, 1)
            ax_txt.set_ylim(0, 1)
            ax_txt.axis("off")
            ax_txt.set_facecolor("#0d1117")

            y_pos = 0.9
            for feat_name, feat_idx in KEY_FEATURES.items():
                pred_v = predicted_all[t_idx, feat_idx]
                act_v = probe_labels[t_idx, feat_idx]
                diff = abs(pred_v - act_v)

                if diff <= 1.0:
                    color = "#2ecc71"
                elif diff <= 2.0:
                    color = "#f39c12"
                else:
                    color = "#e74c3c"

                txt = f"{feat_name[:3].upper()}: {pred_v:.0f}/{act_v:.0f}"
                ax_txt.text(0.05, y_pos, txt, fontsize=6, color=color,
                            fontweight="bold", fontfamily="monospace",
                            transform=ax_txt.transAxes, verticalalignment="top")
                y_pos -= 0.24

    comparison_path = os.path.join(args.output_dir, "epoch_comparison.png")
    fig.savefig(comparison_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"  Saved epoch comparison to {comparison_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  OVERLAY VISUALIZATION COMPLETE")
    print("=" * 70)

    # Compute accuracy progression
    print("\n  Prediction accuracy progression (% of predictions within +/-1):")
    for epoch in epochs_to_show:
        predicted_all = all_epoch_predictions[epoch]
        total = 0
        accurate = 0
        for t_idx in range(T):
            for feat_name, feat_idx in KEY_FEATURES.items():
                diff = abs(predicted_all[t_idx, feat_idx] - probe_labels[t_idx, feat_idx])
                total += 1
                if diff <= 1.0:
                    accurate += 1
        pct = 100 * accurate / max(total, 1)
        bar = "#" * int(pct / 2) + "-" * (50 - int(pct / 2))
        print(f"    Epoch {epoch:>2d}: [{bar}] {pct:.1f}%")

    print(f"\n  Outputs:")
    print(f"    GIF:              {args.output_dir}/overlay_demo.gif")
    print(f"    Accuracy plot:    {args.output_dir}/prediction_accuracy.png")
    print(f"    Epoch comparison: {args.output_dir}/epoch_comparison.png")
    print(f"    Per-epoch PNGs:   {args.output_dir}/overlay_epoch_XX.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
