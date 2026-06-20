#!/usr/bin/env python3
"""
Dynamics-aware probe evaluation for JEPA world model on Crafter.

Unlike the standard inventory-based probes (health, food, etc.) which random
CNNs can easily predict from on-screen bars, these probes test DYNAMICS
understanding -- things a random encoder fundamentally cannot do well:

  1. Action prediction (IDM probe): Given two consecutive latent frames,
     predict which action was taken. Tests action-relevant transition structure.

  2. Next-state prediction: Given latent frame t and action, predict latent
     frame t+1 via a linear probe. Measures cosine similarity.

  3. Temporal ordering: Given two frames from the same trajectory, predict
     which came first. Tests temporal structure in the latent space.

Usage:
    python scripts/eval_probe_spatial.py \
        --checkpoint_path checkpoints/latest.pth.tar \
        --data_dir data/crafter_trajectories \
        --output_dir eval_results/probe_spatial
"""

import argparse
import json
import os
import sys
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so imports work when running as a
# standalone script (python scripts/eval_probe_spatial.py).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eb_jepa.action_encoders import ActionEmbeddingEncoder
from eb_jepa.architectures import (
    ImpalaEncoder,
    InverseDynamicsModel,
    Projector,
    RNNPredictor,
)
from eb_jepa.datasets.utils import init_data
from eb_jepa.jepa import JEPA
from eb_jepa.losses import SquareLossSeq, VC_IDM_Sim_Regularizer

# ---------------------------------------------------------------------------
# Default Crafter JEPA model config (matches crafter.yaml)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_CFG = {
    "dobs": 3,
    "henc": 32,
    "dstc": 32,
    "num_actions": 17,
    "d_action_emb": 32,
    "discrete_actions": True,
    "img_size": 64,
    "mlp_output_dim": 512,
    "cov_coeff": 8,
    "std_coeff": 16,
    "sim_coeff_t": 12,
    "idm_coeff": 1,
    "first_t_only": False,
    "spatial_as_samples": False,
    "use_proj": False,
    "idm_after_proj": False,
    "sim_t_after_proj": False,
}

NUM_ACTIONS = 17
FEAT_DIM = 512
ACTION_EMB_DIM = 32
TEMPORAL_GAP = 5


# ===================================================================
# Model construction (reused from eval_probe.py)
# ===================================================================

def build_jepa(cfg: dict, device: torch.device) -> JEPA:
    """Construct a JEPA model matching the Crafter training config."""
    dobs = cfg["dobs"]
    img_size = cfg["img_size"]

    encoder = ImpalaEncoder(
        width=1,
        stack_sizes=(16, cfg["henc"], cfg["dstc"]),
        num_blocks=2,
        dropout_rate=None,
        layer_norm=False,
        input_channels=dobs,
        final_ln=True,
        mlp_output_dim=cfg["mlp_output_dim"],
        input_shape=(dobs, img_size, img_size),
    )

    test_input = torch.rand(1, dobs, 1, img_size, img_size)
    with torch.no_grad():
        test_output = encoder(test_input)
    _, f, _, h, w = test_output.shape

    d_action_emb = cfg["d_action_emb"]
    num_actions = cfg["num_actions"]
    aencoder = ActionEmbeddingEncoder(num_actions, d_action_emb)
    predictor = RNNPredictor(
        hidden_size=encoder.mlp_output_dim,
        action_dim=d_action_emb,
        final_ln=encoder.final_ln,
    )

    if cfg["use_proj"]:
        projector = Projector(
            f"{encoder.mlp_output_dim}-{encoder.mlp_output_dim*4}-{encoder.mlp_output_dim*4}"
        )
    else:
        projector = None

    idm_action_dim = num_actions
    idm = InverseDynamicsModel(
        state_dim=h * w * (projector.out_dim if cfg["idm_after_proj"] else f),
        hidden_dim=256,
        action_dim=idm_action_dim,
    )
    regularizer = VC_IDM_Sim_Regularizer(
        cov_coeff=cfg["cov_coeff"],
        std_coeff=cfg["std_coeff"],
        sim_coeff_t=cfg["sim_coeff_t"],
        idm_coeff=cfg["idm_coeff"],
        idm=idm,
        first_t_only=cfg["first_t_only"],
        projector=projector,
        spatial_as_samples=cfg["spatial_as_samples"],
        idm_after_proj=cfg["idm_after_proj"],
        sim_t_after_proj=cfg["sim_t_after_proj"],
        discrete=True,
    )
    ploss = SquareLossSeq()
    jepa = JEPA(encoder, aencoder, predictor, regularizer, ploss).to(device)
    return jepa


def load_trained_jepa(checkpoint_path: str, cfg: dict, device: torch.device) -> JEPA:
    """Build a JEPA and load trained weights from a checkpoint."""
    jepa = build_jepa(cfg, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    jepa.load_state_dict(state_dict, strict=False)
    epoch = checkpoint.get("epoch", "?")
    step = checkpoint.get("step", "?")
    print(f"Loaded checkpoint from {checkpoint_path} (epoch={epoch}, step={step})")
    return jepa


def build_random_jepa(cfg: dict, device: torch.device) -> JEPA:
    """Build a fresh JEPA with random weights (untrained baseline)."""
    jepa = build_jepa(cfg, device)
    print("Built random (untrained) JEPA baseline")
    return jepa


# ===================================================================
# Encoding — returns features AND per-timestep actions
# ===================================================================

@torch.no_grad()
def encode_dataset(jepa: JEPA, loader, device: torch.device):
    """
    Run the frozen JEPA encoder over the dataset.

    Returns:
        features: [N, T, 512] tensor of latent representations (per-trajectory slices)
        actions:  [N, T] tensor of actions
    """
    jepa.eval()
    all_features = []
    all_actions = []

    for batch in loader:
        obs = batch[0].to(device)       # [B, C, T, H, W]
        actions = batch[1]              # [B, T]

        # Encode: output is [B, D, T, 1, 1]
        enc = jepa.encode(obs)          # [B, D, T, 1, 1]
        B, D, T, _, _ = enc.shape

        # Reshape to [B, T, D]
        enc_flat = enc.squeeze(-1).squeeze(-1)   # [B, D, T]
        enc_flat = enc_flat.permute(0, 2, 1)     # [B, T, D]

        all_features.append(enc_flat.cpu())
        all_actions.append(actions.cpu())

    features = torch.cat(all_features, dim=0)  # [N, T, D]
    actions = torch.cat(all_actions, dim=0)     # [N, T]
    return features, actions


# ===================================================================
# Probe 1: Action Prediction (Inverse Dynamics Model probe)
# ===================================================================

def prepare_idm_data(features: torch.Tensor, actions: torch.Tensor):
    """
    Prepare data for the IDM probe.

    Args:
        features: [N, T, D] encoded frames
        actions:  [N, T] actions

    Returns:
        inputs:  [M, 2*D] concatenated consecutive frame pairs
        targets: [M] action indices
    """
    N, T, D = features.shape
    # For each trajectory slice, pair (t, t+1) with action[t]
    feat_t = features[:, :-1, :]       # [N, T-1, D]
    feat_t1 = features[:, 1:, :]       # [N, T-1, D]
    act_t = actions[:, :-1]            # [N, T-1]

    # Flatten trajectory dimension
    inputs = torch.cat([feat_t, feat_t1], dim=-1).reshape(-1, 2 * D)  # [M, 2D]
    targets = act_t.reshape(-1)                                        # [M]
    return inputs, targets


def train_idm_probe(train_x, train_y, val_x, val_y, epochs=10, batch_size=256,
                    lr=1e-3, device=torch.device("cpu")):
    """Train linear probe for action prediction and return val accuracy."""
    input_dim = train_x.shape[1]
    probe = nn.Linear(input_dim, NUM_ACTIONS).to(device)
    optimizer = AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)

    n_train = train_x.shape[0]
    n_batches = max(1, n_train // batch_size)

    for epoch in range(1, epochs + 1):
        probe.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0

        for i in range(n_batches):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            x = train_x[idx].to(device)
            y = train_y[idx].to(device)

            logits = probe(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validate
        probe.eval()
        with torch.no_grad():
            val_preds = []
            for i in range(0, val_x.shape[0], batch_size):
                x = val_x[i : i + batch_size].to(device)
                val_preds.append(probe(x).argmax(dim=-1).cpu())
            val_preds = torch.cat(val_preds, dim=0)
            val_acc = (val_preds == val_y).float().mean().item()

        print(f"  [IDM] Epoch {epoch:>2d}/{epochs} | "
              f"Train CE: {epoch_loss / n_batches:.4f} | Val Acc: {val_acc:.4f}")

    return probe, val_acc


# ===================================================================
# Probe 2: Next-State Prediction
# ===================================================================

def prepare_next_state_data(features: torch.Tensor, actions: torch.Tensor):
    """
    Prepare data for next-state prediction probe.

    Args:
        features: [N, T, D]
        actions:  [N, T]

    Returns:
        inputs:  [M, D + ACTION_EMB_DIM] will be built during training with embedding
        feat_t:  [M, D]
        act_t:   [M]
        targets: [M, D] next-state features
    """
    N, T, D = features.shape
    feat_t = features[:, :-1, :].reshape(-1, D)       # [M, D]
    feat_t1 = features[:, 1:, :].reshape(-1, D)       # [M, D]
    act_t = actions[:, :-1].reshape(-1)                # [M]
    return feat_t, act_t, feat_t1


class NextStateProbe(nn.Module):
    """Linear probe: enc(t) + action_embedding(a_t) -> enc(t+1)"""
    def __init__(self, feat_dim=FEAT_DIM, action_emb_dim=ACTION_EMB_DIM,
                 num_actions=NUM_ACTIONS):
        super().__init__()
        self.action_emb = nn.Embedding(num_actions, action_emb_dim)
        self.linear = nn.Linear(feat_dim + action_emb_dim, feat_dim)

    def forward(self, feat, action):
        a_emb = self.action_emb(action)                    # [B, action_emb_dim]
        x = torch.cat([feat, a_emb], dim=-1)               # [B, feat_dim + action_emb_dim]
        return self.linear(x)                               # [B, feat_dim]


def train_next_state_probe(feat_t, act_t, feat_t1, val_feat_t, val_act_t, val_feat_t1,
                           epochs=10, batch_size=256, lr=1e-3,
                           device=torch.device("cpu")):
    """Train next-state prediction probe. Returns probe and val cosine similarity."""
    probe = NextStateProbe().to(device)
    optimizer = AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)

    n_train = feat_t.shape[0]
    n_batches = max(1, n_train // batch_size)

    for epoch in range(1, epochs + 1):
        probe.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0

        for i in range(n_batches):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            f = feat_t[idx].to(device)
            a = act_t[idx].to(device)
            target = feat_t1[idx].to(device)

            pred = probe(f, a)
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validate: cosine similarity
        probe.eval()
        with torch.no_grad():
            cos_sims = []
            for i in range(0, val_feat_t.shape[0], batch_size):
                f = val_feat_t[i : i + batch_size].to(device)
                a = val_act_t[i : i + batch_size].to(device)
                target = val_feat_t1[i : i + batch_size].to(device)
                pred = probe(f, a)
                cs = F.cosine_similarity(pred, target, dim=-1)
                cos_sims.append(cs.cpu())
            val_cos = torch.cat(cos_sims).mean().item()

        print(f"  [NextState] Epoch {epoch:>2d}/{epochs} | "
              f"Train MSE: {epoch_loss / n_batches:.6f} | Val CosSim: {val_cos:.4f}")

    return probe, val_cos


# ===================================================================
# Probe 3: Temporal Ordering
# ===================================================================

def prepare_temporal_ordering_data(features: torch.Tensor, gap: int = TEMPORAL_GAP):
    """
    Prepare data for temporal ordering probe.

    Args:
        features: [N, T, D]
        gap: temporal gap between paired frames

    Returns:
        inputs:  [M, 2*D] paired frames (possibly in reversed order)
        targets: [M] binary labels (1 = forward order, 0 = reversed)
    """
    N, T, D = features.shape
    if T <= gap:
        raise ValueError(f"Temporal gap {gap} >= sequence length {T}")

    feat_early = features[:, :-gap, :]     # [N, T-gap, D]
    feat_late = features[:, gap:, :]       # [N, T-gap, D]

    M_per_traj = T - gap
    M = N * M_per_traj

    feat_early = feat_early.reshape(M, D)
    feat_late = feat_late.reshape(M, D)

    # Create labels: 1 = forward (early, late), 0 = reversed (late, early)
    # Randomly flip 50% of pairs
    rng = torch.Generator().manual_seed(123)
    flip_mask = torch.rand(M, generator=rng) < 0.5   # True = flip

    # Build input pairs
    left = torch.where(flip_mask.unsqueeze(-1), feat_late, feat_early)
    right = torch.where(flip_mask.unsqueeze(-1), feat_early, feat_late)

    inputs = torch.cat([left, right], dim=-1)   # [M, 2D]
    targets = (~flip_mask).float()               # 1 = forward, 0 = reversed

    return inputs, targets


def train_temporal_probe(train_x, train_y, val_x, val_y, epochs=10, batch_size=256,
                         lr=1e-3, device=torch.device("cpu")):
    """Train binary classifier for temporal ordering."""
    input_dim = train_x.shape[1]
    probe = nn.Linear(input_dim, 1).to(device)
    optimizer = AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)

    n_train = train_x.shape[0]
    n_batches = max(1, n_train // batch_size)

    for epoch in range(1, epochs + 1):
        probe.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0

        for i in range(n_batches):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            x = train_x[idx].to(device)
            y = train_y[idx].to(device)

            logits = probe(x).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validate
        probe.eval()
        with torch.no_grad():
            val_preds = []
            for i in range(0, val_x.shape[0], batch_size):
                x = val_x[i : i + batch_size].to(device)
                pred = (probe(x).squeeze(-1) > 0.0).float()
                val_preds.append(pred.cpu())
            val_preds = torch.cat(val_preds, dim=0)
            val_acc = (val_preds == val_y).float().mean().item()

        print(f"  [Temporal] Epoch {epoch:>2d}/{epochs} | "
              f"Train BCE: {epoch_loss / n_batches:.4f} | Val Acc: {val_acc:.4f}")

    return probe, val_acc


# ===================================================================
# Results display
# ===================================================================

def print_comparison_table(trained_results: dict, random_results: dict):
    """Print a comparison table of trained vs random encoder on all 3 probes."""
    print(f"\n{'=' * 78}")
    print(f"  DYNAMICS PROBE COMPARISON: Trained vs Random Encoder")
    print(f"{'=' * 78}")
    print(f"  {'Probe':<30} {'Metric':<15} {'Trained':>10} {'Random':>10} {'Delta':>10}")
    print(f"  {'-' * 30} {'-' * 15} {'-' * 10} {'-' * 10} {'-' * 10}")

    rows = [
        ("Action Prediction (IDM)", "Top-1 Acc",
         trained_results["idm_accuracy"], random_results["idm_accuracy"]),
        ("Next-State Prediction", "Cos Sim",
         trained_results["next_state_cosine_sim"], random_results["next_state_cosine_sim"]),
        ("Temporal Ordering", "Binary Acc",
         trained_results["temporal_ordering_accuracy"], random_results["temporal_ordering_accuracy"]),
    ]

    for name, metric, t_val, r_val in rows:
        delta = t_val - r_val
        print(f"  {name:<30} {metric:<15} {t_val:>10.4f} {r_val:>10.4f} {delta:>+10.4f}")

    print(f"{'=' * 78}")

    # Interpretation
    print(f"\n  Interpretation:")
    idm_delta = trained_results["idm_accuracy"] - random_results["idm_accuracy"]
    ns_delta = trained_results["next_state_cosine_sim"] - random_results["next_state_cosine_sim"]
    to_delta = trained_results["temporal_ordering_accuracy"] - random_results["temporal_ordering_accuracy"]

    if idm_delta > 0.05:
        print(f"    IDM: Trained encoder captures action-relevant transitions (+{idm_delta:.1%})")
    else:
        print(f"    IDM: Minimal difference -- encoder may not capture actions well")

    if ns_delta > 0.02:
        print(f"    Next-State: Trained latent is more predictable (+{ns_delta:.4f} cos sim)")
    else:
        print(f"    Next-State: Minimal difference in latent predictability")

    if to_delta > 0.02:
        print(f"    Temporal: Trained encoder encodes temporal structure (+{to_delta:.1%})")
    else:
        print(f"    Temporal: Minimal temporal structure difference")


def generate_bar_chart(trained_results: dict, random_results: dict, output_path: str):
    """Generate a bar chart comparing trained vs random on all 3 probes."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARNING] matplotlib not installed, skipping chart generation")
        return

    probe_names = ["Action Prediction\n(IDM Accuracy)", "Next-State\n(Cosine Sim)",
                   "Temporal Ordering\n(Binary Accuracy)"]
    trained_vals = [
        trained_results["idm_accuracy"],
        trained_results["next_state_cosine_sim"],
        trained_results["temporal_ordering_accuracy"],
    ]
    random_vals = [
        random_results["idm_accuracy"],
        random_results["next_state_cosine_sim"],
        random_results["temporal_ordering_accuracy"],
    ]

    x = np.arange(len(probe_names))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_trained = ax.bar(x - width / 2, trained_vals, width, label="Trained Encoder",
                          color="#2196F3", edgecolor="black", linewidth=0.5)
    bars_random = ax.bar(x + width / 2, random_vals, width, label="Random Encoder",
                         color="#FF9800", edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar in bars_trained:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar in bars_random:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Dynamics Probes: Trained vs Random Encoder", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(probe_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(max(trained_vals), max(random_vals)) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved to {output_path}")


# ===================================================================
# Main
# ===================================================================

def run_all_probes(features_train, actions_train, features_val, actions_val,
                   label: str, epochs=10, lr=1e-3, device=torch.device("cpu")):
    """Run all 3 probes on a set of encoded features. Returns results dict."""
    print(f"\n--- Running probes on {label} encoder ---")
    results = {}

    # --- Probe 1: Action Prediction (IDM) ---
    print(f"\n  Probe 1: Action Prediction (IDM)")
    train_x_idm, train_y_idm = prepare_idm_data(features_train, actions_train)
    val_x_idm, val_y_idm = prepare_idm_data(features_val, actions_val)
    print(f"    Train samples: {train_x_idm.shape[0]}, Val samples: {val_x_idm.shape[0]}")
    _, idm_acc = train_idm_probe(train_x_idm, train_y_idm, val_x_idm, val_y_idm,
                                 epochs=epochs, lr=lr, device=device)
    results["idm_accuracy"] = idm_acc

    # --- Probe 2: Next-State Prediction ---
    print(f"\n  Probe 2: Next-State Prediction")
    feat_t_tr, act_t_tr, feat_t1_tr = prepare_next_state_data(features_train, actions_train)
    feat_t_val, act_t_val, feat_t1_val = prepare_next_state_data(features_val, actions_val)
    print(f"    Train samples: {feat_t_tr.shape[0]}, Val samples: {feat_t_val.shape[0]}")
    _, ns_cos = train_next_state_probe(feat_t_tr, act_t_tr, feat_t1_tr,
                                       feat_t_val, act_t_val, feat_t1_val,
                                       epochs=epochs, lr=lr, device=device)
    results["next_state_cosine_sim"] = ns_cos

    # --- Probe 3: Temporal Ordering ---
    print(f"\n  Probe 3: Temporal Ordering (gap={TEMPORAL_GAP})")
    train_x_to, train_y_to = prepare_temporal_ordering_data(features_train, gap=TEMPORAL_GAP)
    val_x_to, val_y_to = prepare_temporal_ordering_data(features_val, gap=TEMPORAL_GAP)
    print(f"    Train samples: {train_x_to.shape[0]}, Val samples: {val_x_to.shape[0]}")
    _, to_acc = train_temporal_probe(train_x_to, train_y_to, val_x_to, val_y_to,
                                    epochs=epochs, lr=lr, device=device)
    results["temporal_ordering_accuracy"] = to_acc

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Dynamics-aware probe evaluation for JEPA world model on Crafter"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to trained JEPA checkpoint (.pth.tar)",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to crafter trajectory data directory",
    )
    parser.add_argument(
        "--output_dir", type=str, default="eval_results/probe_spatial",
        help="Directory to save results (default: eval_results/probe_spatial)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of probe training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for data loading (default: 64)",
    )
    parser.add_argument(
        "--probe_lr", type=float, default=1e-3,
        help="Learning rate for probe training (default: 1e-3)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (default: auto-detect cuda/mps/cpu)",
    )
    args = parser.parse_args()

    # --- Device ---
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Load data ---
    print("\n--- Loading Crafter dataset ---")
    cfg_data = {
        "data_dir": args.data_dir,
        "batch_size": args.batch_size,
        "num_workers": 4,
        "pin_mem": True,
        "persistent_workers": False,
        "sample_length": 17,
    }
    train_loader, val_loader, data_config = init_data("crafter", cfg_data)
    print(f"Train: {data_config.size} slices, Val: {data_config.val_size} slices")

    model_cfg = dict(DEFAULT_MODEL_CFG)

    # =================================================================
    # 1. Encode with TRAINED encoder
    # =================================================================
    print("\n" + "=" * 78)
    print("  Encoding with TRAINED encoder")
    print("=" * 78)

    jepa_trained = load_trained_jepa(args.checkpoint_path, model_cfg, device)
    jepa_trained.eval()

    print("\nEncoding training set...")
    t0 = time()
    train_feats, train_actions = encode_dataset(jepa_trained, train_loader, device)
    print(f"  Encoded {train_feats.shape[0]} trajectory slices in {time() - t0:.1f}s "
          f"-> features {train_feats.shape}")

    print("Encoding validation set...")
    t0 = time()
    val_feats, val_actions = encode_dataset(jepa_trained, val_loader, device)
    print(f"  Encoded {val_feats.shape[0]} trajectory slices in {time() - t0:.1f}s "
          f"-> features {val_feats.shape}")

    del jepa_trained
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # =================================================================
    # 2. Encode with RANDOM encoder
    # =================================================================
    print("\n" + "=" * 78)
    print("  Encoding with RANDOM (untrained) encoder")
    print("=" * 78)

    jepa_random = build_random_jepa(model_cfg, device)
    jepa_random.eval()

    print("\nEncoding training set...")
    t0 = time()
    train_feats_rand, train_actions_rand = encode_dataset(jepa_random, train_loader, device)
    print(f"  Encoded {train_feats_rand.shape[0]} trajectory slices in {time() - t0:.1f}s")

    print("Encoding validation set...")
    t0 = time()
    val_feats_rand, val_actions_rand = encode_dataset(jepa_random, val_loader, device)
    print(f"  Encoded {val_feats_rand.shape[0]} trajectory slices in {time() - t0:.1f}s")

    del jepa_random
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # =================================================================
    # 3. Run all probes
    # =================================================================
    trained_results = run_all_probes(
        train_feats, train_actions, val_feats, val_actions,
        label="TRAINED", epochs=args.epochs, lr=args.probe_lr, device=device,
    )

    random_results = run_all_probes(
        train_feats_rand, train_actions_rand, val_feats_rand, val_actions_rand,
        label="RANDOM", epochs=args.epochs, lr=args.probe_lr, device=device,
    )

    # =================================================================
    # 4. Print comparison
    # =================================================================
    print_comparison_table(trained_results, random_results)

    # =================================================================
    # 5. Save results
    # =================================================================
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_json = {
        "checkpoint_path": str(args.checkpoint_path),
        "data_dir": str(args.data_dir),
        "probe_epochs": args.epochs,
        "probe_lr": args.probe_lr,
        "batch_size": args.batch_size,
        "temporal_gap": TEMPORAL_GAP,
        "num_actions": NUM_ACTIONS,
        "feature_dim": FEAT_DIM,
        "train_slices": int(train_feats.shape[0]),
        "val_slices": int(val_feats.shape[0]),
        "sequence_length": int(train_feats.shape[1]),
        "trained_encoder": trained_results,
        "random_encoder": random_results,
        "deltas": {
            "idm_accuracy": trained_results["idm_accuracy"] - random_results["idm_accuracy"],
            "next_state_cosine_sim": (trained_results["next_state_cosine_sim"]
                                      - random_results["next_state_cosine_sim"]),
            "temporal_ordering_accuracy": (trained_results["temporal_ordering_accuracy"]
                                           - random_results["temporal_ordering_accuracy"]),
        },
    }

    json_path = output_dir / "probe_spatial_results.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to {json_path}")

    chart_path = output_dir / "probe_spatial_comparison.png"
    generate_bar_chart(trained_results, random_results, str(chart_path))

    # --- Final summary ---
    print(f"\n{'=' * 78}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 78}")
    print(f"  {'Probe':<30} {'Trained':>10} {'Random':>10} {'Delta':>10}")
    print(f"  {'-' * 30} {'-' * 10} {'-' * 10} {'-' * 10}")
    for key, label in [("idm_accuracy", "Action Prediction (IDM)"),
                       ("next_state_cosine_sim", "Next-State Prediction"),
                       ("temporal_ordering_accuracy", "Temporal Ordering")]:
        t = trained_results[key]
        r = random_results[key]
        print(f"  {label:<30} {t:>10.4f} {r:>10.4f} {t - r:>+10.4f}")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()
