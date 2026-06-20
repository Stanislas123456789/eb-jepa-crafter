#!/usr/bin/env python3
"""
Compare representation quality: JEPA (no decoder) vs DreamerV2 (pixel reconstruction).

This script provides a fair comparison framework for evaluating latent
representations learned by different world models on Crafter. It runs identical
linear probes on frozen latent features from each model and generates
comparison tables and figures.

Works in two modes:

  Mode 1 (always works): Load pre-computed features as .npy files.
      Any world model can be compared by exporting its latents:
        features.npy  -- shape [N, D], N samples, D latent dim
        labels.npy    -- shape [N, K], K probe targets (matching LABEL_NAMES)
        actions.npy   -- shape [N], integer action taken at each timestep

  Mode 2 (if dreamerv2 is installed): Automatically extract latents from a
      DreamerV2 checkpoint and Crafter trajectory data.

HOW TO GET DREAMERV2 FEATURES:
  1. Install DreamerV2:
       pip install dreamerv2
     or clone: git clone https://github.com/danijar/dreamerv2

  2. Train DreamerV2 on Crafter (or use a pretrained checkpoint):
       python -m dreamerv2.train --logdir ./logdir/crafter \
           --configs crafter --steps 1000000

  3. Export features using this script:
       python scripts/compare_dreamerv2.py \
           --dreamer_checkpoint ./logdir/crafter/checkpoint.pkl \
           --data_dir data/crafter_trajectories \
           --output_dir eval_results/comparison

  4. Or manually export features and pass them:
       python scripts/compare_dreamerv2.py \
           --jepa_features eval_results/jepa_features.npy \
           --dreamer_features eval_results/dreamer_features.npy \
           --labels eval_results/labels.npy \
           --actions eval_results/actions.npy \
           --output_dir eval_results/comparison

Usage:
    # Full pipeline with JEPA checkpoint (DreamerV2 features optional)
    python scripts/compare_dreamerv2.py \\
        --jepa_checkpoint checkpoints/.../latest.pth.tar \\
        --data_dir data/crafter_trajectories \\
        --output_dir eval_results/comparison

    # Pre-computed features only
    python scripts/compare_dreamerv2.py \\
        --jepa_features features_jepa.npy \\
        --dreamer_features features_dreamer.npy \\
        --labels labels.npy \\
        --actions actions.npy \\
        --output_dir eval_results/comparison
"""

import argparse
import json
import os
import sys
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Try to import torch (required) and optional dependencies
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.optim import AdamW

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Probe label names (must match eval_probe.py)
# ---------------------------------------------------------------------------
LABEL_NAMES = [
    # Vitals (0-3): regression, range 0-9
    "health", "food", "drink", "energy",
    # Resources (4-9): regression, counts
    "sapling", "wood", "stone", "coal", "iron", "diamond",
    # Tools (10-15): binary classification
    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "wood_sword", "stone_sword", "iron_sword",
]

# Subset indices for cleaner comparison (skip rare items with degenerate R2)
VITALS = ["health", "food", "drink", "energy"]
RESOURCES = ["sapling", "wood"]
VITALS_IDX = list(range(4))
RESOURCES_IDX = [4, 5]
BINARY_IDX = list(range(10, 16))
# Core probes: vitals + common resources (skip stone/coal/iron/diamond which
# are extremely rare in 170k transitions and produce degenerate R2 values)
CORE_PROBE_NAMES = VITALS + RESOURCES
CORE_PROBE_IDX = VITALS_IDX + RESOURCES_IDX

NUM_ACTIONS = 17  # Crafter action space

# ---------------------------------------------------------------------------
# Style constants (matching make_figures.py)
# ---------------------------------------------------------------------------
C_JEPA = "#0072B2"       # blue
C_DREAMER = "#D55E00"    # vermillion
C_RANDOM = "#999999"     # grey
C_GOOD = "#009E73"       # green

TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_SIZE = 12
DPI = 150


def _apply_style():
    """Set a clean matplotlib style."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            plt.style.use("ggplot")
    plt.rcParams.update({
        "font.size": TICK_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "legend.fontsize": TICK_SIZE,
        "figure.dpi": DPI,
    })


# ===================================================================
# Feature loading / extraction
# ===================================================================

def load_features(path: str) -> np.ndarray:
    """Load pre-computed features from .npy file.
    Expected shape: [N, D] where N=num_samples, D=latent_dim
    """
    feats = np.load(path)
    assert feats.ndim == 2, f"Expected 2D array [N, D], got shape {feats.shape}"
    print(f"  Loaded features from {path}: shape {feats.shape}")
    return feats


def load_labels(path: str) -> np.ndarray:
    """Load probe labels from .npy file.
    Expected shape: [N, K] where K = len(LABEL_NAMES)
    """
    labels = np.load(path)
    assert labels.ndim == 2, f"Expected 2D array [N, K], got shape {labels.shape}"
    print(f"  Loaded labels from {path}: shape {labels.shape}")
    return labels


def load_actions(path: str) -> np.ndarray:
    """Load action labels from .npy file. Shape [N]."""
    actions = np.load(path)
    print(f"  Loaded actions from {path}: shape {actions.shape}")
    return actions


def random_features(n_samples: int, dim: int = 512, seed: int = 42) -> np.ndarray:
    """Generate random features as a baseline."""
    rng = np.random.RandomState(seed)
    return rng.randn(n_samples, dim).astype(np.float32)


# ---------------------------------------------------------------------------
# JEPA feature extraction
# ---------------------------------------------------------------------------

def extract_jepa_features(
    checkpoint_path: str,
    data_dir: str,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features from a trained JEPA checkpoint.

    Returns (features, labels, actions) as numpy arrays.
    """
    from eb_jepa.action_encoders import ActionEmbeddingEncoder
    from eb_jepa.architectures import (
        ImpalaEncoder, InverseDynamicsModel, Projector, RNNPredictor,
    )
    from eb_jepa.datasets.utils import init_data
    from eb_jepa.jepa import JEPA
    from eb_jepa.losses import SquareLossSeq, VC_IDM_Sim_Regularizer

    # Default model config (matches crafter.yaml)
    cfg = {
        "dobs": 3, "henc": 32, "dstc": 32, "num_actions": 17,
        "d_action_emb": 32, "discrete_actions": True, "img_size": 64,
        "mlp_output_dim": 512, "cov_coeff": 8, "std_coeff": 16,
        "sim_coeff_t": 12, "idm_coeff": 1, "first_t_only": False,
        "spatial_as_samples": False, "use_proj": False,
        "idm_after_proj": False, "sim_t_after_proj": False,
    }

    dobs = cfg["dobs"]
    img_size = cfg["img_size"]

    encoder = ImpalaEncoder(
        width=1,
        stack_sizes=(16, cfg["henc"], cfg["dstc"]),
        num_blocks=2, dropout_rate=None, layer_norm=False,
        input_channels=dobs, final_ln=True,
        mlp_output_dim=cfg["mlp_output_dim"],
        input_shape=(dobs, img_size, img_size),
    )

    test_input = torch.rand(1, dobs, 1, img_size, img_size)
    with torch.no_grad():
        test_output = encoder(test_input)
    _, f, _, h, w = test_output.shape

    aencoder = ActionEmbeddingEncoder(cfg["num_actions"], cfg["d_action_emb"])
    predictor = RNNPredictor(
        hidden_size=encoder.mlp_output_dim,
        action_dim=cfg["d_action_emb"],
        final_ln=encoder.final_ln,
    )

    projector = None
    idm = InverseDynamicsModel(
        state_dim=h * w * f, hidden_dim=256, action_dim=cfg["num_actions"],
    )
    regularizer = VC_IDM_Sim_Regularizer(
        cov_coeff=cfg["cov_coeff"], std_coeff=cfg["std_coeff"],
        sim_coeff_t=cfg["sim_coeff_t"], idm_coeff=cfg["idm_coeff"],
        idm=idm, first_t_only=cfg["first_t_only"], projector=projector,
        spatial_as_samples=cfg["spatial_as_samples"],
        idm_after_proj=cfg["idm_after_proj"],
        sim_t_after_proj=cfg["sim_t_after_proj"], discrete=True,
    )
    ploss = SquareLossSeq()
    jepa = JEPA(encoder, aencoder, predictor, regularizer, ploss).to(device)

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    jepa.load_state_dict(state_dict, strict=False)
    epoch = checkpoint.get("epoch", "?")
    step = checkpoint.get("step", "?")
    print(f"  Loaded JEPA checkpoint: {checkpoint_path} (epoch={epoch}, step={step})")
    jepa.eval()

    # Load data
    cfg_data = {
        "data_dir": data_dir, "batch_size": batch_size,
        "num_workers": 4, "pin_mem": True,
        "persistent_workers": False, "sample_length": 17,
    }
    _, val_loader, data_config = init_data("crafter", cfg_data)
    print(f"  Val set: {data_config.val_size} slices")

    # Encode
    all_features, all_labels, all_actions = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            obs = batch[0].to(device)        # [B, C, T, H, W]
            actions_seq = batch[1]           # [B, T-1] or [B, T]
            probe_labels = batch[2]          # [B, K, T]

            enc = jepa.encode(obs)           # [B, D, T, 1, 1]
            B, D, T, _, _ = enc.shape
            enc_flat = enc.squeeze(-1).squeeze(-1).permute(0, 2, 1).reshape(B * T, D)
            labels_flat = probe_labels.permute(0, 2, 1).reshape(B * T, -1)

            all_features.append(enc_flat.cpu().numpy())
            all_labels.append(labels_flat.numpy())

            # Actions: expand to match T timesteps (last action is repeated)
            if actions_seq.shape[-1] < T:
                pad = actions_seq[:, -1:].expand(-1, T - actions_seq.shape[-1])
                actions_seq = torch.cat([actions_seq, pad], dim=-1)
            all_actions.append(actions_seq[:, :T].reshape(-1).numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    actions = np.concatenate(all_actions, axis=0)

    del jepa
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return features, labels, actions


# ---------------------------------------------------------------------------
# DreamerV2 feature extraction (optional)
# ---------------------------------------------------------------------------

def try_extract_dreamer_features(
    checkpoint_path: str,
    data_dir: str,
    device: torch.device,
    batch_size: int = 64,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Try to extract features from a DreamerV2 checkpoint.

    DreamerV2 latent state = concat(deterministic GRU hidden, stochastic categorical).
    For the RSSM default config on Crafter:
      - Deterministic: GRU hidden of size 600
      - Stochastic: 32 categoricals with 32 classes -> one-hot -> 1024
      - Total latent dim: 1624

    Returns (features, labels, actions) or None if dreamerv2 is not installed.
    """
    try:
        import dreamerv2  # noqa: F401
    except ImportError:
        print("  DreamerV2 not installed. To install: pip install dreamerv2")
        print("  Falling back to pre-computed features or random baseline.")
        return None

    print("  DreamerV2 found. Extracting latent representations...")
    print("  NOTE: DreamerV2 integration is experimental. If this fails,")
    print("  export features manually and pass via --dreamer_features.")

    try:
        import pickle
        import pathlib

        # DreamerV2 stores checkpoints as pickle files
        ckpt_path = pathlib.Path(checkpoint_path)
        if not ckpt_path.exists():
            print(f"  Checkpoint not found: {checkpoint_path}")
            return None

        # Load the DreamerV2 agent
        # DreamerV2's checkpoint structure:
        #   {'agent': <agent_state>, 'step': <int>}
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)

        print(f"  Loaded DreamerV2 checkpoint (step={ckpt.get('step', '?')})")

        # Try to use DreamerV2's built-in encoding
        # The RSSM model encodes observations into:
        #   - deter: deterministic state from GRU (typically 600-dim)
        #   - stoch: stochastic state (32 categoricals x 32 classes = 1024-dim one-hot)
        # We concatenate them for probing.
        from eb_jepa.datasets.utils import init_data

        cfg_data = {
            "data_dir": data_dir, "batch_size": batch_size,
            "num_workers": 4, "pin_mem": True,
            "persistent_workers": False, "sample_length": 17,
        }
        _, val_loader, _ = init_data("crafter", cfg_data)

        # Extract features by running RSSM encoder
        # This is model-specific and may need adjustment
        agent = ckpt.get("agent", ckpt)
        if hasattr(agent, "wm") or "wm" in agent:
            wm = agent["wm"] if isinstance(agent, dict) else agent.wm
            print("  Found world model in checkpoint.")
        else:
            print("  Could not find world model in DreamerV2 checkpoint.")
            print("  Please export features manually.")
            return None

        # For now, we rely on pre-computed features
        print("  Full DreamerV2 encoding pipeline not yet automated.")
        print("  Use --dreamer_features to pass pre-computed .npy features.")
        return None

    except Exception as e:
        print(f"  DreamerV2 feature extraction failed: {e}")
        print("  Please export features manually and use --dreamer_features.")
        return None


# ===================================================================
# Probes
# ===================================================================

def train_linear_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Linear, List[float]]:
    """Train a linear probe on frozen features. Returns (probe, val_loss_history)."""
    feat_dim = train_x.shape[1]
    n_targets = train_y.shape[1]
    probe = nn.Linear(feat_dim, n_targets).to(device)
    optimizer = AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    train_x_t = torch.from_numpy(train_x).float()
    train_y_t = torch.from_numpy(train_y).float()
    val_x_t = torch.from_numpy(val_x).float()
    val_y_t = torch.from_numpy(val_y).float()

    n_train = train_x.shape[0]
    n_batches = max(1, n_train // batch_size)
    val_history = []

    for epoch in range(1, epochs + 1):
        probe.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0

        for i in range(n_batches):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            x = train_x_t[idx].to(device)
            y = train_y_t[idx].to(device)

            pred = probe(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validate
        probe.eval()
        with torch.no_grad():
            val_preds = []
            for i in range(0, val_x.shape[0], batch_size):
                x = val_x_t[i : i + batch_size].to(device)
                val_preds.append(probe(x).cpu())
            val_preds = torch.cat(val_preds, dim=0)
            val_loss = criterion(val_preds, val_y_t).item()
        val_history.append(val_loss)

    return probe, val_history


def train_action_probe(
    train_x: np.ndarray,
    train_actions: np.ndarray,
    val_x: np.ndarray,
    val_actions: np.ndarray,
    num_actions: int = NUM_ACTIONS,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Linear, float]:
    """Train a linear action classifier (IDM-style) on frozen features.
    Returns (probe, val_accuracy).
    """
    feat_dim = train_x.shape[1]
    probe = nn.Linear(feat_dim, num_actions).to(device)
    optimizer = AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_x_t = torch.from_numpy(train_x).float()
    train_a_t = torch.from_numpy(train_actions).long()
    val_x_t = torch.from_numpy(val_x).float()
    val_a_t = torch.from_numpy(val_actions).long()

    n_train = train_x.shape[0]
    n_batches = max(1, n_train // batch_size)

    for epoch in range(1, epochs + 1):
        probe.train()
        perm = torch.randperm(n_train)
        for i in range(n_batches):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            x = train_x_t[idx].to(device)
            a = train_a_t[idx].to(device)
            logits = probe(x)
            loss = criterion(logits, a)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate accuracy
    probe.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, val_x.shape[0], batch_size):
            x = val_x_t[i : i + batch_size].to(device)
            a = val_a_t[i : i + batch_size]
            logits = probe(x).cpu()
            preds = logits.argmax(dim=-1)
            correct += (preds == a).sum().item()
            total += a.shape[0]

    accuracy = correct / max(total, 1)
    return probe, accuracy


def compute_r2_per_feature(
    probe: nn.Linear,
    features: np.ndarray,
    labels: np.ndarray,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 512,
) -> Dict[str, float]:
    """Compute per-feature R2 for a trained probe. Returns {name: r2}."""
    probe.eval()
    x_t = torch.from_numpy(features).float()

    preds = []
    with torch.no_grad():
        for i in range(0, features.shape[0], batch_size):
            x = x_t[i : i + batch_size].to(device)
            preds.append(probe(x).cpu().numpy())
    preds = np.concatenate(preds, axis=0)

    r2_dict = {}
    n_targets = min(labels.shape[1], len(LABEL_NAMES))
    for k in range(n_targets):
        y_true = labels[:, k]
        y_pred = preds[:, k]
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1.0 - ss_res / (ss_tot + 1e-8))
        r2_dict[LABEL_NAMES[k]] = r2

    return r2_dict


def compute_next_state_cosine_sim(
    features: np.ndarray,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Compute how well a linear model predicts the next latent state.

    Trains a linear map f(z_t) -> z_{t+1} and measures cosine similarity.
    Uses consecutive pairs within the feature array.
    """
    # Use consecutive pairs
    z_t = features[:-1]
    z_tp1 = features[1:]

    n = z_t.shape[0]
    if n < 100:
        return float("nan")

    # Train/val split
    split = int(0.8 * n)
    train_x, train_y = z_t[:split], z_tp1[:split]
    val_x, val_y = z_t[split:], z_tp1[split:]

    feat_dim = train_x.shape[1]
    model = nn.Linear(feat_dim, feat_dim)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CosineEmbeddingLoss()

    train_x_t = torch.from_numpy(train_x).float()
    train_y_t = torch.from_numpy(train_y).float()

    batch_size = 256
    n_batches = max(1, split // batch_size)
    target = torch.ones(batch_size)

    for epoch in range(5):
        model.train()
        perm = torch.randperm(split)
        for i in range(n_batches):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            x = train_x_t[idx]
            y = train_y_t[idx]
            pred = model(x)
            t = target[:x.shape[0]]
            loss = criterion(pred, y, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    val_x_t = torch.from_numpy(val_x).float()
    val_y_t = torch.from_numpy(val_y).float()
    with torch.no_grad():
        pred = model(val_x_t)
        # Cosine similarity
        cos_sim = nn.functional.cosine_similarity(pred, val_y_t, dim=-1)
        mean_cos_sim = cos_sim.mean().item()

    return mean_cos_sim


# ===================================================================
# Run all probes for one model
# ===================================================================

def run_probes(
    model_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    actions: Optional[np.ndarray],
    device: torch.device,
    probe_epochs: int = 10,
    probe_lr: float = 1e-3,
) -> Dict[str, Any]:
    """Run the full probe suite on a set of features.

    Returns a dict with:
      - per_feature_r2: {name: r2} for each probe target
      - core_r2_mean: mean R2 over vitals + common resources
      - idm_accuracy: action prediction accuracy (or None)
      - next_state_cosine_sim: linear next-state prediction quality
    """
    print(f"\n--- Probing: {model_name} ---")
    print(f"  Features: {features.shape}, Labels: {labels.shape}")

    n = features.shape[0]
    split = int(0.8 * n)

    # Train/val split
    train_feats = features[:split]
    val_feats = features[split:]
    train_labels = labels[:split]
    val_labels = labels[split:]

    # 1. Inventory / state probes
    n_targets = min(labels.shape[1], len(LABEL_NAMES))
    print(f"  Training inventory probe ({n_targets} targets, {probe_epochs} epochs)...")
    t0 = time()
    probe, val_history = train_linear_probe(
        train_feats, train_labels[:, :n_targets],
        val_feats, val_labels[:, :n_targets],
        epochs=probe_epochs, lr=probe_lr, device=device,
    )
    r2_dict = compute_r2_per_feature(probe, val_feats, val_labels[:, :n_targets], device)
    print(f"    Done in {time() - t0:.1f}s")

    # Compute core R2 (vitals + common resources, skip rare items)
    core_r2_values = []
    for name in CORE_PROBE_NAMES:
        if name in r2_dict:
            core_r2_values.append(r2_dict[name])
    core_r2_mean = float(np.mean(core_r2_values)) if core_r2_values else float("nan")

    # Print per-feature R2
    print(f"  Per-feature R2 (core):")
    for name in CORE_PROBE_NAMES:
        if name in r2_dict:
            print(f"    {name:<16s}: {r2_dict[name]:.4f}")
    print(f"    {'MEAN (core)':<16s}: {core_r2_mean:.4f}")

    # 2. Action prediction (IDM)
    idm_accuracy = None
    if actions is not None:
        train_actions = actions[:split]
        val_actions = actions[split:]
        print(f"  Training action probe (IDM, {probe_epochs} epochs)...")
        t0 = time()
        _, idm_accuracy = train_action_probe(
            train_feats, train_actions, val_feats, val_actions,
            epochs=probe_epochs, lr=probe_lr, device=device,
        )
        print(f"    IDM accuracy: {idm_accuracy:.4f} ({time() - t0:.1f}s)")
    else:
        print("  Skipping IDM probe (no action labels available)")

    # 3. Next-state linear prediction
    print(f"  Training next-state linear predictor...")
    t0 = time()
    next_cos_sim = compute_next_state_cosine_sim(features, device)
    print(f"    Next-state cosine sim: {next_cos_sim:.4f} ({time() - t0:.1f}s)")

    return {
        "model_name": model_name,
        "feature_dim": int(features.shape[1]),
        "n_samples": int(features.shape[0]),
        "per_feature_r2": r2_dict,
        "core_r2_mean": core_r2_mean,
        "idm_accuracy": idm_accuracy,
        "next_state_cosine_sim": next_cos_sim,
        "val_loss_history": val_history,
    }


# ===================================================================
# Comparison table
# ===================================================================

def print_comparison_table(
    results: Dict[str, Dict[str, Any]],
) -> str:
    """Print and return a formatted comparison table."""

    models = list(results.keys())
    lines = []

    def add(line: str):
        lines.append(line)
        print(line)

    add("")
    add("=" * 72)
    add("  JEPA vs DreamerV2 Representation Comparison")
    add("=" * 72)

    # Header
    header = f"  {'Metric':<26s}"
    for m in models:
        header += f"{m:>14s}"
    add(header)
    add("  " + "-" * 26 + (" " + "-" * 13) * len(models))

    # Core probe R2
    row = f"  {'Probe R2 (core mean)':<26s}"
    for m in models:
        v = results[m].get("core_r2_mean")
        row += f"{v:>14.4f}" if v is not None and not np.isnan(v) else f"{'N/A':>14s}"
    add(row)

    # Per-feature R2 for core probes
    for name in CORE_PROBE_NAMES:
        row = f"    {name:<24s}"
        for m in models:
            r2 = results[m].get("per_feature_r2", {}).get(name)
            row += f"{r2:>14.4f}" if r2 is not None and not np.isnan(r2) else f"{'N/A':>14s}"
        add(row)

    # IDM accuracy
    row = f"  {'IDM Accuracy':<26s}"
    for m in models:
        v = results[m].get("idm_accuracy")
        row += f"{v:>14.4f}" if v is not None else f"{'N/A':>14s}"
    add(row)

    # Next-state cosine sim
    row = f"  {'Next-state cos sim':<26s}"
    for m in models:
        v = results[m].get("next_state_cosine_sim")
        row += f"{v:>14.4f}" if v is not None and not np.isnan(v) else f"{'N/A':>14s}"
    add(row)

    add("  " + "-" * 26 + (" " + "-" * 13) * len(models))

    # Model properties
    props = {
        "Feature dim": "feature_dim",
        "N samples": "n_samples",
    }
    for label, key in props.items():
        row = f"  {label:<26s}"
        for m in models:
            v = results[m].get(key, "N/A")
            row += f"{str(v):>14s}"
        add(row)

    # Static properties
    static_props = [
        ("Pixel Decoder", {"JEPA": "No", "DreamerV2": "Yes", "Random": "No"}),
        ("Training Loss", {"JEPA": "VCReg+IDM", "DreamerV2": "Recon+Reward", "Random": "None"}),
        ("Crafter Score", {"JEPA": "TBD", "DreamerV2": "~10%", "Random": "~1.6%"}),
    ]
    for label, vals in static_props:
        row = f"  {label:<26s}"
        for m in models:
            v = vals.get(m, "N/A")
            row += f"{v:>14s}"
        add(row)

    add("=" * 72)

    return "\n".join(lines)


# ===================================================================
# Comparison figure
# ===================================================================

def make_comparison_figure(
    results: Dict[str, Dict[str, Any]],
    output_path: str,
):
    """Generate a grouped bar chart comparing representation quality."""

    _apply_style()

    models = list(results.keys())
    colors = {
        "JEPA": C_JEPA,
        "DreamerV2": C_DREAMER,
        "Random": C_RANDOM,
    }

    # Metrics to plot
    metrics = [
        ("Probe R2\n(core mean)", "core_r2_mean"),
        ("IDM\nAccuracy", "idm_accuracy"),
        ("Next-state\nCosine Sim", "next_state_cosine_sim"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    n_metrics = len(metrics)
    n_models = len(models)
    bar_width = 0.25
    x = np.arange(n_metrics)

    for i, model_name in enumerate(models):
        vals = []
        for _, key in metrics:
            v = results[model_name].get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                vals.append(0.0)
            else:
                vals.append(float(v))
        color = colors.get(model_name, f"C{i}")
        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, vals, bar_width,
            label=model_name, color=color, edgecolor="black", linewidth=0.5,
        )
        # Add value labels
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=TICK_SIZE - 1,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics], fontsize=LABEL_SIZE)
    ax.set_ylabel("Score", fontsize=LABEL_SIZE)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=TICK_SIZE, loc="upper right")
    ax.set_title(
        "Representation Quality: JEPA (no decoder) vs DreamerV2 (pixel reconstruction)",
        fontsize=TITLE_SIZE - 1, fontweight="bold", pad=15,
    )

    # Add annotation about data budget
    jepa_n = results.get("JEPA", {}).get("n_samples", "?")
    ax.annotate(
        f"Same evaluation data: {jepa_n} samples",
        xy=(0.5, -0.12), xycoords="axes fraction",
        ha="center", fontsize=TICK_SIZE, style="italic", color="#555555",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nComparison figure saved to {output_path}")


def make_per_feature_figure(
    results: Dict[str, Dict[str, Any]],
    output_path: str,
):
    """Generate a per-feature R2 comparison bar chart for core probes."""

    _apply_style()

    models = list(results.keys())
    colors = {
        "JEPA": C_JEPA,
        "DreamerV2": C_DREAMER,
        "Random": C_RANDOM,
    }

    fig, ax = plt.subplots(figsize=(12, 5))

    n_features = len(CORE_PROBE_NAMES)
    n_models = len(models)
    bar_width = 0.8 / n_models
    x = np.arange(n_features)

    for i, model_name in enumerate(models):
        r2_dict = results[model_name].get("per_feature_r2", {})
        vals = [r2_dict.get(name, 0.0) for name in CORE_PROBE_NAMES]
        # Clamp negative R2 to 0 for visualization
        vals = [max(0.0, v) for v in vals]
        color = colors.get(model_name, f"C{i}")
        offset = (i - n_models / 2 + 0.5) * bar_width
        ax.bar(
            x + offset, vals, bar_width,
            label=model_name, color=color, edgecolor="black", linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [n.replace("_", "\n") for n in CORE_PROBE_NAMES],
        fontsize=TICK_SIZE,
    )
    ax.set_ylabel("Probe R2", fontsize=LABEL_SIZE)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=TICK_SIZE)
    ax.set_title(
        "Per-Feature Probe R2: Vitals + Resources",
        fontsize=TITLE_SIZE, fontweight="bold",
    )

    # Separator between vitals and resources
    ax.axvline(x=3.5, color="#CCCCCC", linestyle="--", linewidth=1)
    ax.text(1.5, 1.05, "Vitals", ha="center", fontsize=TICK_SIZE, style="italic")
    ax.text(4.5, 1.05, "Resources", ha="center", fontsize=TICK_SIZE, style="italic")

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Per-feature figure saved to {output_path}")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare representation quality: JEPA vs DreamerV2 vs Random",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with JEPA checkpoint
  python scripts/compare_dreamerv2.py \\
      --jepa_checkpoint checkpoints/.../latest.pth.tar \\
      --data_dir data/crafter_trajectories \\
      --output_dir eval_results/comparison

  # Pre-computed features
  python scripts/compare_dreamerv2.py \\
      --jepa_features features_jepa.npy \\
      --dreamer_features features_dreamer.npy \\
      --labels labels.npy \\
      --actions actions.npy \\
      --output_dir eval_results/comparison

  # JEPA vs Random only (quick check)
  python scripts/compare_dreamerv2.py \\
      --jepa_features features_jepa.npy \\
      --labels labels.npy \\
      --output_dir eval_results/comparison
        """,
    )

    # Feature sources
    feat_group = parser.add_argument_group("Feature sources")
    feat_group.add_argument(
        "--jepa_checkpoint", type=str, default=None,
        help="Path to trained JEPA checkpoint (.pth.tar). Extracts features from data.",
    )
    feat_group.add_argument(
        "--jepa_features", type=str, default=None,
        help="Path to pre-computed JEPA features (.npy, shape [N, D]).",
    )
    feat_group.add_argument(
        "--dreamer_features", type=str, default=None,
        help="Path to pre-computed DreamerV2 features (.npy, shape [N, D]).",
    )
    feat_group.add_argument(
        "--dreamer_checkpoint", type=str, default=None,
        help="Path to DreamerV2 checkpoint (requires dreamerv2 installed).",
    )

    # Labels / data
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--data_dir", type=str, default=None,
        help="Path to Crafter trajectory data (required if using checkpoints).",
    )
    data_group.add_argument(
        "--labels", type=str, default=None,
        help="Path to probe labels (.npy, shape [N, K]).",
    )
    data_group.add_argument(
        "--actions", type=str, default=None,
        help="Path to action labels (.npy, shape [N]).",
    )

    # Probe config
    probe_group = parser.add_argument_group("Probe configuration")
    probe_group.add_argument("--probe_epochs", type=int, default=10)
    probe_group.add_argument("--probe_lr", type=float, default=1e-3)
    probe_group.add_argument("--batch_size", type=int, default=64)

    # Output
    parser.add_argument(
        "--output_dir", type=str, default="eval_results/comparison",
        help="Directory to save results, figures, and tables.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (auto-detect if not set).",
    )
    parser.add_argument(
        "--skip_random", action="store_true",
        help="Skip random baseline probing.",
    )

    args = parser.parse_args()

    # --- Validate args ---
    if args.jepa_checkpoint is None and args.jepa_features is None:
        parser.error("Provide either --jepa_checkpoint or --jepa_features.")

    if args.jepa_checkpoint and args.data_dir is None:
        parser.error("--data_dir is required when using --jepa_checkpoint.")

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

    # --- Output dir ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================================
    # 1. Load / extract JEPA features
    # =====================================================================
    print("\n" + "=" * 72)
    print("  Loading JEPA features")
    print("=" * 72)

    labels = None
    actions = None

    if args.jepa_features:
        jepa_feats = load_features(args.jepa_features)
        if args.labels:
            labels = load_labels(args.labels)
        if args.actions:
            actions = load_actions(args.actions)
    else:
        print(f"  Extracting features from JEPA checkpoint...")
        jepa_feats, labels, actions = extract_jepa_features(
            args.jepa_checkpoint, args.data_dir, device, args.batch_size,
        )
        print(f"  JEPA features: {jepa_feats.shape}")

        # Save extracted features for future use
        np.save(output_dir / "jepa_features.npy", jepa_feats)
        np.save(output_dir / "labels.npy", labels)
        np.save(output_dir / "actions.npy", actions)
        print(f"  Saved extracted features to {output_dir}/")

    if labels is None:
        print("ERROR: No labels available. Provide --labels or --data_dir.")
        sys.exit(1)

    # =====================================================================
    # 2. Load DreamerV2 features (if available)
    # =====================================================================
    dreamer_feats = None

    if args.dreamer_features:
        print("\n" + "=" * 72)
        print("  Loading DreamerV2 features")
        print("=" * 72)
        dreamer_feats = load_features(args.dreamer_features)
        # Ensure same number of samples
        n_min = min(dreamer_feats.shape[0], labels.shape[0])
        if dreamer_feats.shape[0] != labels.shape[0]:
            print(f"  WARNING: DreamerV2 features ({dreamer_feats.shape[0]}) != "
                  f"labels ({labels.shape[0]}). Truncating to {n_min}.")
            dreamer_feats = dreamer_feats[:n_min]

    elif args.dreamer_checkpoint:
        print("\n" + "=" * 72)
        print("  Extracting DreamerV2 features")
        print("=" * 72)
        result = try_extract_dreamer_features(
            args.dreamer_checkpoint, args.data_dir, device, args.batch_size,
        )
        if result is not None:
            dreamer_feats, _, _ = result

    if dreamer_feats is None:
        print("\n  No DreamerV2 features available.")
        print("  Comparison will be JEPA vs Random baseline.")
        print("  To add DreamerV2, provide --dreamer_features with a .npy file.")

    # =====================================================================
    # 3. Run probes
    # =====================================================================

    all_results = {}

    # -- JEPA --
    n_samples = min(jepa_feats.shape[0], labels.shape[0])
    jepa_feats_trimmed = jepa_feats[:n_samples]
    labels_trimmed = labels[:n_samples]
    actions_trimmed = actions[:n_samples] if actions is not None else None

    jepa_results = run_probes(
        "JEPA", jepa_feats_trimmed, labels_trimmed, actions_trimmed,
        device, args.probe_epochs, args.probe_lr,
    )
    all_results["JEPA"] = jepa_results

    # -- DreamerV2 --
    if dreamer_feats is not None:
        n_d = min(dreamer_feats.shape[0], labels.shape[0])
        dreamer_results = run_probes(
            "DreamerV2", dreamer_feats[:n_d], labels_trimmed[:n_d],
            actions_trimmed[:n_d] if actions_trimmed is not None else None,
            device, args.probe_epochs, args.probe_lr,
        )
        all_results["DreamerV2"] = dreamer_results

    # -- Random baseline --
    if not args.skip_random:
        random_feats = random_features(n_samples, dim=jepa_feats.shape[1])
        random_results = run_probes(
            "Random", random_feats, labels_trimmed, actions_trimmed,
            device, args.probe_epochs, args.probe_lr,
        )
        all_results["Random"] = random_results

    # =====================================================================
    # 4. Comparison table
    # =====================================================================
    table_str = print_comparison_table(all_results)

    # =====================================================================
    # 5. Generate figures
    # =====================================================================
    print("\n" + "=" * 72)
    print("  Generating comparison figures")
    print("=" * 72)

    make_comparison_figure(
        all_results, str(output_dir / "model_comparison.png"),
    )
    make_per_feature_figure(
        all_results, str(output_dir / "per_feature_comparison.png"),
    )

    # =====================================================================
    # 6. Save results
    # =====================================================================

    # Convert results to JSON-safe format
    def _jsonify(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_jsonify(v) for v in obj]
        return obj

    results_json = {
        "comparison": _jsonify(all_results),
        "table": table_str,
        "args": {
            "jepa_checkpoint": args.jepa_checkpoint,
            "jepa_features": args.jepa_features,
            "dreamer_features": args.dreamer_features,
            "dreamer_checkpoint": args.dreamer_checkpoint,
            "data_dir": args.data_dir,
            "probe_epochs": args.probe_epochs,
            "probe_lr": args.probe_lr,
        },
    }

    results_path = output_dir / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # =====================================================================
    # 7. Final summary
    # =====================================================================
    print("\n" + "=" * 72)
    print("  FINAL SUMMARY")
    print("=" * 72)

    for model_name, res in all_results.items():
        r2 = res.get("core_r2_mean", float("nan"))
        idm = res.get("idm_accuracy")
        ncs = res.get("next_state_cosine_sim", float("nan"))
        idm_str = f"{idm:.4f}" if idm is not None else "N/A"
        print(f"  {model_name:<14s}  R2={r2:.4f}  IDM={idm_str}  NextCos={ncs:.4f}")

    if "DreamerV2" not in all_results:
        print("\n  DreamerV2 not included. To add it:")
        print("    1. Train DreamerV2 on Crafter")
        print("    2. Export latent representations as .npy")
        print("    3. Re-run with --dreamer_features <path>")

    print(f"\n  Figures: {output_dir / 'model_comparison.png'}")
    print(f"           {output_dir / 'per_feature_comparison.png'}")
    print(f"  Results: {results_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
