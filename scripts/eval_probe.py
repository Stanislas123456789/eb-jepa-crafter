#!/usr/bin/env python3
"""
Probe evaluation for JEPA world model on Crafter.

Freezes the JEPA encoder and trains small linear probes on top of frozen latent
representations to predict game state (vitals, resources, tools). Compares a
trained encoder against a random (untrained) baseline to demonstrate that the
learned representations capture meaningful world knowledge.

Usage:
    python scripts/eval_probe.py \
        --checkpoint_path checkpoints/latest.pth.tar \
        --data_dir data/crafter_trajectories
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so imports work when running as a
# standalone script (python scripts/eval_probe.py).
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
# Probe label names (K=16)
# ---------------------------------------------------------------------------
LABEL_NAMES = [
    # Vitals (0-3): regression, range 0-9
    "health",
    "food",
    "drink",
    "energy",
    # Resources (4-9): regression, counts
    "sapling",
    "wood",
    "stone",
    "coal",
    "iron",
    "diamond",
    # Tools (10-15): binary classification
    "wood_pickaxe",
    "stone_pickaxe",
    "iron_pickaxe",
    "wood_sword",
    "stone_sword",
    "iron_sword",
]

BINARY_INDICES = list(range(10, 16))
REGRESSION_INDICES = list(range(0, 10))

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
    # Regularizer params (needed to construct JEPA, but irrelevant for probing)
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

    # Compute spatial dims from a test input
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
    # Strip torch.compile prefixes if present
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


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_dataset(jepa: JEPA, loader, device: torch.device):
    """
    Run the frozen JEPA encoder over the dataset.

    Returns:
        features: [N*T, 512] tensor of latent representations
        labels:   [N*T, 16]  tensor of probe labels
    """
    jepa.eval()
    all_features = []
    all_labels = []

    for batch in loader:
        obs = batch[0].to(device)         # [B, C, T, H, W]
        probe_labels = batch[2]           # [B, K, T]

        # Encode: output is [B, 512, T, 1, 1]
        enc = jepa.encode(obs)            # [B, D, T, 1, 1]
        B, D, T, _, _ = enc.shape

        # Flatten to [B*T, D]
        enc_flat = enc.squeeze(-1).squeeze(-1)  # [B, D, T]
        enc_flat = enc_flat.permute(0, 2, 1).reshape(B * T, D)  # [B*T, D]

        # Flatten labels to [B*T, K]
        labels_flat = probe_labels.permute(0, 2, 1).reshape(B * T, -1)  # [B*T, K]

        all_features.append(enc_flat.cpu())
        all_labels.append(labels_flat)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return features, labels


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------

def train_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
):
    """
    Train a linear probe (nn.Linear(512, 16)) on frozen features.

    Returns the trained probe and per-epoch val loss history.
    """
    feat_dim = train_features.shape[1]
    num_targets = train_labels.shape[1]
    probe = nn.Linear(feat_dim, num_targets).to(device)
    optimizer = AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    n_train = train_features.shape[0]
    n_batches = max(1, n_train // batch_size)

    val_history = []

    for epoch in range(1, epochs + 1):
        # --- Train ---
        probe.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0

        for i in range(n_batches):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            x = train_features[idx].to(device)
            y = train_labels[idx].to(device)

            pred = probe(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / n_batches

        # --- Validate ---
        probe.eval()
        with torch.no_grad():
            val_pred = []
            n_val = val_features.shape[0]
            for i in range(0, n_val, batch_size):
                x = val_features[i : i + batch_size].to(device)
                val_pred.append(probe(x).cpu())
            val_pred = torch.cat(val_pred, dim=0)
            val_loss = criterion(val_pred, val_labels).item()

        val_history.append(val_loss)
        print(
            f"  Epoch {epoch:>2d}/{epochs} | "
            f"Train MSE: {avg_train_loss:.6f} | Val MSE: {val_loss:.6f}"
        )

    return probe, val_history


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def evaluate_probe(
    probe: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 512,
    device: torch.device = torch.device("cpu"),
):
    """
    Evaluate a trained probe on a dataset.

    Returns a dict with per-feature MSE, R-squared, and binary accuracy.
    """
    probe.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, features.shape[0], batch_size):
            x = features[i : i + batch_size].to(device)
            preds.append(probe(x).cpu())
        preds = torch.cat(preds, dim=0)  # [N, 16]

    results = {}
    n = labels.shape[0]

    for k in range(labels.shape[1]):
        y_true = labels[:, k].numpy()
        y_pred = preds[:, k].numpy()

        # MSE
        mse = float(np.mean((y_true - y_pred) ** 2))

        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1.0 - ss_res / (ss_tot + 1e-8))

        feat_result = {
            "name": LABEL_NAMES[k],
            "mse": mse,
            "r2": r2,
        }

        # Binary accuracy for tool features
        if k in BINARY_INDICES:
            binary_pred = (y_pred > 0.5).astype(np.float32)
            accuracy = float(np.mean(binary_pred == y_true))
            feat_result["accuracy"] = accuracy

        results[LABEL_NAMES[k]] = feat_result

    return results


def print_results_table(results: dict, title: str = "Results"):
    """Print a formatted results table."""
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")
    print(f"  {'Feature':<18} {'MSE':>10} {'R2':>10} {'Accuracy':>10}")
    print(f"  {'-' * 18} {'-' * 10} {'-' * 10} {'-' * 10}")

    for name in LABEL_NAMES:
        r = results[name]
        acc_str = f"{r['accuracy']:.4f}" if "accuracy" in r else "   --"
        print(f"  {name:<18} {r['mse']:>10.6f} {r['r2']:>10.4f} {acc_str:>10}")

    # Averages
    all_mse = [results[n]["mse"] for n in LABEL_NAMES]
    all_r2 = [results[n]["r2"] for n in LABEL_NAMES]
    binary_acc = [results[LABEL_NAMES[i]]["accuracy"] for i in BINARY_INDICES]

    print(f"  {'-' * 18} {'-' * 10} {'-' * 10} {'-' * 10}")
    print(f"  {'MEAN (all)':<18} {np.mean(all_mse):>10.6f} {np.mean(all_r2):>10.4f} {'--':>10}")
    print(f"  {'MEAN (vitals)':<18} {'':>10} {np.mean([results[LABEL_NAMES[i]]['r2'] for i in range(4)]):>10.4f} {'--':>10}")
    print(f"  {'MEAN (resources)':<18} {'':>10} {np.mean([results[LABEL_NAMES[i]]['r2'] for i in range(4, 10)]):>10.4f} {'--':>10}")
    print(f"  {'MEAN (tools)':<18} {'':>10} {np.mean([results[LABEL_NAMES[i]]['r2'] for i in BINARY_INDICES]):>10.4f} {np.mean(binary_acc):>10.4f}")
    print(f"{'=' * 72}")


def print_comparison_table(trained_results: dict, random_results: dict):
    """Print a side-by-side comparison of trained vs random encoder."""
    print(f"\n{'=' * 82}")
    print(f"  COMPARISON: Trained Encoder vs Random Encoder")
    print(f"{'=' * 82}")
    print(f"  {'Feature':<18} {'Trained R2':>12} {'Random R2':>12} {'Delta':>10} {'Lift':>10}")
    print(f"  {'-' * 18} {'-' * 12} {'-' * 12} {'-' * 10} {'-' * 10}")

    for name in LABEL_NAMES:
        t_r2 = trained_results[name]["r2"]
        r_r2 = random_results[name]["r2"]
        delta = t_r2 - r_r2
        # Relative lift over random (handle edge cases)
        if abs(r_r2) > 0.01:
            lift_str = f"{delta / abs(r_r2) * 100:>8.1f}%"
        elif delta > 0.01:
            lift_str = "    +inf"
        else:
            lift_str = "    ~0"
        print(f"  {name:<18} {t_r2:>12.4f} {r_r2:>12.4f} {delta:>+10.4f} {lift_str:>10}")

    # Overall
    t_mean = np.mean([trained_results[n]["r2"] for n in LABEL_NAMES])
    r_mean = np.mean([random_results[n]["r2"] for n in LABEL_NAMES])
    delta_mean = t_mean - r_mean

    print(f"  {'-' * 18} {'-' * 12} {'-' * 12} {'-' * 10} {'-' * 10}")
    print(f"  {'MEAN R2':<18} {t_mean:>12.4f} {r_mean:>12.4f} {delta_mean:>+10.4f}")
    print(f"{'=' * 82}")

    # Binary accuracy comparison
    print(f"\n  Binary Tools Accuracy:")
    for i in BINARY_INDICES:
        name = LABEL_NAMES[i]
        t_acc = trained_results[name].get("accuracy", 0)
        r_acc = random_results[name].get("accuracy", 0)
        print(f"    {name:<18} Trained: {t_acc:.4f}  Random: {r_acc:.4f}  Delta: {t_acc - r_acc:+.4f}")

    t_mean_acc = np.mean([trained_results[LABEL_NAMES[i]]["accuracy"] for i in BINARY_INDICES])
    r_mean_acc = np.mean([random_results[LABEL_NAMES[i]]["accuracy"] for i in BINARY_INDICES])
    print(f"    {'MEAN':<18} Trained: {t_mean_acc:.4f}  Random: {r_mean_acc:.4f}  Delta: {t_mean_acc - r_mean_acc:+.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Probe evaluation for JEPA world model on Crafter"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to trained JEPA checkpoint (.pth.tar)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to crafter trajectory data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Directory to save results JSON (default: eval_results)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of probe training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for data loading (default: 64)",
    )
    parser.add_argument(
        "--probe_lr",
        type=float,
        default=1e-3,
        help="Learning rate for probe training (default: 1e-3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
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
        "persistent_workers": False,  # Avoid issues with short runs
        "sample_length": 17,
    }
    train_loader, val_loader, data_config = init_data("crafter", cfg_data)
    print(f"Train: {data_config.size} slices, Val: {data_config.val_size} slices")

    model_cfg = dict(DEFAULT_MODEL_CFG)

    # =====================================================================
    # 1. Trained encoder probe
    # =====================================================================
    print("\n" + "=" * 72)
    print("  PART 1: Probe with TRAINED encoder")
    print("=" * 72)

    # Load trained JEPA
    jepa_trained = load_trained_jepa(args.checkpoint_path, model_cfg, device)
    jepa_trained.eval()

    # Encode datasets
    print("\nEncoding training set with trained encoder...")
    t0 = time()
    train_feats, train_labels = encode_dataset(jepa_trained, train_loader, device)
    print(f"  Encoded {train_feats.shape[0]} samples in {time() - t0:.1f}s -> features {train_feats.shape}")

    print("Encoding validation set with trained encoder...")
    t0 = time()
    val_feats, val_labels = encode_dataset(jepa_trained, val_loader, device)
    print(f"  Encoded {val_feats.shape[0]} samples in {time() - t0:.1f}s -> features {val_feats.shape}")

    # Free GPU memory
    del jepa_trained
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Train probe
    print("\nTraining linear probe on trained encoder features...")
    probe_trained, train_history = train_probe(
        train_feats, train_labels,
        val_feats, val_labels,
        epochs=args.epochs,
        batch_size=256,
        lr=args.probe_lr,
        device=device,
    )

    # Evaluate
    trained_results = evaluate_probe(probe_trained, val_feats, val_labels, device=device)
    print_results_table(trained_results, title="TRAINED Encoder Probe Results")

    # =====================================================================
    # 2. Random encoder baseline
    # =====================================================================
    print("\n" + "=" * 72)
    print("  PART 2: Probe with RANDOM (untrained) encoder")
    print("=" * 72)

    jepa_random = build_random_jepa(model_cfg, device)
    jepa_random.eval()

    print("\nEncoding training set with random encoder...")
    t0 = time()
    train_feats_rand, train_labels_rand = encode_dataset(jepa_random, train_loader, device)
    print(f"  Encoded {train_feats_rand.shape[0]} samples in {time() - t0:.1f}s")

    print("Encoding validation set with random encoder...")
    t0 = time()
    val_feats_rand, val_labels_rand = encode_dataset(jepa_random, val_loader, device)
    print(f"  Encoded {val_feats_rand.shape[0]} samples in {time() - t0:.1f}s")

    del jepa_random
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\nTraining linear probe on random encoder features...")
    probe_random, rand_history = train_probe(
        train_feats_rand, train_labels_rand,
        val_feats_rand, val_labels_rand,
        epochs=args.epochs,
        batch_size=256,
        lr=args.probe_lr,
        device=device,
    )

    random_results = evaluate_probe(probe_random, val_feats_rand, val_labels_rand, device=device)
    print_results_table(random_results, title="RANDOM Encoder Probe Results")

    # =====================================================================
    # 3. Comparison
    # =====================================================================
    print_comparison_table(trained_results, random_results)

    # =====================================================================
    # 4. Save results
    # =====================================================================
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_json = {
        "checkpoint_path": str(args.checkpoint_path),
        "data_dir": str(args.data_dir),
        "probe_epochs": args.epochs,
        "probe_lr": args.probe_lr,
        "batch_size": args.batch_size,
        "train_samples": int(train_feats.shape[0]),
        "val_samples": int(val_feats.shape[0]),
        "feature_dim": int(train_feats.shape[1]),
        "label_names": LABEL_NAMES,
        "binary_indices": BINARY_INDICES,
        "trained_encoder": {
            "per_feature": trained_results,
            "mean_r2": float(np.mean([trained_results[n]["r2"] for n in LABEL_NAMES])),
            "mean_mse": float(np.mean([trained_results[n]["mse"] for n in LABEL_NAMES])),
            "mean_binary_accuracy": float(
                np.mean([trained_results[LABEL_NAMES[i]]["accuracy"] for i in BINARY_INDICES])
            ),
            "val_loss_history": train_history,
        },
        "random_encoder": {
            "per_feature": random_results,
            "mean_r2": float(np.mean([random_results[n]["r2"] for n in LABEL_NAMES])),
            "mean_mse": float(np.mean([random_results[n]["mse"] for n in LABEL_NAMES])),
            "mean_binary_accuracy": float(
                np.mean([random_results[LABEL_NAMES[i]]["accuracy"] for i in BINARY_INDICES])
            ),
            "val_loss_history": rand_history,
        },
        "summary": {
            "r2_lift": float(
                np.mean([trained_results[n]["r2"] for n in LABEL_NAMES])
                - np.mean([random_results[n]["r2"] for n in LABEL_NAMES])
            ),
            "accuracy_lift": float(
                np.mean([trained_results[LABEL_NAMES[i]]["accuracy"] for i in BINARY_INDICES])
                - np.mean([random_results[LABEL_NAMES[i]]["accuracy"] for i in BINARY_INDICES])
            ),
        },
    }

    output_path = output_dir / "probe_results.json"
    with open(output_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # --- Final summary ---
    print("\n" + "=" * 72)
    print("  FINAL SUMMARY")
    print("=" * 72)
    t_r2 = results_json["trained_encoder"]["mean_r2"]
    r_r2 = results_json["random_encoder"]["mean_r2"]
    t_acc = results_json["trained_encoder"]["mean_binary_accuracy"]
    r_acc = results_json["random_encoder"]["mean_binary_accuracy"]
    print(f"  Trained encoder mean R2:           {t_r2:.4f}")
    print(f"  Random encoder mean R2:            {r_r2:.4f}")
    print(f"  R2 improvement:                    {t_r2 - r_r2:+.4f}")
    print(f"  Trained encoder binary accuracy:   {t_acc:.4f}")
    print(f"  Random encoder binary accuracy:    {r_acc:.4f}")
    print(f"  Accuracy improvement:              {t_acc - r_acc:+.4f}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
