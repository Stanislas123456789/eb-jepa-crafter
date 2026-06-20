#!/usr/bin/env python3
"""
Train a reward prediction head on top of a frozen JEPA encoder.

Loads a trained JEPA checkpoint, freezes the encoder, and trains a small MLP
to predict per-timestep rewards from the latent representations. The trained
reward head is saved for use by the planning agent.

Usage:
    python scripts/train_reward_head.py \
        --checkpoint_path checkpoints/latest.pth.tar \
        --data_dir data/crafter_trajectories \
        --output_dir eval_results/reward_head
"""

import argparse
import glob
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
# standalone script (python scripts/train_reward_head.py).
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
from eb_jepa.datasets.crafter.normalizer import CrafterNormalizer
from eb_jepa.jepa import JEPA
from eb_jepa.losses import SquareLossSeq, VC_IDM_Sim_Regularizer

# ---------------------------------------------------------------------------
# Default Crafter JEPA model config (matches crafter.yaml / eval_probe.py)
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


# ---------------------------------------------------------------------------
# Reward Head
# ---------------------------------------------------------------------------
class RewardHead(nn.Module):
    """Small MLP that predicts scalar reward from JEPA latent."""

    def __init__(self, input_dim=512, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent):
        # latent: [B, 512, T, 1, 1] or [B, 512] or [N, 512]
        if latent.dim() == 5:
            B, D, T, H, W = latent.shape
            latent = (
                latent.squeeze(-1).squeeze(-1).permute(0, 2, 1).reshape(-1, D)
            )  # [B*T, 512]
        return self.net(latent).squeeze(-1)  # [N]


# ---------------------------------------------------------------------------
# Model construction (reused from eval_probe.py)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_episodes(data_dir: str):
    """Load all episode .npz files and return observations + rewards."""
    pattern = os.path.join(data_dir, "episode_*.npz")
    paths = sorted(glob.glob(pattern))
    if len(paths) == 0:
        raise FileNotFoundError(f"No episode_*.npz files found in {data_dir}")

    episodes = []
    total_frames = 0
    for path in paths:
        data = np.load(path)
        obs = data["observations"]  # [T, 64, 64, 3] uint8
        rewards = data["rewards"]   # [T] float32
        episodes.append({"observations": obs, "rewards": rewards})
        total_frames += obs.shape[0]

    print(f"Loaded {len(episodes)} episodes ({total_frames} total frames) from {data_dir}")
    return episodes


def compute_normalizer(episodes, num_samples=5000):
    """Compute per-channel normalization stats from raw episodes."""
    rng = np.random.RandomState(42)
    frames = []
    for _ in range(num_samples):
        ep = episodes[rng.randint(0, len(episodes))]
        t = rng.randint(0, ep["observations"].shape[0])
        frame = ep["observations"][t].astype(np.float32) / 255.0
        frames.append(frame)
    frames = np.stack(frames, axis=0)  # [N, H, W, C]
    mean = frames.mean(axis=(0, 1, 2))
    std = frames.std(axis=(0, 1, 2))
    return CrafterNormalizer(
        mean=torch.tensor(mean, dtype=torch.float32),
        std=torch.tensor(std, dtype=torch.float32),
    )


@torch.no_grad()
def encode_episodes(jepa, episodes, normalizer, device, batch_frames=512):
    """
    Encode all episode observations through the frozen JEPA encoder.

    Returns:
        features: [N_total, 512] tensor
        rewards:  [N_total] tensor
    """
    jepa.eval()
    all_features = []
    all_rewards = []

    for ep_idx, ep in enumerate(episodes):
        obs_np = ep["observations"]  # [T, 64, 64, 3] uint8
        rewards_np = ep["rewards"]   # [T] float32
        T = obs_np.shape[0]

        # Process episode in chunks to avoid OOM
        for start in range(0, T, batch_frames):
            end = min(start + batch_frames, T)
            chunk = obs_np[start:end]  # [chunk_T, 64, 64, 3]
            chunk_T = chunk.shape[0]

            # Convert to tensor: [chunk_T, H, W, C] -> [C, chunk_T, H, W]
            obs_t = torch.from_numpy(chunk.copy()).float() / 255.0
            obs_t = obs_t.permute(3, 0, 1, 2)  # [C, T, H, W]
            obs_t = normalizer.normalize_state(obs_t)
            # Add batch dim: [1, C, T, H, W]
            obs_t = obs_t.unsqueeze(0).to(device)

            # Encode: [1, D, T, 1, 1]
            enc = jepa.encode(obs_t)
            B, D, T_enc, _, _ = enc.shape
            # Flatten to [T, D]
            enc_flat = enc.squeeze(0).squeeze(-1).squeeze(-1)  # [D, T]
            enc_flat = enc_flat.permute(1, 0)  # [T, D]

            all_features.append(enc_flat.cpu())
            all_rewards.append(
                torch.from_numpy(rewards_np[start:end].copy()).float()
            )

        if (ep_idx + 1) % 50 == 0:
            print(f"  Encoded {ep_idx + 1}/{len(episodes)} episodes...")

    features = torch.cat(all_features, dim=0)
    rewards = torch.cat(all_rewards, dim=0)
    return features, rewards


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_reward_head(
    train_features,
    train_rewards,
    val_features,
    val_rewards,
    input_dim=512,
    hidden_dim=128,
    epochs=20,
    batch_size=256,
    lr=1e-3,
    device=torch.device("cpu"),
):
    """
    Train a RewardHead on frozen latent features.

    Uses Huber loss (smooth L1) to handle sparse reward distribution gracefully.
    Returns the trained head and training history.
    """
    head = RewardHead(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    optimizer = AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()  # Huber loss -- robust to sparse rewards

    n_train = train_features.shape[0]
    n_batches = max(1, n_train // batch_size)

    # Reward stats for context
    nonzero_train = (train_rewards != 0).sum().item()
    nonzero_val = (val_rewards != 0).sum().item()
    print(f"  Train: {n_train} samples, {nonzero_train} non-zero rewards "
          f"({100*nonzero_train/n_train:.1f}%)")
    print(f"  Val:   {val_features.shape[0]} samples, {nonzero_val} non-zero rewards "
          f"({100*nonzero_val/val_features.shape[0]:.1f}%)")

    history = []

    for epoch in range(1, epochs + 1):
        # --- Train ---
        head.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0

        for i in range(n_batches):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            x = train_features[idx].to(device)
            y = train_rewards[idx].to(device)

            pred = head(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / n_batches

        # --- Validate ---
        head.eval()
        with torch.no_grad():
            val_preds = []
            n_val = val_features.shape[0]
            for i in range(0, n_val, batch_size):
                x = val_features[i : i + batch_size].to(device)
                val_preds.append(head(x).cpu())
            val_preds = torch.cat(val_preds, dim=0)
            val_loss = criterion(val_preds, val_rewards).item()

        history.append({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": val_loss})
        print(
            f"  Epoch {epoch:>2d}/{epochs} | "
            f"Train Huber: {avg_train_loss:.6f} | Val Huber: {val_loss:.6f}"
        )

    return head, history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_reward_head(head, features, rewards, device, batch_size=512):
    """
    Evaluate the trained reward head.

    Returns dict with MSE, R-squared, and sign-match accuracy (for non-zero rewards).
    """
    head.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, features.shape[0], batch_size):
            x = features[i : i + batch_size].to(device)
            preds.append(head(x).cpu())
        preds = torch.cat(preds, dim=0)

    y_true = rewards.numpy()
    y_pred = preds.numpy()

    # MSE
    mse = float(np.mean((y_true - y_pred) ** 2))

    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-8))

    # MAE
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # Sign-match accuracy on non-zero rewards (does the head know reward direction?)
    nonzero_mask = y_true != 0
    n_nonzero = nonzero_mask.sum()
    if n_nonzero > 0:
        sign_match = float(
            np.mean(np.sign(y_pred[nonzero_mask]) == np.sign(y_true[nonzero_mask]))
        )
    else:
        sign_match = float("nan")

    # Zero vs non-zero detection: can the head tell when a reward occurs?
    # Threshold: predict non-zero if |pred| > 0.05
    pred_nonzero = np.abs(y_pred) > 0.05
    true_nonzero = y_true != 0
    if true_nonzero.sum() > 0:
        precision = float(
            np.sum(pred_nonzero & true_nonzero) / (np.sum(pred_nonzero) + 1e-8)
        )
        recall = float(
            np.sum(pred_nonzero & true_nonzero) / (np.sum(true_nonzero) + 1e-8)
        )
        f1 = float(2 * precision * recall / (precision + recall + 1e-8))
    else:
        precision = recall = f1 = float("nan")

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "sign_accuracy_nonzero": sign_match,
        "n_nonzero": int(n_nonzero),
        "n_total": len(y_true),
        "nonzero_detection": {
            "threshold": 0.05,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train a reward prediction head on frozen JEPA latents"
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
        help="Path to crafter trajectory data directory (episode_*.npz)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results/reward_head",
        help="Directory to save reward head and results (default: eval_results/reward_head)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension of reward head MLP (default: 128)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for training (default: 256)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation (default: 0.1)",
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

    # --- Output directory ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================================
    # 1. Load episodes
    # =====================================================================
    print("\n--- Loading episodes ---")
    episodes = load_episodes(args.data_dir)

    # Print reward statistics
    all_rewards = np.concatenate([ep["rewards"] for ep in episodes])
    print(f"Reward stats: min={all_rewards.min():.3f}, max={all_rewards.max():.3f}, "
          f"mean={all_rewards.mean():.4f}, std={all_rewards.std():.4f}")
    print(f"Non-zero rewards: {np.count_nonzero(all_rewards)}/{len(all_rewards)} "
          f"({100*np.count_nonzero(all_rewards)/len(all_rewards):.1f}%)")
    unique_vals = np.unique(all_rewards)
    print(f"Unique reward values: {unique_vals}")

    # =====================================================================
    # 2. Build and load JEPA, freeze encoder
    # =====================================================================
    print("\n--- Building JEPA and loading checkpoint ---")
    model_cfg = dict(DEFAULT_MODEL_CFG)
    jepa = load_trained_jepa(args.checkpoint_path, model_cfg, device)
    jepa.eval()
    for param in jepa.parameters():
        param.requires_grad = False
    print("JEPA encoder frozen.")

    # =====================================================================
    # 3. Compute normalizer and encode all episodes
    # =====================================================================
    print("\n--- Computing normalizer ---")
    normalizer = compute_normalizer(episodes)
    print(f"  mean={normalizer.state_mean.tolist()}, std={normalizer.state_std.tolist()}")

    print("\n--- Encoding episodes through frozen JEPA ---")
    t0 = time()
    features, rewards = encode_episodes(jepa, episodes, normalizer, device)
    print(f"Encoded {features.shape[0]} frames in {time() - t0:.1f}s -> features {features.shape}")

    # Free GPU memory from JEPA
    del jepa
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # =====================================================================
    # 4. Train/val split
    # =====================================================================
    n_total = features.shape[0]
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val

    # Deterministic shuffle
    rng = torch.Generator().manual_seed(42)
    perm = torch.randperm(n_total, generator=rng)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_features = features[train_idx]
    train_rewards = rewards[train_idx]
    val_features = features[val_idx]
    val_rewards = rewards[val_idx]

    print(f"\nSplit: {n_train} train / {n_val} val")

    # =====================================================================
    # 5. Train reward head
    # =====================================================================
    print("\n--- Training reward head ---")
    t0 = time()
    head, history = train_reward_head(
        train_features,
        train_rewards,
        val_features,
        val_rewards,
        input_dim=model_cfg["mlp_output_dim"],
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )
    train_time = time() - t0
    print(f"Training completed in {train_time:.1f}s")

    # =====================================================================
    # 6. Evaluate
    # =====================================================================
    print("\n--- Evaluation ---")
    train_metrics = evaluate_reward_head(head, train_features, train_rewards, device)
    val_metrics = evaluate_reward_head(head, val_features, val_rewards, device)

    print(f"\n{'=' * 60}")
    print(f"  REWARD HEAD RESULTS")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<30} {'Train':>12} {'Val':>12}")
    print(f"  {'-' * 30} {'-' * 12} {'-' * 12}")
    print(f"  {'MSE':<30} {train_metrics['mse']:>12.6f} {val_metrics['mse']:>12.6f}")
    print(f"  {'MAE':<30} {train_metrics['mae']:>12.6f} {val_metrics['mae']:>12.6f}")
    print(f"  {'R-squared':<30} {train_metrics['r2']:>12.4f} {val_metrics['r2']:>12.4f}")
    print(f"  {'Sign accuracy (non-zero)':<30} {train_metrics['sign_accuracy_nonzero']:>12.4f} {val_metrics['sign_accuracy_nonzero']:>12.4f}")
    print(f"  {'Non-zero detection F1':<30} {train_metrics['nonzero_detection']['f1']:>12.4f} {val_metrics['nonzero_detection']['f1']:>12.4f}")
    print(f"  {'Non-zero detection precision':<30} {train_metrics['nonzero_detection']['precision']:>12.4f} {val_metrics['nonzero_detection']['precision']:>12.4f}")
    print(f"  {'Non-zero detection recall':<30} {train_metrics['nonzero_detection']['recall']:>12.4f} {val_metrics['nonzero_detection']['recall']:>12.4f}")
    print(f"{'=' * 60}")

    # =====================================================================
    # 7. Save reward head and results
    # =====================================================================
    # Save reward head weights
    head_path = output_dir / "reward_head.pth"
    torch.save(
        {
            "model_state_dict": head.state_dict(),
            "input_dim": model_cfg["mlp_output_dim"],
            "hidden_dim": args.hidden_dim,
            "architecture": "RewardHead",
        },
        head_path,
    )
    print(f"\nReward head saved to {head_path}")

    # Save normalizer stats (needed to encode new observations at inference)
    normalizer_path = output_dir / "normalizer.pth"
    torch.save(
        {
            "mean": normalizer.state_mean,
            "std": normalizer.state_std,
        },
        normalizer_path,
    )
    print(f"Normalizer saved to {normalizer_path}")

    # Save results JSON
    results = {
        "checkpoint_path": str(args.checkpoint_path),
        "data_dir": str(args.data_dir),
        "epochs": args.epochs,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "batch_size": args.batch_size,
        "train_samples": n_train,
        "val_samples": n_val,
        "feature_dim": int(features.shape[1]),
        "training_time_s": round(train_time, 1),
        "reward_stats": {
            "min": float(all_rewards.min()),
            "max": float(all_rewards.max()),
            "mean": float(all_rewards.mean()),
            "std": float(all_rewards.std()),
            "nonzero_frac": float(np.count_nonzero(all_rewards) / len(all_rewards)),
            "unique_values": [float(v) for v in unique_vals],
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "training_history": history,
    }

    results_path = output_dir / "reward_head_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    print(f"\nDone. To use the reward head in planning:")
    print(f"  head_ckpt = torch.load('{head_path}')")
    print(f"  reward_head = RewardHead(input_dim={model_cfg['mlp_output_dim']}, hidden_dim={args.hidden_dim})")
    print(f"  reward_head.load_state_dict(head_ckpt['model_state_dict'])")


if __name__ == "__main__":
    main()
