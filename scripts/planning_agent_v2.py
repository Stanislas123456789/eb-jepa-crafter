#!/usr/bin/env python3
"""
CEM-based planning agent for Crafter using a trained JEPA world model (v2).

Major upgrade over planning_agent.py (random shooting):
  - Cross-Entropy Method (CEM) adapted for discrete actions: iteratively
    refines a categorical distribution over action sequences by keeping elite
    trajectories.
  - Multiple reward-aligned objectives: survival, resources, reward, exploration,
    and composite (weighted combination).
  - Probe-based planning: a fast linear probe (512 -> 16) predicts game-state
    labels (health, food, wood, ...) from latent states. Objectives optimize
    those predictions directly.
  - Longer default horizon (20 vs 8) and more samples (300 with 5 CEM iters).

Usage:
    python scripts/planning_agent_v2.py \
        --checkpoint_path checkpoints/latest.pth.tar \
        --data_dir data/crafter_trajectories \
        --objective survival \
        --num_episodes 50
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so imports work when running as a
# standalone script (python scripts/planning_agent_v2.py).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eb_jepa.action_encoders import ActionEmbeddingEncoder
from eb_jepa.architectures import (
    ImpalaEncoder,
    InverseDynamicsModel,
    RNNPredictor,
)
from eb_jepa.jepa import JEPA
from eb_jepa.losses import SquareLossSeq, VC_IDM_Sim_Regularizer
from eb_jepa.datasets.crafter.normalizer import CrafterNormalizer


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
    # Regularizer params (needed to construct JEPA, irrelevant for planning)
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

# Hardcoded normalizer stats computed from the training data
DEFAULT_NORM_MEAN = [0.117, 0.349, 0.117]
DEFAULT_NORM_STD = [0.147, 0.260, 0.163]

# The 22 standard Crafter achievements
CRAFTER_ACHIEVEMENTS = [
    "collect_coal",
    "collect_diamond",
    "collect_drink",
    "collect_iron",
    "collect_sapling",
    "collect_stone",
    "collect_wood",
    "defeat_skeleton",
    "defeat_zombie",
    "eat_cow",
    "eat_plant",
    "make_iron_pickaxe",
    "make_iron_sword",
    "make_stone_pickaxe",
    "make_stone_sword",
    "make_wood_pickaxe",
    "make_wood_sword",
    "place_furnace",
    "place_plant",
    "place_stone",
    "place_table",
    "wake_up",
]

# Probe label names (K=16), matching eval_probe.py
PROBE_LABEL_NAMES = [
    "health", "food", "drink", "energy",       # vitals 0-3
    "sapling", "wood", "stone", "coal",         # resources 4-7
    "iron", "diamond",                          # resources 8-9
    "wood_pickaxe", "stone_pickaxe",            # tools 10-11
    "iron_pickaxe", "wood_sword",               # tools 12-13
    "stone_sword", "iron_sword",                # tools 14-15
]


# ---------------------------------------------------------------------------
# Model construction (identical to planning_agent.py)
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

    # Compute spatial dims from a test input
    with torch.no_grad():
        test_input = torch.zeros(1, dobs, 1, img_size, img_size)
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
        from eb_jepa.architectures import Projector
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


def load_checkpoint(jepa: JEPA, path: str, device: torch.device):
    """Load a training checkpoint into the JEPA model."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    # Strip torch.compile prefixes if present
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    jepa.load_state_dict(state_dict, strict=False)
    epoch = checkpoint.get("epoch", "?")
    step = checkpoint.get("step", "?")
    print(f"Loaded checkpoint from {path} (epoch={epoch}, step={step})")
    return checkpoint


# ---------------------------------------------------------------------------
# Reward Head (mirrors train_reward_head.py)
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
        # latent: [N, 512]
        if latent.dim() == 5:
            B, D, T, H, W = latent.shape
            latent = latent.squeeze(-1).squeeze(-1).permute(0, 2, 1).reshape(-1, D)
        return self.net(latent).squeeze(-1)


# ---------------------------------------------------------------------------
# Objective Functions
# ---------------------------------------------------------------------------

class SurvivalObjective:
    """Maximize predicted health maintenance via the probe."""

    def __init__(self, probe):
        self.probe = probe  # nn.Linear(512, 16)

    @torch.no_grad()
    def __call__(self, predicted_states, initial_state):
        # predicted_states: [N, 512, T, 1, 1]
        # Use probe to predict health at final state
        final = predicted_states[:, :, -1, 0, 0]  # [N, 512]
        pred_labels = self.probe(final)  # [N, 16]
        health = pred_labels[:, 0]  # health is index 0
        return health


class ResourceObjective:
    """Maximize predicted resource collection via the probe."""

    def __init__(self, probe):
        self.probe = probe  # nn.Linear(512, 16)

    @torch.no_grad()
    def __call__(self, predicted_states, initial_state):
        # predicted_states: [N, 512, T, 1, 1]
        final = predicted_states[:, :, -1, 0, 0]  # [N, 512]
        pred = self.probe(final)  # [N, 16]
        # Sum key resources: wood(5), stone(6), coal(7), iron(8)
        resources = pred[:, 5] + pred[:, 6] + pred[:, 7] + pred[:, 8]
        return resources


class RewardObjective:
    """Maximize predicted cumulative reward using the reward head."""

    def __init__(self, reward_head):
        self.reward_head = reward_head

    @torch.no_grad()
    def __call__(self, predicted_states, initial_state):
        # predicted_states: [N, 512, T, 1, 1]
        B, D, T, H, W = predicted_states.shape
        all_states = predicted_states.squeeze(-1).squeeze(-1).permute(0, 2, 1).reshape(-1, D)
        rewards = self.reward_head(all_states).reshape(B, T)
        return rewards.sum(dim=1)  # cumulative reward per trajectory


class ExplorationObjective:
    """Maximize latent distance from initial state (same as v1 baseline)."""

    @torch.no_grad()
    def __call__(self, predicted_states, initial_state):
        # predicted_states: [N, 512, T, 1, 1]
        initial = predicted_states[:, :, 0:1]  # [N, 512, 1, 1, 1]
        final = predicted_states[:, :, -1:]    # [N, 512, 1, 1, 1]
        scores = ((final - initial) ** 2).mean(dim=(1, 2, 3, 4))
        return scores


class CompositeObjective:
    """Weighted combination of multiple objectives."""

    def __init__(self, objectives, weights):
        self.objectives = objectives
        self.weights = weights

    @torch.no_grad()
    def __call__(self, predicted_states, initial_state):
        score = torch.zeros(predicted_states.shape[0], device=predicted_states.device)
        for obj, w in zip(self.objectives, self.weights):
            score = score + w * obj(predicted_states, initial_state)
        return score


# ---------------------------------------------------------------------------
# Discrete CEM Planner
# ---------------------------------------------------------------------------

class DiscreteCEMPlanner:
    """Cross-Entropy Method planner adapted for discrete actions.

    Instead of random shooting, iteratively refines a categorical distribution
    over actions at each timestep by keeping the elite (highest-scoring)
    trajectories and re-fitting the distribution.
    """

    def __init__(
        self,
        jepa: JEPA,
        normalizer: CrafterNormalizer,
        num_actions: int = 17,
        horizon: int = 20,
        num_samples: int = 300,
        num_elites: int = 30,
        num_iterations: int = 5,
        device: torch.device = torch.device("cpu"),
        batch_limit: int = 0,
    ):
        self.jepa = jepa
        self.normalizer = normalizer
        self.num_actions = num_actions
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_iterations = num_iterations
        self.device = device
        self.batch_limit = batch_limit if batch_limit > 0 else num_samples

    @torch.no_grad()
    def plan(self, obs_frame: np.ndarray, objective_fn) -> int:
        """Select an action for a single Crafter observation using CEM.

        Args:
            obs_frame: numpy array [64, 64, 3] uint8 (raw Crafter observation).
            objective_fn: callable(predicted_states, initial_state) -> [N] scores.

        Returns:
            int action in [0, num_actions).
        """
        # 1. Normalize and encode current observation
        obs = self._encode_obs(obs_frame)  # [1, C, 1, H, W]

        # 2. Initialize action distribution: uniform categorical at each timestep
        # action_probs[t] = [p(a=0), p(a=1), ..., p(a=16)] for timestep t
        action_probs = torch.ones(
            self.horizon, self.num_actions, device=self.device
        ) / self.num_actions

        best_actions = None
        best_scores = None

        for iteration in range(self.num_iterations):
            # Sample action sequences from current distribution
            actions = torch.stack([
                torch.multinomial(action_probs[t], self.num_samples, replacement=True)
                for t in range(self.horizon)
            ], dim=0)  # [H, N]
            actions = actions.T  # [N, H]

            # Unroll world model in (possibly chunked) batches
            all_predicted = []
            for start in range(0, self.num_samples, self.batch_limit):
                end = min(start + self.batch_limit, self.num_samples)
                batch_size = end - start

                obs_batch = obs.expand(batch_size, -1, -1, -1, -1)
                actions_batch = actions[start:end]

                predicted, _ = self.jepa.unroll(
                    obs_batch,
                    actions_batch,
                    nsteps=self.horizon,
                    unroll_mode="autoregressive",
                    compute_loss=False,
                )
                # predicted: [B, 512, H+1, 1, 1]
                all_predicted.append(predicted)

            predicted = torch.cat(all_predicted, dim=0)  # [N, 512, H+1, 1, 1]

            # Score each sequence with the objective function
            scores = objective_fn(predicted, obs)  # [N]

            # Select elite sequences
            elite_indices = scores.topk(self.num_elites).indices
            elite_actions = actions[elite_indices]  # [num_elites, H]

            # Update action distribution from elite sequences
            for t in range(self.horizon):
                counts = torch.zeros(self.num_actions, device=self.device)
                for a in elite_actions[:, t]:
                    counts[a.item()] += 1
                # Smooth with uniform prior to avoid collapsing to single action
                action_probs[t] = (counts + 1.0) / (self.num_elites + self.num_actions)

            # Track the best across iterations
            if best_scores is None or scores.max() > best_scores.max():
                best_idx = scores.argmax()
                best_actions = actions[best_idx].clone()

        # Return first action of the best sequence across all iterations
        return best_actions[0].item()

    def _encode_obs(self, obs_frame: np.ndarray) -> torch.Tensor:
        """Normalize and encode a raw observation frame.

        Args:
            obs_frame: [64, 64, 3] uint8 numpy array.

        Returns:
            [1, C, 1, H, W] tensor on self.device.
        """
        obs = torch.from_numpy(obs_frame).float() / 255.0  # [H, W, C]
        obs = obs.permute(2, 0, 1)  # [C, H, W]
        obs = self.normalizer.normalize_state(obs)  # [C, H, W]
        obs = obs.unsqueeze(0).unsqueeze(2).to(self.device)  # [1, C, 1, H, W]
        return obs


# ---------------------------------------------------------------------------
# Quick Probe Training (for probe-based objectives)
# ---------------------------------------------------------------------------

def load_episodes_for_probe(data_dir: str, max_episodes: int = 0):
    """Load episode .npz files for quick probe training."""
    pattern = os.path.join(data_dir, "episode_*.npz")
    paths = sorted(glob.glob(pattern))
    if len(paths) == 0:
        raise FileNotFoundError(f"No episode_*.npz files found in {data_dir}")
    if max_episodes > 0:
        paths = paths[:max_episodes]

    episodes = []
    total_frames = 0
    for path in paths:
        data = np.load(path)
        ep = {"observations": data["observations"]}  # [T, 64, 64, 3] uint8
        # Probe labels if available
        if "probe_labels" in data:
            ep["probe_labels"] = data["probe_labels"]  # [T, 16]
        elif "inventory" in data:
            ep["inventory"] = data["inventory"]
        episodes.append(ep)
        total_frames += ep["observations"].shape[0]

    print(f"Loaded {len(episodes)} episodes ({total_frames} total frames) from {data_dir}")
    return episodes


def compute_normalizer_from_episodes(episodes, num_samples=5000):
    """Compute per-channel normalization stats from raw episodes."""
    rng = np.random.RandomState(42)
    frames = []
    for _ in range(min(num_samples, sum(ep["observations"].shape[0] for ep in episodes))):
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
def encode_frames_for_probe(jepa, episodes, normalizer, device, max_frames=5000):
    """Encode a subset of frames through the frozen JEPA encoder for probe training.

    Returns:
        features: [N, 512] tensor
        labels:   [N, 16] tensor (or None if no probe_labels available)
    """
    jepa.eval()
    rng = np.random.RandomState(42)

    all_features = []
    all_labels = []
    has_labels = "probe_labels" in episodes[0]

    # Sample random frames
    num_frames = 0
    batch_frames = []
    batch_labels = []

    while num_frames < max_frames:
        ep_idx = rng.randint(0, len(episodes))
        ep = episodes[ep_idx]
        T = ep["observations"].shape[0]
        t = rng.randint(0, T)

        frame = ep["observations"][t].astype(np.float32) / 255.0  # [H, W, C]
        frame_t = torch.from_numpy(frame).permute(2, 0, 1)  # [C, H, W]
        frame_t = normalizer.normalize_state(frame_t)
        batch_frames.append(frame_t)

        if has_labels:
            label = ep["probe_labels"][t]  # [16]
            batch_labels.append(torch.from_numpy(label.copy()).float())

        num_frames += 1

        # Encode in batches of 256
        if len(batch_frames) == 256 or num_frames >= max_frames:
            batch = torch.stack(batch_frames, dim=0)  # [B, C, H, W]
            batch = batch.unsqueeze(2).to(device)  # [B, C, 1, H, W]
            enc = jepa.encode(batch)  # [B, 512, 1, 1, 1]
            enc_flat = enc.squeeze(-1).squeeze(-1).squeeze(-1)  # [B, 512]
            all_features.append(enc_flat.cpu())
            if has_labels:
                all_labels.append(torch.stack(batch_labels, dim=0))
            batch_frames = []
            batch_labels = []

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0) if has_labels else None
    return features, labels


def train_linear_probe(features, labels, epochs=5, lr=1e-3, device=torch.device("cpu")):
    """Train a fast linear probe (512 -> 16) on encoded features.

    Args:
        features: [N, 512] tensor
        labels: [N, 16] tensor
        epochs: number of training epochs (default: 5 for speed)
        lr: learning rate
        device: compute device

    Returns:
        Trained nn.Linear(512, 16) probe
    """
    feat_dim = features.shape[1]
    num_targets = labels.shape[1]
    probe = nn.Linear(feat_dim, num_targets).to(device)
    optimizer = AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    n = features.shape[0]
    batch_size = 256
    n_batches = max(1, n // batch_size)

    # Train/val split: 90/10
    n_train = int(0.9 * n)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_feats = features[train_idx]
    train_labels = labels[train_idx]
    val_feats = features[val_idx]
    val_labels = labels[val_idx]

    for epoch in range(1, epochs + 1):
        probe.train()
        perm_train = torch.randperm(n_train)
        epoch_loss = 0.0

        for i in range(n_batches):
            idx = perm_train[i * batch_size : (i + 1) * batch_size]
            x = train_feats[idx].to(device)
            y = train_labels[idx].to(device)

            pred = probe(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches

        # Quick validation
        probe.eval()
        with torch.no_grad():
            val_pred = probe(val_feats.to(device))
            val_loss = criterion(val_pred, val_labels.to(device)).item()

        print(f"  Probe epoch {epoch}/{epochs} | Train MSE: {avg_loss:.6f} | Val MSE: {val_loss:.6f}")

    probe.eval()
    return probe


# ---------------------------------------------------------------------------
# Crafter score computation
# ---------------------------------------------------------------------------

def compute_crafter_score(episode_achievements: dict) -> tuple:
    """Compute the Crafter score (geometric mean of per-achievement success rates).

    Args:
        episode_achievements: dict mapping achievement name to a list of 0/1
            values (one per episode).

    Returns:
        (crafter_score, success_rates_dict)
    """
    success_rates = {}
    for k in sorted(episode_achievements.keys()):
        vals = episode_achievements[k]
        success_rates[k] = float(np.mean(vals)) if len(vals) > 0 else 0.0

    rates = np.array(list(success_rates.values()))
    # Geometric mean with epsilon to avoid log(0)
    crafter_score = float(np.exp(np.mean(np.log(rates + 1e-6))))

    return crafter_score, success_rates


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_results_table(success_rates: dict, crafter_score: float, num_episodes: int):
    """Print a formatted table of achievement success rates."""
    print(f"\n{'=' * 55}")
    print(f"  CRAFTER CEM PLANNING RESULTS  ({num_episodes} episodes)")
    print(f"{'=' * 55}")
    print(f"  {'Achievement':<30} {'Success Rate':>12}")
    print(f"  {'-' * 30} {'-' * 12}")

    for name in sorted(success_rates.keys()):
        rate = success_rates[name]
        bar = "#" * int(rate * 20)
        print(f"  {name:<30} {rate:>10.1%}  {bar}")

    print(f"  {'-' * 30} {'-' * 12}")
    print(f"  {'CRAFTER SCORE':<30} {crafter_score:>12.4f}")
    print(f"{'=' * 55}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CEM-based planning agent for Crafter using a trained JEPA world model (v2)."
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
        default=None,
        help="Path to crafter trajectory data directory (for probe training). "
             "Required for survival/resources/composite objectives.",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="survival",
        choices=["survival", "resources", "reward", "exploration", "composite"],
        help="Planning objective (default: survival)",
    )
    parser.add_argument(
        "--reward_head_path",
        type=str,
        default=None,
        help="Path to trained reward head .pth (required for 'reward' objective)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=20,
        help="CEM prediction horizon (default: 20)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=300,
        help="Number of action sequences sampled per CEM iteration (default: 300)",
    )
    parser.add_argument(
        "--num_elites",
        type=int,
        default=30,
        help="Number of elite sequences kept per CEM iteration (default: 30)",
    )
    parser.add_argument(
        "--cem_iterations",
        type=int,
        default=5,
        help="Number of CEM refinement iterations (default: 5)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=50,
        help="Number of Crafter episodes to run (default: 50)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=300,
        help="Maximum steps per episode (default: 300)",
    )
    parser.add_argument(
        "--batch_limit",
        type=int,
        default=0,
        help="Max batch size for parallel unrolling (0 = all at once). "
             "Reduce if running out of memory.",
    )
    parser.add_argument(
        "--probe_frames",
        type=int,
        default=5000,
        help="Number of frames to encode for quick probe training (default: 5000)",
    )
    parser.add_argument(
        "--probe_epochs",
        type=int,
        default=5,
        help="Number of epochs for quick probe training (default: 5)",
    )
    parser.add_argument(
        "--composite_weights",
        type=float,
        nargs=3,
        default=[1.0, 0.5, 0.3],
        help="Weights for composite objective [survival, resources, exploration] "
             "(default: 1.0 0.5 0.3)",
    )
    parser.add_argument(
        "--norm_mean",
        type=float,
        nargs=3,
        default=None,
        help="Per-channel normalizer mean [R, G, B] (default: compute from data or use hardcoded)",
    )
    parser.add_argument(
        "--norm_std",
        type=float,
        nargs=3,
        default=None,
        help="Per-channel normalizer std [R, G, B] (default: compute from data or use hardcoded)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="planning_v2_results.json",
        help="Path to save results JSON (default: planning_v2_results.json)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect cuda/mps/cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # --- Validate arguments ---
    needs_probe = args.objective in ("survival", "resources", "composite")
    if needs_probe and args.data_dir is None:
        parser.error(
            f"--data_dir is required for objective '{args.objective}' (needed for probe training)"
        )
    if args.objective == "reward" and args.reward_head_path is None:
        parser.error("--reward_head_path is required for objective 'reward'")

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

    # --- Seed ---
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    # =====================================================================
    # 1. Build and load JEPA
    # =====================================================================
    print("\nBuilding JEPA model...")
    model_cfg = dict(DEFAULT_MODEL_CFG)
    jepa = build_jepa(model_cfg, device)

    print("Loading checkpoint...")
    load_checkpoint(jepa, args.checkpoint_path, device)
    jepa.eval()

    # =====================================================================
    # 2. Set up normalizer
    # =====================================================================
    episodes = None
    if args.data_dir is not None:
        episodes = load_episodes_for_probe(args.data_dir)

    if args.norm_mean is not None and args.norm_std is not None:
        normalizer = CrafterNormalizer(
            mean=torch.tensor(args.norm_mean, dtype=torch.float32),
            std=torch.tensor(args.norm_std, dtype=torch.float32),
        )
        print(f"Normalizer (from args): mean={args.norm_mean}, std={args.norm_std}")
    elif episodes is not None:
        normalizer = compute_normalizer_from_episodes(episodes)
        print(f"Normalizer (computed): mean={normalizer.state_mean.tolist()}, "
              f"std={normalizer.state_std.tolist()}")
    else:
        normalizer = CrafterNormalizer(
            mean=torch.tensor(DEFAULT_NORM_MEAN, dtype=torch.float32),
            std=torch.tensor(DEFAULT_NORM_STD, dtype=torch.float32),
        )
        print(f"Normalizer (hardcoded): mean={DEFAULT_NORM_MEAN}, std={DEFAULT_NORM_STD}")

    # =====================================================================
    # 3. Train probe if needed
    # =====================================================================
    probe = None
    if needs_probe:
        print(f"\n--- Training linear probe ({args.probe_frames} frames, {args.probe_epochs} epochs) ---")
        t0 = time.time()
        features, labels = encode_frames_for_probe(
            jepa, episodes, normalizer, device, max_frames=args.probe_frames
        )
        print(f"  Encoded {features.shape[0]} frames in {time.time() - t0:.1f}s")

        if labels is None:
            print("ERROR: probe_labels not found in episode data. "
                  "Cannot train probe for this objective.")
            print("Use --objective exploration or --objective reward instead.")
            sys.exit(1)

        probe = train_linear_probe(
            features, labels,
            epochs=args.probe_epochs,
            lr=1e-3,
            device=device,
        )
        print(f"Probe training completed in {time.time() - t0:.1f}s total")

    # =====================================================================
    # 4. Load reward head if needed
    # =====================================================================
    reward_head = None
    if args.objective == "reward":
        print(f"\n--- Loading reward head from {args.reward_head_path} ---")
        ckpt = torch.load(args.reward_head_path, map_location=device, weights_only=False)
        input_dim = ckpt.get("input_dim", 512)
        hidden_dim = ckpt.get("hidden_dim", 128)
        reward_head = RewardHead(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
        reward_head.load_state_dict(ckpt["model_state_dict"])
        reward_head.eval()
        print("Reward head loaded.")

    # =====================================================================
    # 5. Create objective function
    # =====================================================================
    print(f"\n--- Setting up '{args.objective}' objective ---")

    if args.objective == "survival":
        objective_fn = SurvivalObjective(probe)
    elif args.objective == "resources":
        objective_fn = ResourceObjective(probe)
    elif args.objective == "reward":
        objective_fn = RewardObjective(reward_head)
    elif args.objective == "exploration":
        objective_fn = ExplorationObjective()
    elif args.objective == "composite":
        w_surv, w_res, w_expl = args.composite_weights
        objectives = []
        weights = []
        if w_surv > 0:
            objectives.append(SurvivalObjective(probe))
            weights.append(w_surv)
        if w_res > 0:
            objectives.append(ResourceObjective(probe))
            weights.append(w_res)
        if w_expl > 0:
            objectives.append(ExplorationObjective())
            weights.append(w_expl)
        objective_fn = CompositeObjective(objectives, weights)
        print(f"  Composite weights: survival={w_surv}, resources={w_res}, exploration={w_expl}")
    else:
        raise ValueError(f"Unknown objective: {args.objective}")

    print(f"Objective function: {objective_fn.__class__.__name__}")

    # =====================================================================
    # 6. Create CEM planner
    # =====================================================================
    planner = DiscreteCEMPlanner(
        jepa=jepa,
        normalizer=normalizer,
        num_actions=model_cfg["num_actions"],
        horizon=args.horizon,
        num_samples=args.num_samples,
        num_elites=args.num_elites,
        num_iterations=args.cem_iterations,
        device=device,
        batch_limit=args.batch_limit,
    )
    print(
        f"CEM Planner: N={args.num_samples}, E={args.num_elites}, "
        f"H={args.horizon}, iters={args.cem_iterations}, "
        f"batch_limit={args.batch_limit or 'all'}"
    )

    # =====================================================================
    # 7. Run episodes
    # =====================================================================
    import crafter

    env = crafter.Env()
    print(f"\nCrafter environment created (action_space={env.action_space})")
    print(f"\nRunning {args.num_episodes} episodes (max {args.max_steps} steps each)...\n")

    # Track per-episode achievements
    all_episode_achievements = []
    aggregated_achievements = {a: [] for a in CRAFTER_ACHIEVEMENTS}
    all_episode_rewards = []
    all_episode_lengths = []

    total_start = time.time()

    for ep in range(args.num_episodes):
        ep_start = time.time()
        obs = env.reset()
        episode_reward = 0.0
        episode_unlocked = set()

        for step in range(args.max_steps):
            action = planner.plan(obs, objective_fn)
            obs, reward, done, info = env.step(action)
            episode_reward += reward

            # Track newly unlocked achievements this step
            if "achievements" in info:
                for ach_name, ach_val in info["achievements"].items():
                    if ach_val > 0:
                        episode_unlocked.add(ach_name)

            if done:
                break

        ep_duration = time.time() - ep_start
        ep_steps = step + 1

        # Record this episode
        ep_achievement_dict = {}
        for a in CRAFTER_ACHIEVEMENTS:
            unlocked = 1 if a in episode_unlocked else 0
            ep_achievement_dict[a] = unlocked
            aggregated_achievements[a].append(unlocked)
        all_episode_achievements.append(ep_achievement_dict)
        all_episode_rewards.append(float(episode_reward))
        all_episode_lengths.append(ep_steps)

        num_unlocked = len(episode_unlocked)

        if (ep + 1) % 5 == 0 or ep == 0:
            elapsed = time.time() - total_start
            eps_per_sec = (ep + 1) / elapsed
            print(
                f"  Episode {ep + 1:>3d}/{args.num_episodes} | "
                f"steps={ep_steps:>3d} | reward={episode_reward:>6.1f} | "
                f"achievements={num_unlocked:>2d} | "
                f"time={ep_duration:.1f}s | "
                f"total={elapsed:.0f}s ({eps_per_sec:.2f} ep/s)"
            )

    total_time = time.time() - total_start

    # =====================================================================
    # 8. Compute Crafter score and print results
    # =====================================================================
    crafter_score, success_rates = compute_crafter_score(aggregated_achievements)

    print_results_table(success_rates, crafter_score, args.num_episodes)

    print(f"Total time: {total_time:.1f}s ({total_time / args.num_episodes:.1f}s/episode)")
    print(f"Mean reward per episode: {np.mean(all_episode_rewards):.2f}")
    print(f"Mean episode length: {np.mean(all_episode_lengths):.1f}")

    # =====================================================================
    # 9. Save results
    # =====================================================================
    results = {
        "config": {
            "checkpoint_path": str(args.checkpoint_path),
            "data_dir": str(args.data_dir) if args.data_dir else None,
            "objective": args.objective,
            "reward_head_path": str(args.reward_head_path) if args.reward_head_path else None,
            "num_episodes": args.num_episodes,
            "max_steps": args.max_steps,
            "horizon": args.horizon,
            "num_samples": args.num_samples,
            "num_elites": args.num_elites,
            "cem_iterations": args.cem_iterations,
            "batch_limit": args.batch_limit,
            "probe_frames": args.probe_frames,
            "probe_epochs": args.probe_epochs,
            "composite_weights": args.composite_weights if args.objective == "composite" else None,
            "device": str(device),
            "seed": args.seed,
            "planner": "DiscreteCEM",
        },
        "crafter_score": crafter_score,
        "success_rates": success_rates,
        "per_episode": {
            "achievements": all_episode_achievements,
            "rewards": all_episode_rewards,
            "lengths": all_episode_lengths,
        },
        "summary": {
            "mean_reward": float(np.mean(all_episode_rewards)),
            "std_reward": float(np.std(all_episode_rewards)),
            "mean_length": float(np.mean(all_episode_lengths)),
            "total_time_seconds": total_time,
            "seconds_per_episode": total_time / args.num_episodes,
        },
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
