#!/usr/bin/env python3
"""
Random-shooting planning agent for Crafter using a trained JEPA world model.

At each environment step the planner:
  1. Encodes the current observation through the JEPA encoder.
  2. Samples N random discrete action sequences of length H.
  3. Unrolls the world model autoregressively for each sequence (in parallel).
  4. Scores every sequence with a chosen objective (exploration or health).
  5. Executes the first action of the best-scoring sequence.

Usage:
    python scripts/planning_agent.py \
        --checkpoint_path checkpoints/latest.pth.tar \
        --num_episodes 50 \
        --num_samples 500 \
        --horizon 10 \
        --objective exploration
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so imports work when running as a
# standalone script (python scripts/planning_agent.py).
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


# ---------------------------------------------------------------------------
# Model construction (mirrors eval_probe.py / eval_rollout.py)
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
# Planner
# ---------------------------------------------------------------------------

class RandomShootingPlanner:
    """Random shooting planner using a JEPA world model.

    Samples many random action sequences, unrolls the world model for each,
    scores them with a chosen objective, and returns the first action of the
    best sequence.
    """

    def __init__(
        self,
        jepa: JEPA,
        normalizer: CrafterNormalizer,
        num_samples: int = 500,
        horizon: int = 10,
        objective: str = "exploration",
        num_actions: int = 17,
        device: torch.device = torch.device("cpu"),
        batch_limit: int = 0,
    ):
        self.jepa = jepa
        self.normalizer = normalizer
        self.num_samples = num_samples
        self.horizon = horizon
        self.objective = objective
        self.num_actions = num_actions
        self.device = device
        # If batch_limit > 0 we chunk the N samples into sub-batches to
        # avoid OOM on GPU. 0 means no chunking (process all at once).
        self.batch_limit = batch_limit if batch_limit > 0 else num_samples

    @torch.no_grad()
    def plan(self, obs_frame: np.ndarray) -> int:
        """Select an action for a single Crafter observation.

        Args:
            obs_frame: numpy array [64, 64, 3] uint8 (raw Crafter observation).

        Returns:
            int action in [0, num_actions).
        """
        # 1. Normalize and encode current observation
        obs = torch.from_numpy(obs_frame).float() / 255.0  # [H, W, C]
        obs = obs.permute(2, 0, 1)  # [C, H, W]
        obs = self.normalizer.normalize_state(obs)  # [C, H, W]
        obs_init = obs.unsqueeze(0).unsqueeze(2).to(self.device)  # [1, C, 1, H, W]

        # 2. Sample random action sequences
        actions = torch.randint(
            0, self.num_actions, (self.num_samples, self.horizon), device=self.device
        )  # [N, H]

        # 3. Unroll world model in (possibly chunked) batches
        all_scores = []
        for start in range(0, self.num_samples, self.batch_limit):
            end = min(start + self.batch_limit, self.num_samples)
            batch_size = end - start

            obs_batch = obs_init.expand(batch_size, -1, -1, -1, -1)  # [B, C, 1, H, W]
            actions_batch = actions[start:end]  # [B, H]

            predicted, _ = self.jepa.unroll(
                obs_batch,
                actions_batch,
                nsteps=self.horizon,
                unroll_mode="autoregressive",
                compute_loss=False,
            )
            # predicted: [B, 512, H+1, 1, 1]  (1 context frame + H predictions)

            # 4. Score sequences
            scores = self._score(predicted)  # [B]
            all_scores.append(scores)

        all_scores = torch.cat(all_scores, dim=0)  # [N]

        # 5. Pick best sequence, return its first action
        best_idx = all_scores.argmax().item()
        return actions[best_idx, 0].item()

    def _score(self, predicted: torch.Tensor) -> torch.Tensor:
        """Score predicted latent trajectories.

        Args:
            predicted: [B, D, T, 1, 1] with T = horizon + 1.

        Returns:
            scores: [B] (higher is better).
        """
        if self.objective == "exploration":
            # Maximise the L2 distance between initial and final latent state.
            # This encourages the agent to seek states that differ from the
            # current one, promoting exploration.
            initial = predicted[:, :, 0:1]  # [B, D, 1, 1, 1]
            final = predicted[:, :, -1:]    # [B, D, 1, 1, 1]
            scores = ((final - initial) ** 2).mean(dim=(1, 2, 3, 4))  # [B]

        elif self.objective == "max_spread":
            # Maximise the average pairwise distance across the full predicted
            # trajectory.  This rewards trajectories that visit diverse states.
            # predicted: [B, D, T, 1, 1]
            flat = predicted.squeeze(-1).squeeze(-1)  # [B, D, T]
            mean_state = flat.mean(dim=2, keepdim=True)  # [B, D, 1]
            scores = ((flat - mean_state) ** 2).mean(dim=(1, 2))  # [B]

        elif self.objective == "momentum":
            # Score by the cumulative change across consecutive steps.
            # Rewards trajectories that keep moving rather than staying put.
            diffs = predicted[:, :, 1:] - predicted[:, :, :-1]  # [B, D, T-1, 1, 1]
            scores = (diffs ** 2).mean(dim=(1, 2, 3, 4))  # [B]

        else:
            raise ValueError(f"Unknown objective: {self.objective}")

        return scores


# ---------------------------------------------------------------------------
# Crafter score computation
# ---------------------------------------------------------------------------

def compute_crafter_score(
    episode_achievements: dict,
) -> tuple:
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
    print(f"  CRAFTER PLANNING RESULTS  ({num_episodes} episodes)")
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
        description="Random-shooting planning agent for Crafter using a trained JEPA world model."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to trained JEPA checkpoint (.pth.tar)",
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
        "--num_samples",
        type=int,
        default=500,
        help="Number of random action sequences to sample per step (default: 500)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Prediction horizon length (default: 10)",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="exploration",
        choices=["exploration", "max_spread", "momentum"],
        help="Scoring objective for action selection (default: exploration)",
    )
    parser.add_argument(
        "--batch_limit",
        type=int,
        default=0,
        help="Max batch size for parallel unrolling (0 = all at once, default: 0). "
             "Reduce if running out of memory.",
    )
    parser.add_argument(
        "--norm_mean",
        type=float,
        nargs=3,
        default=DEFAULT_NORM_MEAN,
        help="Per-channel normalizer mean [R, G, B] (default: precomputed)",
    )
    parser.add_argument(
        "--norm_std",
        type=float,
        nargs=3,
        default=DEFAULT_NORM_STD,
        help="Per-channel normalizer std [R, G, B] (default: precomputed)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="planning_results.json",
        help="Path to save results JSON (default: planning_results.json)",
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
        help="Random seed for reproducibility (default: None)",
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

    # --- Seed ---
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    # --- Build and load JEPA ---
    print("\nBuilding JEPA model...")
    model_cfg = dict(DEFAULT_MODEL_CFG)
    jepa = build_jepa(model_cfg, device)

    print("Loading checkpoint...")
    load_checkpoint(jepa, args.checkpoint_path, device)
    jepa.eval()

    # --- Normalizer ---
    normalizer = CrafterNormalizer(
        mean=torch.tensor(args.norm_mean, dtype=torch.float32),
        std=torch.tensor(args.norm_std, dtype=torch.float32),
    )
    print(f"Normalizer mean={args.norm_mean}, std={args.norm_std}")

    # --- Planner ---
    planner = RandomShootingPlanner(
        jepa=jepa,
        normalizer=normalizer,
        num_samples=args.num_samples,
        horizon=args.horizon,
        objective=args.objective,
        num_actions=model_cfg["num_actions"],
        device=device,
        batch_limit=args.batch_limit,
    )
    print(
        f"Planner: N={args.num_samples}, H={args.horizon}, "
        f"objective={args.objective}, batch_limit={args.batch_limit or 'all'}"
    )

    # --- Crafter environment ---
    import crafter

    env = crafter.Env()
    print(f"Crafter environment created (action_space={env.action_space})")

    # --- Run episodes ---
    print(f"\nRunning {args.num_episodes} episodes (max {args.max_steps} steps each)...\n")

    # Track per-episode achievements
    all_episode_achievements = []  # list of dicts per episode
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
            action = planner.plan(obs)
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

        if (ep + 1) % 10 == 0 or ep == 0:
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

    # --- Compute Crafter score ---
    crafter_score, success_rates = compute_crafter_score(aggregated_achievements)

    # --- Print results ---
    print_results_table(success_rates, crafter_score, args.num_episodes)

    print(f"Total time: {total_time:.1f}s ({total_time / args.num_episodes:.1f}s/episode)")
    print(f"Mean reward per episode: {np.mean(all_episode_rewards):.2f}")
    print(f"Mean episode length: {np.mean(all_episode_lengths):.1f}")

    # --- Save results ---
    results = {
        "config": {
            "checkpoint_path": str(args.checkpoint_path),
            "num_episodes": args.num_episodes,
            "max_steps": args.max_steps,
            "num_samples": args.num_samples,
            "horizon": args.horizon,
            "objective": args.objective,
            "batch_limit": args.batch_limit,
            "device": str(device),
            "seed": args.seed,
            "norm_mean": args.norm_mean,
            "norm_std": args.norm_std,
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
