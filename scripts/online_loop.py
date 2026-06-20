#!/usr/bin/env python3
"""
DreamerV2-style online training loop for JEPA world model on Crafter.

Iterates between:
  1. Acting in Crafter with a random-shooting planner driven by the JEPA world model
  2. Retraining the world model on all collected data (offline + newly collected)

Usage:
    python scripts/online_loop.py \
        --checkpoint_path checkpoints/.../latest.pth.tar \
        --data_dir data/crafter_online \
        --num_iterations 5 \
        --episodes_per_iter 100 \
        --train_epochs 3

    # Optionally seed with existing offline data:
    python scripts/online_loop.py \
        --checkpoint_path checkpoints/.../latest.pth.tar \
        --data_dir data/crafter_online \
        --offline_data_dir data/crafter_trajectories \
        --num_iterations 5
"""

import argparse
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so imports work when running as a
# standalone script (python scripts/online_loop.py).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eb_jepa.datasets.crafter.normalizer import CrafterNormalizer
from eb_jepa.datasets.crafter.crafter_dataset import CrafterTrajDataset

# Import building blocks from planning_agent
from scripts.planning_agent import (
    DEFAULT_MODEL_CFG,
    DEFAULT_NORM_MEAN,
    DEFAULT_NORM_STD,
    CRAFTER_ACHIEVEMENTS,
    RandomShootingPlanner,
    build_jepa,
    load_checkpoint as load_jepa_checkpoint,
    compute_crafter_score,
    print_results_table,
)

# Import the collect_crafter_data helpers for probe label extraction
from scripts.collect_crafter_data import (
    INVENTORY_KEYS,
    MAX_STEPS_PER_EPISODE,
    extract_probe_labels,
)


# ---------------------------------------------------------------------------
# Helper: compute normalizer from data directory
# ---------------------------------------------------------------------------

def compute_normalizer_from_data(data_dir: str, num_samples: int = 5000) -> CrafterNormalizer:
    """Compute per-channel normalizer statistics from episode .npz files.

    Falls back to default hardcoded stats if the data directory has no episodes
    (e.g. first iteration before any data has been collected).
    """
    pattern = os.path.join(data_dir, "episode_*.npz")
    episode_paths = sorted(glob.glob(pattern))
    if len(episode_paths) == 0:
        print("  No episodes found in data dir; using default normalizer stats.")
        return CrafterNormalizer(
            mean=torch.tensor(DEFAULT_NORM_MEAN, dtype=torch.float32),
            std=torch.tensor(DEFAULT_NORM_STD, dtype=torch.float32),
        )

    # Sample random frames and compute stats
    rng = np.random.RandomState(42)
    all_pixels = []
    frames_collected = 0

    while frames_collected < num_samples:
        ep_path = rng.choice(episode_paths)
        data = np.load(ep_path)
        obs = data["observations"]  # [T, 64, 64, 3] uint8
        T = obs.shape[0]
        t = rng.randint(0, T)
        frame = obs[t].astype(np.float32) / 255.0  # [H, W, C]
        all_pixels.append(frame)
        frames_collected += 1

    all_pixels = np.stack(all_pixels, axis=0)  # [N, H, W, C]
    mean = all_pixels.mean(axis=(0, 1, 2))  # [C]
    std = all_pixels.std(axis=(0, 1, 2))    # [C]

    normalizer = CrafterNormalizer(
        mean=torch.tensor(mean, dtype=torch.float32),
        std=torch.tensor(std, dtype=torch.float32),
    )
    print(f"  Computed normalizer: mean={mean.tolist()}, std={std.tolist()}")
    return normalizer


# ---------------------------------------------------------------------------
# Helper: run a single episode with the planner, returning data for saving
# ---------------------------------------------------------------------------

def run_episode(
    planner: RandomShootingPlanner,
    env,
    max_steps: int = MAX_STEPS_PER_EPISODE,
) -> dict:
    """Run one Crafter episode using the planner.

    Returns a dict with the same keys as collect_crafter_data.py episodes:
        observations, actions, rewards, probe_labels, player_positions
    Also returns 'achievements' (set of unlocked achievement names).
    """
    obs = env.reset()

    observations = [obs]
    actions = []
    rewards = []
    probe_labels = []
    player_positions = []
    episode_achievements = set()

    for _ in range(max_steps):
        action = planner.plan(obs)
        obs, reward, done, info = env.step(action)

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        probe_labels.append(extract_probe_labels(info))
        player_positions.append(np.array(info["player_pos"], dtype=np.int64))

        # Track achievements
        if "achievements" in info:
            for ach_name, ach_val in info["achievements"].items():
                if ach_val > 0:
                    episode_achievements.add(ach_name)

        if done:
            break

    # Match the format from collect_crafter_data.py: observations[:-1]
    observations = np.stack(observations[:-1], axis=0).astype(np.uint8)    # [T, 64, 64, 3]
    actions = np.array(actions, dtype=np.int64)                            # [T]
    rewards = np.array(rewards, dtype=np.float32)                          # [T]
    probe_labels = np.stack(probe_labels, axis=0).astype(np.float32)       # [T, 16]
    player_positions = np.stack(player_positions, axis=0).astype(np.int64) # [T, 2]

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "probe_labels": probe_labels,
        "player_positions": player_positions,
        "achievements": episode_achievements,
    }


# ---------------------------------------------------------------------------
# Helper: save an episode to disk
# ---------------------------------------------------------------------------

def save_episode(episode_data: dict, data_dir: str, episode_idx: int) -> str:
    """Save an episode as episode_XXXX.npz (same format as collect_crafter_data.py)."""
    fname = os.path.join(data_dir, f"episode_{episode_idx:04d}.npz")
    np.savez_compressed(
        fname,
        observations=episode_data["observations"],
        actions=episode_data["actions"],
        rewards=episode_data["rewards"],
        probe_labels=episode_data["probe_labels"],
        player_positions=episode_data["player_positions"],
    )
    return fname


# ---------------------------------------------------------------------------
# Helper: count existing episodes in a directory
# ---------------------------------------------------------------------------

def count_existing_episodes(data_dir: str) -> int:
    """Count the number of episode_*.npz files in data_dir."""
    pattern = os.path.join(data_dir, "episode_*.npz")
    return len(glob.glob(pattern))


# ---------------------------------------------------------------------------
# Helper: copy offline data to online data directory
# ---------------------------------------------------------------------------

def copy_offline_data(offline_dir: str, online_dir: str) -> int:
    """Copy all episode .npz files from offline_dir to online_dir.

    Returns the number of files copied.
    """
    os.makedirs(online_dir, exist_ok=True)
    pattern = os.path.join(offline_dir, "episode_*.npz")
    src_files = sorted(glob.glob(pattern))
    copied = 0
    for src in src_files:
        dst = os.path.join(online_dir, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied += 1
    return copied


# ---------------------------------------------------------------------------
# Helper: compute Crafter score from episodes
# ---------------------------------------------------------------------------

def compute_score_from_episodes(episodes: list) -> tuple:
    """Compute Crafter score from a list of episode dicts (with 'achievements' key).

    Returns (crafter_score, success_rates_dict).
    """
    aggregated = {a: [] for a in CRAFTER_ACHIEVEMENTS}
    for ep in episodes:
        ep_achievements = ep.get("achievements", set())
        for a in CRAFTER_ACHIEVEMENTS:
            aggregated[a].append(1 if a in ep_achievements else 0)

    return compute_crafter_score(aggregated)


# ---------------------------------------------------------------------------
# Main online loop
# ---------------------------------------------------------------------------

def online_loop(
    checkpoint_path: str,
    data_dir: str,
    num_iterations: int = 5,
    episodes_per_iter: int = 100,
    train_epochs_per_iter: int = 3,
    planner_horizon: int = 10,
    planner_samples: int = 500,
    planner_objective: str = "exploration",
    planner_batch_limit: int = 0,
    max_steps_per_episode: int = MAX_STEPS_PER_EPISODE,
    offline_data_dir: str = None,
    config_path: str = None,
    device: torch.device = None,
    seed: int = 42,
):
    """Run the DreamerV2-style online training loop.

    Args:
        checkpoint_path: Path to the initial offline-trained JEPA checkpoint.
        data_dir: Directory for storing all episode data (combined offline + online).
        num_iterations: Number of collect-then-retrain iterations.
        episodes_per_iter: Episodes to collect per iteration.
        train_epochs_per_iter: Epochs to retrain the model per iteration.
        planner_horizon: Planning horizon for random shooting.
        planner_samples: Number of random action sequences to evaluate per step.
        planner_objective: Scoring objective for the planner.
        planner_batch_limit: Sub-batch size for planner unrolling (0 = all at once).
        max_steps_per_episode: Maximum environment steps per episode.
        offline_data_dir: If set, copy offline data to data_dir before starting.
        config_path: Path to YAML config for training (default: crafter.yaml).
        device: Torch device to use.
        seed: Random seed.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    if config_path is None:
        config_path = str(PROJECT_ROOT / "examples" / "ac_video_jepa" / "cfgs" / "crafter.yaml")

    print("=" * 70)
    print("  JEPA ONLINE TRAINING LOOP (DreamerV2-style)")
    print("=" * 70)
    print(f"  Checkpoint      : {checkpoint_path}")
    print(f"  Data directory  : {data_dir}")
    print(f"  Offline data    : {offline_data_dir or '(none)'}")
    print(f"  Iterations      : {num_iterations}")
    print(f"  Episodes/iter   : {episodes_per_iter}")
    print(f"  Train epochs    : {train_epochs_per_iter}")
    print(f"  Planner horizon : {planner_horizon}")
    print(f"  Planner samples : {planner_samples}")
    print(f"  Planner obj     : {planner_objective}")
    print(f"  Max steps/ep    : {max_steps_per_episode}")
    print(f"  Device          : {device}")
    print(f"  Seed            : {seed}")
    print(f"  Config          : {config_path}")
    print("=" * 70)

    # Seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Step 0: Copy offline data if provided
    if offline_data_dir and os.path.isdir(offline_data_dir):
        print(f"\nCopying offline data from {offline_data_dir} to {data_dir}...")
        num_copied = copy_offline_data(offline_data_dir, data_dir)
        print(f"  Copied {num_copied} episode files.")
    else:
        if offline_data_dir:
            print(f"\nWarning: offline_data_dir '{offline_data_dir}' not found; skipping copy.")

    existing_count = count_existing_episodes(data_dir)
    print(f"Starting with {existing_count} episodes in {data_dir}")

    # Determine the model folder from the checkpoint path
    # The training loop saves checkpoints to the model_folder
    model_folder = str(Path(checkpoint_path).parent)

    # Track results across iterations
    iteration_results = []
    all_scores = []

    total_start_time = time.time()

    for iteration in range(num_iterations):
        iter_start_time = time.time()
        print(f"\n{'=' * 70}")
        print(f"  ONLINE ITERATION {iteration + 1}/{num_iterations}")
        print(f"{'=' * 70}")

        # -----------------------------------------------------------------
        # 1. Build / reload model from latest checkpoint
        # -----------------------------------------------------------------
        print(f"\n  [1/4] Loading JEPA from {checkpoint_path}")
        model_cfg = dict(DEFAULT_MODEL_CFG)
        jepa = build_jepa(model_cfg, device)
        load_jepa_checkpoint(jepa, checkpoint_path, device)
        jepa.eval()

        # -----------------------------------------------------------------
        # 2. Build normalizer and planner
        # -----------------------------------------------------------------
        print("\n  [2/4] Setting up planner...")
        normalizer = compute_normalizer_from_data(data_dir)
        planner = RandomShootingPlanner(
            jepa=jepa,
            normalizer=normalizer,
            num_samples=planner_samples,
            horizon=planner_horizon,
            objective=planner_objective,
            num_actions=model_cfg["num_actions"],
            device=device,
            batch_limit=planner_batch_limit,
        )

        # -----------------------------------------------------------------
        # 3. Collect episodes with planner
        # -----------------------------------------------------------------
        print(f"\n  [3/4] Collecting {episodes_per_iter} episodes...")
        import crafter
        env = crafter.Env()

        current_count = count_existing_episodes(data_dir)
        new_episodes = []
        iter_rewards = []
        iter_lengths = []
        collect_start = time.time()

        for ep_idx in range(episodes_per_iter):
            episode_data = run_episode(planner, env, max_steps=max_steps_per_episode)
            new_episodes.append(episode_data)

            # Save to data directory
            save_idx = current_count + ep_idx
            save_episode(episode_data, data_dir, save_idx)

            ep_reward = float(episode_data["rewards"].sum())
            ep_len = len(episode_data["actions"])
            iter_rewards.append(ep_reward)
            iter_lengths.append(ep_len)

            if (ep_idx + 1) % max(1, episodes_per_iter // 10) == 0 or ep_idx == 0:
                elapsed = time.time() - collect_start
                eps_per_sec = (ep_idx + 1) / elapsed if elapsed > 0 else 0
                print(
                    f"    Episode {ep_idx + 1:>4d}/{episodes_per_iter} | "
                    f"steps={ep_len:>3d} | reward={ep_reward:>6.1f} | "
                    f"speed={eps_per_sec:.2f} ep/s"
                )

        collect_time = time.time() - collect_start

        # Compute Crafter score for this iteration
        crafter_score, success_rates = compute_score_from_episodes(new_episodes)
        all_scores.append(crafter_score)

        print(f"\n    Collection complete: {episodes_per_iter} episodes in {collect_time:.1f}s")
        print(f"    Mean reward: {np.mean(iter_rewards):.2f}, Mean length: {np.mean(iter_lengths):.1f}")
        print(f"    Crafter score: {crafter_score:.4f}")

        # Print achievement rates
        print_results_table(success_rates, crafter_score, episodes_per_iter)

        # Free model memory before retraining
        del jepa, planner
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # -----------------------------------------------------------------
        # 4. Retrain model on combined data (old + new)
        # -----------------------------------------------------------------
        print(f"\n  [4/4] Retraining world model for {train_epochs_per_iter} epochs on combined data...")
        total_episodes = count_existing_episodes(data_dir)
        print(f"    Total episodes in replay buffer: {total_episodes}")

        retrain_start = time.time()

        # Import and call the training loop
        from examples.ac_video_jepa.main import run as train_run

        train_overrides = {
            "data.data_dir": data_dir,
            "optim.epochs": train_epochs_per_iter,
            "meta.load_model": True,
            "meta.model_folder": model_folder,
            "logging.log_wandb": False,
            "logging.tqdm_silent": False,
            "meta.seed": seed + iteration,
            # Disable eval during retraining (speed)
            "meta.enable_plan_eval": False,
        }

        try:
            train_run(
                fname=config_path,
                folder=Path(model_folder),
                **train_overrides,
            )
        except Exception as e:
            print(f"    WARNING: Training raised an exception: {e}")
            print(f"    Continuing to next iteration...")

        retrain_time = time.time() - retrain_start
        print(f"    Retraining complete in {retrain_time:.1f}s")

        # Update checkpoint path for next iteration (training saves to latest.pth.tar)
        checkpoint_path = str(Path(model_folder) / "latest.pth.tar")
        print(f"    Updated checkpoint: {checkpoint_path}")

        # Record iteration results
        iter_time = time.time() - iter_start_time
        iter_result = {
            "iteration": iteration + 1,
            "crafter_score": crafter_score,
            "success_rates": success_rates,
            "mean_reward": float(np.mean(iter_rewards)),
            "std_reward": float(np.std(iter_rewards)),
            "mean_episode_length": float(np.mean(iter_lengths)),
            "episodes_collected": episodes_per_iter,
            "total_episodes_in_buffer": total_episodes,
            "collection_time_seconds": collect_time,
            "retrain_time_seconds": retrain_time,
            "iteration_time_seconds": iter_time,
            "checkpoint_path": checkpoint_path,
        }
        iteration_results.append(iter_result)

        # Save intermediate results after each iteration
        results_path = os.path.join(data_dir, "online_results.json")
        _save_results(results_path, iteration_results, all_scores)

    # ---------------------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------------------
    total_time = time.time() - total_start_time

    print("\n" + "=" * 70)
    print("  ONLINE TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"  Iterations: {num_iterations}")
    print(f"  Total new episodes: {num_iterations * episodes_per_iter}")
    print()
    print(f"  {'Iteration':<12} {'Crafter Score':>15} {'Mean Reward':>13} {'Mean Length':>13}")
    print(f"  {'-' * 12} {'-' * 15} {'-' * 13} {'-' * 13}")
    for r in iteration_results:
        print(
            f"  {r['iteration']:<12d} "
            f"{r['crafter_score']:>15.4f} "
            f"{r['mean_reward']:>13.2f} "
            f"{r['mean_episode_length']:>13.1f}"
        )
    print(f"  {'-' * 12} {'-' * 15} {'-' * 13} {'-' * 13}")

    if len(all_scores) >= 2:
        improvement = all_scores[-1] - all_scores[0]
        print(f"\n  Score change (first -> last): {all_scores[0]:.4f} -> {all_scores[-1]:.4f} ({improvement:+.4f})")
    print(f"\n  Results saved to: {os.path.join(data_dir, 'online_results.json')}")
    print("=" * 70)


def _save_results(path: str, iteration_results: list, all_scores: list):
    """Save online loop results to JSON."""
    results = {
        "iteration_results": iteration_results,
        "scores_per_iteration": all_scores,
        "num_iterations_completed": len(iteration_results),
    }
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DreamerV2-style online training loop for JEPA world model on Crafter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with an offline-trained checkpoint:
    python scripts/online_loop.py \\
        --checkpoint_path checkpoints/.../latest.pth.tar \\
        --data_dir data/crafter_online \\
        --num_iterations 5 \\
        --episodes_per_iter 100 \\
        --train_epochs 3

    # Include existing offline data in the replay buffer:
    python scripts/online_loop.py \\
        --checkpoint_path checkpoints/.../latest.pth.tar \\
        --data_dir data/crafter_online \\
        --offline_data_dir data/crafter_trajectories \\
        --num_iterations 5
        """,
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the offline-trained JEPA checkpoint (.pth.tar).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory for storing all episode data (combined offline + online).",
    )
    parser.add_argument(
        "--offline_data_dir",
        type=str,
        default=None,
        help="If set, copy episodes from this dir to data_dir before starting.",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=5,
        help="Number of collect-then-retrain iterations (default: 5).",
    )
    parser.add_argument(
        "--episodes_per_iter",
        type=int,
        default=100,
        help="Number of episodes to collect per iteration (default: 100).",
    )
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=3,
        help="Number of training epochs per iteration (default: 3).",
    )
    parser.add_argument(
        "--planner_horizon",
        type=int,
        default=10,
        help="Planning horizon for random shooting (default: 10).",
    )
    parser.add_argument(
        "--planner_samples",
        type=int,
        default=500,
        help="Number of random action sequences to sample per step (default: 500).",
    )
    parser.add_argument(
        "--planner_objective",
        type=str,
        default="exploration",
        choices=["exploration", "max_spread", "momentum"],
        help="Scoring objective for the planner (default: exploration).",
    )
    parser.add_argument(
        "--planner_batch_limit",
        type=int,
        default=0,
        help="Sub-batch size for planner unrolling (0 = all at once, default: 0). "
             "Reduce if running out of GPU memory.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=MAX_STEPS_PER_EPISODE,
        help=f"Maximum steps per Crafter episode (default: {MAX_STEPS_PER_EPISODE}).",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to YAML config for training (default: examples/ac_video_jepa/cfgs/crafter.yaml).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect cuda/mps/cpu).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    args = parser.parse_args()

    # Resolve device
    device = None
    if args.device:
        device = torch.device(args.device)

    online_loop(
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        num_iterations=args.num_iterations,
        episodes_per_iter=args.episodes_per_iter,
        train_epochs_per_iter=args.train_epochs,
        planner_horizon=args.planner_horizon,
        planner_samples=args.planner_samples,
        planner_objective=args.planner_objective,
        planner_batch_limit=args.planner_batch_limit,
        max_steps_per_episode=args.max_steps,
        offline_data_dir=args.offline_data_dir,
        config_path=args.config_path,
        device=device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
