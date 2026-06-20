#!/usr/bin/env python3
"""
Collect offline trajectories from Crafter using a random policy.

Saves episodes as compressed numpy archives (.npz) with observations,
actions, rewards, probe labels (inventory), and player positions.

Usage:
    python scripts/collect_crafter_data.py --output_dir data/crafter_trajectories
    python scripts/collect_crafter_data.py --output_dir data/crafter_trajectories --num_episodes 100 --seed 0
"""

import argparse
import json
import os
import time

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_ACTIONS = 17  # Crafter actions: 0..16

ACTION_NAMES = [
    "noop",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "do",
    "sleep",
    "place_stone",
    "place_table",
    "place_furnace",
    "place_plant",
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
]

# Inventory keys in the order they appear as probe label columns.
INVENTORY_KEYS = [
    # 4 vitals
    "health",
    "food",
    "drink",
    "energy",
    # 6 resources
    "sapling",
    "wood",
    "stone",
    "coal",
    "iron",
    "diamond",
    # 6 tools (binary)
    "wood_pickaxe",
    "stone_pickaxe",
    "iron_pickaxe",
    "wood_sword",
    "stone_sword",
    "iron_sword",
]

MAX_STEPS_PER_EPISODE = 300


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_probe_labels(info: dict) -> np.ndarray:
    """Return a (16,) float32 vector from the info inventory dict."""
    inventory = info["inventory"]
    return np.array([inventory[k] for k in INVENTORY_KEYS], dtype=np.float32)


def collect_episode(env, rng: np.random.RandomState) -> dict:
    """Run one episode with a random policy, return arrays."""
    obs = env.reset()

    observations = [obs]
    actions = []
    rewards = []
    probe_labels = []
    player_positions = []

    for _ in range(MAX_STEPS_PER_EPISODE):
        action = rng.randint(0, NUM_ACTIONS)
        obs, reward, done, info = env.step(action)

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        probe_labels.append(extract_probe_labels(info))
        player_positions.append(np.array(info["player_pos"], dtype=np.int64))

        if done:
            break

    # observations has length T+1 (includes initial obs), actions/rewards have length T.
    # We store observations[:-1] so every array is length T and action[t] is the
    # action taken *from* observations[t].  The final observation (after the last
    # action) is dropped — it can be reconstructed from the environment if needed.
    observations = np.stack(observations[:-1], axis=0).astype(np.uint8)   # [T, 64, 64, 3]
    actions = np.array(actions, dtype=np.int64)                           # [T]
    rewards = np.array(rewards, dtype=np.float32)                         # [T]
    probe_labels = np.stack(probe_labels, axis=0).astype(np.float32)      # [T, 16]
    player_positions = np.stack(player_positions, axis=0).astype(np.int64)  # [T, 2]

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "probe_labels": probe_labels,
        "player_positions": player_positions,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Collect Crafter trajectories with a random policy."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save episode .npz files and metadata.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1000,
        help="Number of episodes to collect (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    rng = np.random.RandomState(args.seed)

    import crafter  # import here so argparse --help works without crafter installed

    env = crafter.Env()

    print(f"Collecting {args.num_episodes} episodes (max {MAX_STEPS_PER_EPISODE} steps each)")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    print(f"Seed: {args.seed}")
    print()

    # ------------------------------------------------------------------
    # Collection loop
    # ------------------------------------------------------------------
    total_steps = 0
    episode_lengths = []
    t_start = time.time()

    for ep_idx in range(args.num_episodes):
        episode = collect_episode(env, rng)
        ep_len = len(episode["actions"])
        total_steps += ep_len
        episode_lengths.append(ep_len)

        # Save episode
        fname = os.path.join(args.output_dir, f"episode_{ep_idx:04d}.npz")
        np.savez_compressed(fname, **episode)

        # Progress
        if (ep_idx + 1) % 100 == 0 or ep_idx == 0:
            elapsed = time.time() - t_start
            eps_per_sec = (ep_idx + 1) / elapsed if elapsed > 0 else 0
            mean_len = np.mean(episode_lengths)
            print(
                f"  Episode {ep_idx + 1:>5d}/{args.num_episodes} | "
                f"steps so far: {total_steps:>8d} | "
                f"mean ep len: {mean_len:>6.1f} | "
                f"speed: {eps_per_sec:.1f} ep/s"
            )

    elapsed = time.time() - t_start

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    metadata = {
        "num_episodes": args.num_episodes,
        "total_steps": int(total_steps),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "std_episode_length": float(np.std(episode_lengths)),
        "min_episode_length": int(np.min(episode_lengths)),
        "max_episode_length": int(np.max(episode_lengths)),
        "action_names": ACTION_NAMES,
        "probe_label_names": INVENTORY_KEYS,
        "seed": args.seed,
        "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
        "collection_time_seconds": round(elapsed, 1),
    }

    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    # Estimate disk size by summing actual file sizes
    disk_bytes = 0
    for name in os.listdir(args.output_dir):
        fpath = os.path.join(args.output_dir, name)
        if os.path.isfile(fpath):
            disk_bytes += os.path.getsize(fpath)

    print()
    print("=" * 60)
    print("Collection complete!")
    print(f"  Episodes collected : {args.num_episodes}")
    print(f"  Total transitions  : {total_steps}")
    print(f"  Mean episode length: {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f}")
    print(f"  Time elapsed       : {elapsed:.1f}s")
    print(f"  Disk usage         : {disk_bytes / (1024**2):.1f} MB ({disk_bytes / (1024**3):.2f} GB)")
    print(f"  Metadata saved to  : {os.path.abspath(meta_path)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
