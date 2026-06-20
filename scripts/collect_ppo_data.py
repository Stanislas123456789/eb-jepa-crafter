#!/usr/bin/env python3
"""
Train a PPO agent on Crafter, then collect trajectories from the trained policy.

Produces episodes in the same .npz format as collect_crafter_data.py, plus an
optional mixed dataset (50% PPO + 50% random) for diversity.

Requirements:
    pip install stable-baselines3

Usage:
    python scripts/collect_ppo_data.py --output_dir data/crafter_ppo
    python scripts/collect_ppo_data.py --output_dir data/crafter_ppo --train_steps 200000 --collect_episodes 500 --seed 42
"""

import argparse
import json
import os
import time

import numpy as np

# ---------------------------------------------------------------------------
# Constants (shared with collect_crafter_data.py)
# ---------------------------------------------------------------------------

NUM_ACTIONS = 17

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

INVENTORY_KEYS = [
    "health",
    "food",
    "drink",
    "energy",
    "sapling",
    "wood",
    "stone",
    "coal",
    "iron",
    "diamond",
    "wood_pickaxe",
    "stone_pickaxe",
    "iron_pickaxe",
    "wood_sword",
    "stone_sword",
    "iron_sword",
]

MAX_STEPS_PER_EPISODE = 300

# Crafter achievements used for computing the Crafter score
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
# Gymnasium wrapper for Crafter
# ---------------------------------------------------------------------------


def _make_crafter_gym_env():
    """Create a Crafter env wrapped in the gymnasium API."""
    import gymnasium as gym

    class CrafterGymWrapper(gym.Env):
        """Wraps Crafter (old-style gym) into gymnasium (new-style) API."""

        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            import crafter

            self._env = crafter.Env()
            self.observation_space = gym.spaces.Box(
                0, 255, (64, 64, 3), dtype=np.uint8
            )
            self.action_space = gym.spaces.Discrete(NUM_ACTIONS)

        def reset(self, seed=None, options=None):
            if seed is not None:
                # Crafter doesn't support seeded reset directly, but we can
                # set numpy seed for reproducibility.
                np.random.seed(seed)
            obs = self._env.reset()
            return obs.astype(np.uint8), {}

        def step(self, action):
            obs, reward, done, info = self._env.step(action)
            return obs.astype(np.uint8), reward, done, False, info

        def render(self):
            pass

        def close(self):
            pass

    return CrafterGymWrapper()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_probe_labels(info: dict) -> np.ndarray:
    """Return a (16,) float32 vector from the info inventory dict."""
    inventory = info["inventory"]
    return np.array([inventory[k] for k in INVENTORY_KEYS], dtype=np.float32)


def collect_episode_with_policy(env, policy_fn) -> dict:
    """Run one episode using a policy function, return arrays in standard format.

    Args:
        env: Raw crafter.Env (old-style gym API).
        policy_fn: Callable(obs) -> action (int).
    """
    obs = env.reset()

    observations = [obs]
    actions = []
    rewards = []
    probe_labels = []
    player_positions = []

    for _ in range(MAX_STEPS_PER_EPISODE):
        action = policy_fn(obs)
        obs, reward, done, info = env.step(action)

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        probe_labels.append(extract_probe_labels(info))
        player_positions.append(np.array(info["player_pos"], dtype=np.int64))

        if done:
            break

    # Same convention as collect_crafter_data.py: observations[:-1]
    observations = np.stack(observations[:-1], axis=0).astype(np.uint8)
    actions = np.array(actions, dtype=np.int64)
    rewards = np.array(rewards, dtype=np.float32)
    probe_labels = np.stack(probe_labels, axis=0).astype(np.float32)
    player_positions = np.stack(player_positions, axis=0).astype(np.int64)

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "probe_labels": probe_labels,
        "player_positions": player_positions,
    }


def compute_crafter_score(achievement_counts: dict, num_episodes: int) -> float:
    """Compute the Crafter score (geometric mean of success rates)."""
    rates = []
    for ach in CRAFTER_ACHIEVEMENTS:
        rate = achievement_counts.get(ach, 0) / max(num_episodes, 1)
        rates.append(rate)
    # Geometric mean with +1% smoothing (as in the Crafter paper)
    log_rates = [np.log(max(r, 1e-6) + 0.01) for r in rates]
    score = np.exp(np.mean(log_rates)) - 0.01
    return max(score, 0.0)


def save_episodes(episodes, output_dir, start_idx=0):
    """Save a list of episode dicts as .npz files."""
    for i, episode in enumerate(episodes):
        fname = os.path.join(output_dir, f"episode_{start_idx + i:04d}.npz")
        np.savez_compressed(fname, **episode)


def save_metadata(output_dir, episodes, extra_info=None):
    """Save metadata.json summarizing the collected episodes."""
    episode_lengths = [len(ep["actions"]) for ep in episodes]
    total_steps = sum(episode_lengths)

    metadata = {
        "num_episodes": len(episodes),
        "total_steps": int(total_steps),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "std_episode_length": float(np.std(episode_lengths)),
        "min_episode_length": int(np.min(episode_lengths)),
        "max_episode_length": int(np.max(episode_lengths)),
        "action_names": ACTION_NAMES,
        "probe_label_names": INVENTORY_KEYS,
        "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
    }
    if extra_info:
        metadata.update(extra_info)

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return meta_path


# ---------------------------------------------------------------------------
# Fallback policy (if stable-baselines3 is not available)
# ---------------------------------------------------------------------------


class HeuristicPolicy:
    """Simple exploration policy: biased toward movement + interaction.

    Better than pure random because it moves more and uses 'do' action
    more often, leading to more resource collection.
    """

    def __init__(self, rng: np.random.RandomState):
        self.rng = rng
        # Weighted probabilities favoring movement and 'do'
        weights = np.ones(NUM_ACTIONS, dtype=np.float64)
        weights[1:5] = 5.0   # move actions
        weights[5] = 8.0     # 'do' action (interact/attack/collect)
        weights[6] = 0.5     # sleep (less useful)
        weights[7:11] = 1.5  # place actions
        weights[11:] = 1.0   # make actions
        self.probs = weights / weights.sum()

    def __call__(self, obs):
        return self.rng.choice(NUM_ACTIONS, p=self.probs)


# ---------------------------------------------------------------------------
# Training callback for progress logging
# ---------------------------------------------------------------------------


def _make_progress_callback(log_interval=10000):
    """Create an SB3-compatible callback that logs every log_interval steps."""
    from stable_baselines3.common.callbacks import BaseCallback

    class ProgressCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self._last_log = 0

        def _on_step(self) -> bool:
            if self.num_timesteps - self._last_log >= log_interval:
                self._last_log = self.num_timesteps
                print(
                    f"  [Training] {self.num_timesteps:>8d} steps completed"
                )
            return True

    return ProgressCallback()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO on Crafter and collect trajectories."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save episode .npz files and metadata.",
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=200_000,
        help="Number of PPO training steps (default: 200000).",
    )
    parser.add_argument(
        "--collect_episodes",
        type=int,
        default=500,
        help="Number of episodes to collect from trained policy (default: 500).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--no_mixed",
        action="store_true",
        help="Skip creating the mixed (PPO + random) dataset.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for PPO training: 'auto', 'cpu', 'cuda', 'mps' (default: auto).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    # ------------------------------------------------------------------
    # Check if stable-baselines3 is available
    # ------------------------------------------------------------------
    use_sb3 = True
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        print("stable-baselines3 found. Using PPO with CnnPolicy.")
    except ImportError:
        use_sb3 = False
        print(
            "WARNING: stable-baselines3 not found. "
            "Falling back to heuristic policy.\n"
            "  Install with: pip install stable-baselines3\n"
        )

    # ------------------------------------------------------------------
    # Phase 1: Train (or prepare fallback policy)
    # ------------------------------------------------------------------
    if use_sb3:
        print(f"\n{'='*60}")
        print("Phase 1: Training PPO agent")
        print(f"  Steps: {args.train_steps}")
        print(f"  Seed:  {args.seed}")
        print(f"  Device: {args.device}")
        print(f"{'='*60}\n")

        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        train_env = DummyVecEnv([_make_crafter_gym_env])

        model = PPO(
            "CnnPolicy",
            train_env,
            verbose=0,
            seed=args.seed,
            n_steps=256,
            batch_size=64,
            n_epochs=4,
            learning_rate=3e-4,
            clip_range=0.2,
            ent_coef=0.01,
            device=args.device,
        )

        t_train_start = time.time()
        callback = _make_progress_callback(log_interval=10_000)
        model.learn(total_timesteps=args.train_steps, callback=callback)
        t_train_elapsed = time.time() - t_train_start

        print(f"\nTraining complete in {t_train_elapsed:.1f}s")

        # Save model
        model_path = os.path.join(args.output_dir, "ppo_crafter_model")
        model.save(model_path)
        print(f"Model saved to {model_path}.zip")

        train_env.close()

        # Create policy function for collection (uses raw crafter obs)
        def ppo_policy_fn(obs):
            # SB3 expects (1, H, W, C) batch
            obs_batch = np.expand_dims(obs, axis=0)
            action, _ = model.predict(obs_batch, deterministic=False)
            return int(action[0])

        policy_fn = ppo_policy_fn
        policy_name = "ppo"

    else:
        print(f"\n{'='*60}")
        print("Phase 1: Using heuristic policy (stable-baselines3 not available)")
        print(f"{'='*60}\n")

        heuristic = HeuristicPolicy(rng)
        policy_fn = heuristic
        policy_name = "heuristic"
        t_train_elapsed = 0.0

    # ------------------------------------------------------------------
    # Evaluate agent (compute Crafter score on a few episodes)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Evaluating trained agent (20 episodes)...")
    print(f"{'='*60}\n")

    import crafter

    eval_env = crafter.Env()
    eval_episodes = 20
    achievement_counts = {}
    eval_rewards = []

    for ep_i in range(eval_episodes):
        obs = eval_env.reset()
        ep_reward = 0.0
        for _ in range(MAX_STEPS_PER_EPISODE):
            action = policy_fn(obs)
            obs, reward, done, info = eval_env.step(action)
            ep_reward += reward

            # Track achievements
            if "achievements" in info:
                for ach_name, ach_val in info["achievements"].items():
                    if ach_val > 0:
                        achievement_counts[ach_name] = (
                            achievement_counts.get(ach_name, 0) + 1
                        )
            if done:
                break
        eval_rewards.append(ep_reward)

    crafter_score = compute_crafter_score(achievement_counts, eval_episodes)
    mean_reward = np.mean(eval_rewards)

    print(f"  Crafter score: {crafter_score * 100:.2f}%")
    print(f"  Mean episode reward: {mean_reward:.2f}")
    print(f"  Achievement rates:")
    for ach in CRAFTER_ACHIEVEMENTS:
        rate = achievement_counts.get(ach, 0) / eval_episodes
        if rate > 0:
            print(f"    {ach}: {rate:.0%}")

    # ------------------------------------------------------------------
    # Phase 2: Collect episodes from trained policy
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Phase 2: Collecting {args.collect_episodes} episodes ({policy_name} policy)")
    print(f"{'='*60}\n")

    collect_env = crafter.Env()
    ppo_episodes = []
    t_collect_start = time.time()

    for ep_idx in range(args.collect_episodes):
        episode = collect_episode_with_policy(collect_env, policy_fn)
        ppo_episodes.append(episode)

        if (ep_idx + 1) % 100 == 0 or ep_idx == 0:
            total_steps = sum(len(ep["actions"]) for ep in ppo_episodes)
            mean_len = np.mean([len(ep["actions"]) for ep in ppo_episodes])
            elapsed = time.time() - t_collect_start
            eps_per_sec = (ep_idx + 1) / elapsed if elapsed > 0 else 0
            print(
                f"  Episode {ep_idx + 1:>5d}/{args.collect_episodes} | "
                f"steps so far: {total_steps:>8d} | "
                f"mean ep len: {mean_len:>6.1f} | "
                f"speed: {eps_per_sec:.1f} ep/s"
            )

    t_collect_elapsed = time.time() - t_collect_start

    # Save PPO episodes
    ppo_dir = os.path.join(args.output_dir, "ppo")
    os.makedirs(ppo_dir, exist_ok=True)
    save_episodes(ppo_episodes, ppo_dir)

    meta_path = save_metadata(
        ppo_dir,
        ppo_episodes,
        extra_info={
            "policy": policy_name,
            "seed": args.seed,
            "train_steps": args.train_steps if use_sb3 else 0,
            "training_time_seconds": round(t_train_elapsed, 1),
            "collection_time_seconds": round(t_collect_elapsed, 1),
            "crafter_score": round(crafter_score * 100, 2),
            "mean_eval_reward": round(float(mean_reward), 2),
        },
    )

    print(f"\nPPO episodes saved to {os.path.abspath(ppo_dir)}")
    print(f"Metadata saved to {os.path.abspath(meta_path)}")

    # ------------------------------------------------------------------
    # Phase 3: Collect mixed dataset (50% PPO + 50% random)
    # ------------------------------------------------------------------
    if not args.no_mixed:
        n_random = args.collect_episodes  # same count as PPO
        print(f"\n{'='*60}")
        print(f"Phase 3: Collecting {n_random} random episodes for mixed dataset")
        print(f"{'='*60}\n")

        random_policy = lambda obs: rng.randint(0, NUM_ACTIONS)
        random_episodes = []
        t_random_start = time.time()

        for ep_idx in range(n_random):
            episode = collect_episode_with_policy(collect_env, random_policy)
            random_episodes.append(episode)

            if (ep_idx + 1) % 100 == 0 or ep_idx == 0:
                total_steps = sum(len(ep["actions"]) for ep in random_episodes)
                mean_len = np.mean([len(ep["actions"]) for ep in random_episodes])
                elapsed = time.time() - t_random_start
                eps_per_sec = (ep_idx + 1) / elapsed if elapsed > 0 else 0
                print(
                    f"  Episode {ep_idx + 1:>5d}/{n_random} | "
                    f"steps so far: {total_steps:>8d} | "
                    f"mean ep len: {mean_len:>6.1f} | "
                    f"speed: {eps_per_sec:.1f} ep/s"
                )

        t_random_elapsed = time.time() - t_random_start

        # Build mixed dataset: interleave PPO and random episodes
        mixed_dir = os.path.join(args.output_dir, "mixed")
        os.makedirs(mixed_dir, exist_ok=True)

        # Shuffle the combined episodes
        mixed_episodes = list(ppo_episodes) + list(random_episodes)
        mix_rng = np.random.RandomState(args.seed + 1)
        mix_indices = mix_rng.permutation(len(mixed_episodes))
        mixed_episodes = [mixed_episodes[i] for i in mix_indices]

        # Track provenance (which are PPO vs random)
        provenance = []
        n_ppo = len(ppo_episodes)
        for idx in mix_indices:
            provenance.append("ppo" if idx < n_ppo else "random")

        save_episodes(mixed_episodes, mixed_dir)
        meta_path_mixed = save_metadata(
            mixed_dir,
            mixed_episodes,
            extra_info={
                "policy": "mixed (50% ppo + 50% random)",
                "seed": args.seed,
                "train_steps": args.train_steps if use_sb3 else 0,
                "training_time_seconds": round(t_train_elapsed, 1),
                "collection_time_seconds": round(
                    t_collect_elapsed + t_random_elapsed, 1
                ),
                "num_ppo_episodes": len(ppo_episodes),
                "num_random_episodes": len(random_episodes),
                "crafter_score_ppo": round(crafter_score * 100, 2),
            },
        )

        print(f"\nMixed episodes saved to {os.path.abspath(mixed_dir)}")
        print(f"  {len(ppo_episodes)} PPO + {len(random_episodes)} random = {len(mixed_episodes)} total")
        print(f"Metadata saved to {os.path.abspath(meta_path_mixed)}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    disk_bytes = 0
    for root, dirs, files in os.walk(args.output_dir):
        for name in files:
            fpath = os.path.join(root, name)
            if os.path.isfile(fpath):
                disk_bytes += os.path.getsize(fpath)

    total_ppo_steps = sum(len(ep["actions"]) for ep in ppo_episodes)
    ppo_lengths = [len(ep["actions"]) for ep in ppo_episodes]

    print(f"\n{'='*60}")
    print("Collection complete!")
    print(f"  Policy             : {policy_name}")
    print(f"  PPO episodes       : {len(ppo_episodes)}")
    print(f"  PPO total steps    : {total_ppo_steps}")
    print(f"  PPO mean ep length : {np.mean(ppo_lengths):.1f} +/- {np.std(ppo_lengths):.1f}")
    print(f"  Crafter score      : {crafter_score * 100:.2f}%")
    print(f"  Training time      : {t_train_elapsed:.1f}s")
    print(f"  Collection time    : {t_collect_elapsed:.1f}s")
    print(f"  Total disk usage   : {disk_bytes / (1024**2):.1f} MB")
    print(f"  Output directory   : {os.path.abspath(args.output_dir)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
