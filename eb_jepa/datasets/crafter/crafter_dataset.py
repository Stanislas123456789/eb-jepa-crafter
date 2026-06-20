import glob
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from eb_jepa.datasets.traj_dset import TrajDataset
from eb_jepa.datasets.crafter.normalizer import CrafterNormalizer


class CrafterTrajDataset(TrajDataset):
    """
    Loads Crafter episode .npz files into memory.
    Each .npz contains:
        - observations: [T, 64, 64, 3] uint8
        - actions: [T] int64
        - probe_labels: [T, 16] float32
        - player_positions: [T, 2] int64
    """

    action_dim = 1  # single integer action
    proprio_dim = 0
    state_dim = 16

    def __init__(self, data_dir: str, sample_length: int = 17):
        super().__init__()
        self.data_dir = data_dir
        self.sample_length = sample_length

        # Find and sort all episode files
        pattern = os.path.join(data_dir, "episode_*.npz")
        self.episode_paths = sorted(glob.glob(pattern))
        if len(self.episode_paths) == 0:
            raise FileNotFoundError(
                f"No episode_*.npz files found in {data_dir}"
            )

        # Load all episodes into memory (Crafter episodes are small)
        self.episodes = []
        for path in self.episode_paths:
            data = np.load(path)
            self.episodes.append(
                {
                    "observations": data["observations"],  # [T, 64, 64, 3] uint8
                    "actions": data["actions"],  # [T] int64
                    "probe_labels": data["probe_labels"],  # [T, 16] float32
                    "player_positions": data["player_positions"],  # [T, 2] int64
                }
            )

        print(
            f"Loaded {len(self.episodes)} Crafter episodes from {data_dir} "
            f"(total frames: {sum(e['observations'].shape[0] for e in self.episodes)})"
        )

    def get_seq_length(self, idx: int) -> int:
        return self.episodes[idx]["observations"].shape[0]

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict:
        return self.episodes[idx]


class CrafterSlicedDataset(Dataset):
    """
    Wraps CrafterTrajDataset by slicing episodes into fixed-length clips.
    Returns the 5-tuple expected by the training loop:
        (obs, actions, probe_labels, dummy, dummy)

    obs:          [C, T, H, W] float32, per-channel z-score normalized
    actions:      [T] int64 (discrete action indices)
    probe_labels: [K, T] float32 (K=16)
    dummy:        torch.tensor(0.0)
    """

    def __init__(
        self,
        traj_dataset: CrafterTrajDataset,
        sample_length: int = 17,
        normalizer: Optional[CrafterNormalizer] = None,
        num_stats_samples: int = 5000,
    ):
        super().__init__()
        self.traj_dataset = traj_dataset
        self.sample_length = sample_length

        # Build all valid (episode_idx, start_frame) pairs
        self.slices = []
        for ep_idx in range(len(traj_dataset)):
            T = traj_dataset.get_seq_length(ep_idx)
            if T < sample_length:
                print(
                    f"Skipping short episode {ep_idx}: len={T} < sample_length={sample_length}"
                )
                continue
            for start in range(T - sample_length + 1):
                self.slices.append((ep_idx, start))

        # Shuffle slices deterministically
        rng = torch.Generator().manual_seed(42)
        order = torch.randperm(len(self.slices), generator=rng).tolist()
        self.slices = [self.slices[i] for i in order]

        # Set up normalizer
        if normalizer is not None:
            self.normalizer = normalizer
        else:
            print("Computing per-channel normalization stats...")
            self.normalizer = CrafterNormalizer.compute_stats(
                traj_dataset, num_samples=num_stats_samples
            )
            print(
                f"  mean={self.normalizer.state_mean.tolist()}, "
                f"std={self.normalizer.state_std.tolist()}"
            )

        # Dataset attributes for compatibility
        self.action_dim = traj_dataset.action_dim
        self.proprio_dim = traj_dataset.proprio_dim
        self.state_dim = traj_dataset.state_dim

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int):
        ep_idx, start = self.slices[idx]
        ep = self.traj_dataset[ep_idx]

        end = start + self.sample_length

        # --- observations ---
        # [T, H, W, C] uint8 -> float32 [0, 1]
        obs_np = ep["observations"][start:end]  # [T, 64, 64, 3]
        obs = torch.from_numpy(obs_np.copy()).float() / 255.0  # [T, H, W, C]
        # Permute to [C, T, H, W]
        obs = obs.permute(3, 0, 1, 2)  # [C, T, H, W]
        # Normalize per-channel
        obs = self.normalizer.normalize_state(obs)

        # --- actions ---
        actions = torch.from_numpy(ep["actions"][start:end].copy()).long()  # [T]

        # --- probe labels ---
        probe = torch.from_numpy(
            ep["probe_labels"][start:end].copy()
        ).float()  # [T, K]
        probe_labels = probe.permute(1, 0)  # [K, T]

        # --- dummies (match two_rooms wall_x/door_y placeholders) ---
        dummy = torch.tensor(0.0)

        return obs, actions, probe_labels, dummy, dummy
