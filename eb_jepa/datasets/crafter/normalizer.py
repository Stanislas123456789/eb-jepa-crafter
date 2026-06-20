import torch
import numpy as np


class CrafterNormalizer:
    """
    Per-channel z-score normalizer for Crafter RGB observations.
    Supports tensors of shape [C, H, W], [C, T, H, W], or [B, C, T, H, W].
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """
        Args:
            mean: per-channel mean, shape [3]
            std: per-channel std, shape [3]
        """
        self.state_mean = mean.float()
        self.state_std = std.float()

    def _reshape_stats(self, x: torch.Tensor):
        """Reshape mean/std to broadcast against x.

        Supports:
            [C, H, W]       -> mean shape [C, 1, 1]
            [C, T, H, W]    -> mean shape [C, 1, 1, 1]
            [B, C, T, H, W] -> mean shape [1, C, 1, 1, 1]
        """
        mean = self.state_mean.to(x.device)
        std = self.state_std.to(x.device)

        if x.dim() == 3:
            # [C, H, W]
            return mean.view(-1, 1, 1), std.view(-1, 1, 1)
        elif x.dim() == 4:
            # [C, T, H, W]
            return mean.view(-1, 1, 1, 1), std.view(-1, 1, 1, 1)
        elif x.dim() == 5:
            # [B, C, T, H, W]
            return mean.view(1, -1, 1, 1, 1), std.view(1, -1, 1, 1, 1)
        else:
            raise ValueError(f"Unsupported tensor dimension: {x.dim()}")

    def normalize_state(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize per channel: (x - mean) / std.
        Args:
            x: tensor with channel dim = 3 (RGB).
        Returns:
            Normalized tensor with same shape.
        """
        mean, std = self._reshape_stats(x)
        return (x - mean) / (std + 1e-6)

    def unnormalize_state(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse of normalize_state.
        """
        mean, std = self._reshape_stats(x)
        return x * (std + 1e-6) + mean

    def unnormalize_mse(self, mse: torch.Tensor) -> torch.Tensor:
        """
        Unnormalize a mean squared error scalar.
        Approximation using mean of per-channel variances.
        """
        return mse * (self.state_std.mean().to(mse.device) ** 2)

    @classmethod
    def compute_stats(cls, dataset, num_samples: int = 5000) -> "CrafterNormalizer":
        """
        Compute per-channel mean and std over a random subset of frames.

        Args:
            dataset: CrafterTrajDataset instance (raw episodes).
            num_samples: number of frames to sample for computing stats.
        Returns:
            CrafterNormalizer with computed statistics.
        """
        # Collect random frames
        all_pixels = []
        frames_collected = 0
        num_episodes = len(dataset)
        rng = np.random.RandomState(42)

        while frames_collected < num_samples:
            ep_idx = rng.randint(0, num_episodes)
            ep = dataset[ep_idx]
            obs = ep["observations"]  # [T, H, W, C] uint8
            T = obs.shape[0]
            # Pick a random frame
            t = rng.randint(0, T)
            frame = obs[t].astype(np.float32) / 255.0  # [H, W, C] in [0, 1]
            all_pixels.append(frame)
            frames_collected += 1

        all_pixels = np.stack(all_pixels, axis=0)  # [N, H, W, C]
        # Compute per-channel stats
        mean = all_pixels.mean(axis=(0, 1, 2))  # [C]
        std = all_pixels.std(axis=(0, 1, 2))  # [C]

        return cls(
            mean=torch.tensor(mean, dtype=torch.float32),
            std=torch.tensor(std, dtype=torch.float32),
        )
