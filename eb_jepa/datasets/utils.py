from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from eb_jepa.datasets.two_rooms.utils import update_config_from_yaml
from eb_jepa.datasets.two_rooms.wall_dataset import WallDataset, WallDatasetConfig

DATASETS_DIR = Path(__file__).parent


def load_env_data_config(env_name: str, overrides: dict = None) -> dict:
    """Load base data config for an environment and apply overrides."""
    config_path = DATASETS_DIR / env_name / "data_config.yaml"
    with open(config_path) as f:
        base_config = yaml.safe_load(f)
    if overrides:
        base_config.update(overrides)
    return base_config


def _resolve_oc_env(value: str) -> str:
    """Resolve OmegaConf-style ${oc.env:VAR,default} references in strings."""
    import os
    import re

    pattern = r"\$\{oc\.env:([^,}]+)(?:,([^}]*))?\}"
    match = re.match(pattern, str(value))
    if match:
        var_name, default = match.group(1), match.group(2)
        return os.environ.get(var_name, default if default is not None else "")
    return value


def init_data(env_name, cfg_data=None, **kwargs):
    """Initialize data loaders for the specified environment.

    Loads base config from eb_jepa/datasets/{env_name}/data_config.yaml
    and merges with any overrides from cfg_data.

    Args:
        env_name: Name of the environment ("two_rooms" or "crafter").
        cfg_data: Configuration overrides for the dataset.

    Returns:
        Tuple of (train_loader, val_loader, config).
    """
    if env_name == "two_rooms":
        return _init_two_rooms(cfg_data, **kwargs)
    elif env_name == "crafter":
        return _init_crafter(cfg_data, **kwargs)
    else:
        raise ValueError(
            f"Unknown env: {env_name}. Supported: 'two_rooms', 'crafter'."
        )


def _init_two_rooms(cfg_data=None, **kwargs):
    """Initialize Two Rooms data loaders."""
    merged_cfg = load_env_data_config("two_rooms", cfg_data)
    config = update_config_from_yaml(WallDatasetConfig, merged_cfg)

    num_workers = merged_cfg.get("num_workers", 0)
    pin_mem = merged_cfg.get("pin_mem", False)
    persistent_workers = merged_cfg.get("persistent_workers", False) and num_workers > 0

    dset = WallDataset(config=config)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
        persistent_workers=persistent_workers,
    )

    val_dset = WallDataset(config=config)
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=4,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
        persistent_workers=persistent_workers,
    )

    return loader, val_loader, config


def _init_crafter(cfg_data=None, **kwargs):
    """Initialize Crafter data loaders.

    Loads episode .npz files, creates sliced train/val datasets, and returns
    DataLoaders plus a config namespace compatible with the training loop.
    """
    from eb_jepa.datasets.crafter.crafter_dataset import (
        CrafterTrajDataset,
        CrafterSlicedDataset,
    )

    merged_cfg = load_env_data_config("crafter", cfg_data)

    # Resolve data_dir (handle OmegaConf-style env var references)
    data_dir = _resolve_oc_env(merged_cfg["data_dir"])
    sample_length = merged_cfg.get("sample_length", 17)
    batch_size = merged_cfg.get("batch_size", 64)
    num_workers = merged_cfg.get("num_workers", 4)
    pin_mem = merged_cfg.get("pin_mem", True)
    persistent_workers = merged_cfg.get("persistent_workers", True) and num_workers > 0
    img_size = merged_cfg.get("img_size", 64)
    train_fraction = merged_cfg.get("train_fraction", 0.9)

    # Load raw trajectory dataset
    traj_dataset = CrafterTrajDataset(data_dir=data_dir, sample_length=sample_length)

    # Split episodes into train/val (90/10 by default)
    num_episodes = len(traj_dataset)
    num_train = int(train_fraction * num_episodes)
    num_val = num_episodes - num_train

    indices = torch.randperm(num_episodes, generator=torch.Generator().manual_seed(42))
    train_indices = indices[:num_train].tolist()
    val_indices = indices[num_train:].tolist()

    # Create train/val trajectory datasets (subset views)
    train_traj = _SubsetTrajDataset(traj_dataset, train_indices)
    val_traj = _SubsetTrajDataset(traj_dataset, val_indices)

    # Create sliced datasets; compute normalizer on training data
    train_dset = CrafterSlicedDataset(
        train_traj, sample_length=sample_length, num_stats_samples=5000
    )
    # Share normalizer from train to val
    val_dset = CrafterSlicedDataset(
        val_traj, sample_length=sample_length, normalizer=train_dset.normalizer
    )

    print(
        f"Crafter split: {num_train} train episodes ({len(train_dset)} slices), "
        f"{num_val} val episodes ({len(val_dset)} slices)"
    )

    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
        persistent_workers=persistent_workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=min(batch_size, len(val_dset)) if len(val_dset) > 0 else 1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
        persistent_workers=persistent_workers,
    )

    # Build config namespace compatible with the training loop
    config = SimpleNamespace(
        batch_size=batch_size,
        img_size=img_size,
        size=len(train_dset),
        val_size=len(val_dset),
        sample_length=sample_length,
    )

    return train_loader, val_loader, config


class _SubsetTrajDataset:
    """Lightweight subset wrapper for CrafterTrajDataset."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.action_dim = dataset.action_dim
        self.proprio_dim = dataset.proprio_dim
        self.state_dim = dataset.state_dim

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def get_seq_length(self, idx):
        return self.dataset.get_seq_length(self.indices[idx])
