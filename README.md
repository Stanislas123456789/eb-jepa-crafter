# JEPA World Model for Crafter

> **Hack The World(s) 2026** · Team Dreaming Machines
>
> Built on [EB-JEPA](https://github.com/facebookresearch/eb_jepa) (Meta FAIR) · [Paper](https://arxiv.org/abs/2602.03604)

Can a JEPA learn to **understand** a Minecraft-like survival world — without reward signals, labels, or pixel reconstruction?

We port Meta FAIR's Energy-Based JEPA from a simple 2D toy environment (Two Rooms) to **Crafter**, a procedurally-generated survival game with 17 actions, inventory management, enemies, and crafting. The model learns an abstract latent representation of the world purely from random exploration, and we prove it captures meaningful game state.

## Results

### The model learns to imagine the future

The JEPA predictor autoregressively rolls out future latent states given actions. It achieves **3.8x lower error** than simply repeating the current state:

<p align="center">
  <img src="eval_results/figures/rollout_comparison.png" width="700"/>
</p>

| Model | Mean Latent MSE | vs Copy Baseline |
|-------|----------------|------------------|
| **Full JEPA (with IDM)** | **0.159** | **3.8x better** |
| Ablated (no IDM) | 0.182 | 3.3x better |
| Copy Baseline | 0.609 | — |

### The latent encodes game state without labels

A frozen linear probe on top of the encoder predicts health, food, drink, energy, and inventory — features the model was **never trained to predict**:

<p align="center">
  <img src="eval_results/figures/probe_comparison.png" width="700"/>
</p>

### IDM loss is critical

Removing the Inverse Dynamics Model loss degrades both prediction accuracy and representation quality:

<p align="center">
  <img src="eval_results/figures/ablation_summary.png" width="700"/>
</p>

- Rollout MSE: 0.159 → 0.182 (**+14% worse**)
- Probe R²: 0.815 → 0.562 (**-31% drop**)
- Health probe collapses: 0.67 → 0.19 (**-71%**)

### The model "dreams"

Given a single starting frame and a sequence of actions, the model imagines what will happen:

<p align="center">
  <img src="eval_results/rollout_full/dreaming_sample_0.png" width="900"/>
</p>

## What we changed from the original EB-JEPA

| Component | Two Rooms (original) | Crafter (ours) |
|-----------|---------------------|----------------|
| Actions | 2 continuous (`nn.Identity`) | 17 discrete (`nn.Embedding(17, 32)`) |
| IDM loss | MSE regression | Cross-entropy classification |
| Observations | 2-channel 65×65 | 3-channel RGB 64×64 |
| Data | On-the-fly simulation | Pre-collected offline trajectories |
| Probe | XY position (2D) | 16 game state features |
| Environment | Dot + wall + door | Survival game with enemies, inventory, crafting |

## Quick Start

```bash
# 1. Install
pip install -e .
pip install crafter

# 2. Collect data (~5 min, 1000 episodes of random play)
python scripts/collect_crafter_data.py --output_dir data/crafter_trajectories --num_episodes 1000

# 3. Train (~44 min on single GPU)
python -c "
from examples.ac_video_jepa.main import run
run(fname='examples/ac_video_jepa/cfgs/crafter.yaml',
    **{'data.data_dir': 'data/crafter_trajectories', 'logging.log_wandb': False})
"

# 4. Train IDM ablation
python -c "
from examples.ac_video_jepa.main import run
run(fname='examples/ac_video_jepa/cfgs/crafter.yaml',
    **{'data.data_dir': 'data/crafter_trajectories', 'model.regularizer.idm_coeff': 0, 'logging.log_wandb': False})
"

# 5. Evaluate
python scripts/eval_probe.py --checkpoint_path checkpoints/.../latest.pth.tar --data_dir data/crafter_trajectories
python scripts/eval_rollout.py --checkpoint_path checkpoints/.../latest.pth.tar --data_dir data/crafter_trajectories
```

### SLURM (Dalia cluster)
```bash
sbatch scripts/train_crafter.sh              # Full model
sbatch scripts/train_crafter.sh --ablation   # IDM ablation
sbatch scripts/train_crafter.sh --seed 1000  # Second seed
sbatch scripts/run_all_evals.sh              # All evaluations
```

## Architecture

```
Crafter RGB (64×64×3)
    │
    ▼
ImpalaEncoder (3 ResNet stacks, MaxPool)
    │
    ▼
512-dim latent [B, 512, T, 1, 1]
    │                    ┌─────────────────────┐
    ▼                    │ nn.Embedding(17, 32) │
GRU Predictor ◄──────── │ (discrete actions)   │
    │                    └─────────────────────┘
    ▼
Next latent state
    │
    ├──► VCReg (variance + covariance regularization)
    ├──► Temporal Similarity loss
    └──► IDM (Inverse Dynamics Model, cross-entropy)
```

## Project Structure

```
eb_jepa/
├── eb_jepa/
│   ├── action_encoders.py          # NEW: Discrete action embedding
│   ├── architectures.py            # ImpalaEncoder, RNNPredictor, IDM
│   ├── jepa.py                     # JEPA class (modified for discrete actions)
│   ├── losses.py                   # VCReg + IDM loss (added cross-entropy mode)
│   └── datasets/
│       ├── crafter/                # NEW: Crafter dataset pipeline
│       │   ├── crafter_dataset.py  # TrajDataset + SlicedDataset
│       │   ├── normalizer.py       # Per-channel z-score normalization
│       │   └── data_config.yaml
│       └── utils.py                # Modified: added crafter dispatch
├── examples/ac_video_jepa/
│   ├── main.py                     # Modified: discrete action builder
│   └── cfgs/crafter.yaml           # NEW: Crafter training config
├── scripts/
│   ├── collect_crafter_data.py     # Data collection (random policy)
│   ├── eval_probe.py               # Linear probe evaluation
│   ├── eval_rollout.py             # Rollout MSE + dreaming visualization
│   ├── make_figures.py             # Figure generation
│   ├── train_crafter.sh            # SLURM training script
│   └── run_all_evals.sh            # SLURM eval pipeline
└── eval_results/                   # Generated results and figures
```

## Honest Limits

- **Random CNN features are competitive on probes.** Untrained ConvNets extract useful spatial features from structured images (known phenomenon). The IDM ablation is the stronger differentiator.
- **Rare features have degenerate R².** With random-policy data, tools and rare resources are almost always zero, making R² unreliable for those features.
- **Partial observability.** Crafter is partially observed — the model can only predict what's visible, limiting long-horizon accuracy.
- **No planning demo.** We didn't close the loop with a planning agent (stretch goal). The world model understands dynamics but doesn't act.

## Acknowledgments

Built on [EB-JEPA](https://github.com/facebookresearch/eb_jepa) by Meta FAIR (Terver, Balestriero, Dervishi, Fan, Garrido, Nagarajan, Sinha, Zhang, Rabbat, LeCun, Bar). Licensed under Apache 2.0.

Crafter environment by [Danijar Hafner](https://github.com/danijar/crafter).

Trained on NVIDIA GB200 GPUs at the Dalia cluster (IDRIS).
