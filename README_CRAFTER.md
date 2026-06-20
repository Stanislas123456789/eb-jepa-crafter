# EB-JEPA × Crafter — World Model for a Survival Game

> Hack The World(s) 2026 · Team Dreaming Machines

## What is this?

An action-conditioned JEPA world model ported from the Two Rooms example to **Crafter**, a procedurally-generated Minecraft-like survival game. The model learns abstract world dynamics from random exploration, without reward signals or pixel reconstruction.

## Key Results

| Metric | Full Model | No-IDM Ablation | Copy Baseline |
|--------|-----------|----------------|---------------|
| Rollout MSE | **0.159** | 0.182 | 0.609 |
| Probe R² (mean) | **0.815** | 0.562 | — |
| vs Copy Baseline | **3.8× better** | 3.3× better | — |

## Quick Start

### 1. Collect Data (~5 min)
```bash
pip install crafter
python scripts/collect_crafter_data.py --output_dir data/crafter_trajectories --num_episodes 1000
```

### 2. Train (~44 min on single GPU)
```bash
python -c "
from examples.ac_video_jepa.main import run
run(fname='examples/ac_video_jepa/cfgs/crafter.yaml',
    **{'data.data_dir': 'data/crafter_trajectories', 'logging.log_wandb': False})
"
```

### 3. Train IDM Ablation
```bash
python -c "
from examples.ac_video_jepa.main import run
run(fname='examples/ac_video_jepa/cfgs/crafter.yaml',
    **{'data.data_dir': 'data/crafter_trajectories', 'model.regularizer.idm_coeff': 0, 'logging.log_wandb': False})
"
```

### 4. Evaluate
```bash
# Probe evaluation (frozen encoder → game state)
python scripts/eval_probe.py --checkpoint_path checkpoints/.../latest.pth.tar --data_dir data/crafter_trajectories

# Rollout evaluation (MSE vs horizon + dreaming visualizations)
python scripts/eval_rollout.py --checkpoint_path checkpoints/.../latest.pth.tar --data_dir data/crafter_trajectories
```

### SLURM (Dalia cluster)
```bash
sbatch scripts/train_crafter.sh              # Full model
sbatch scripts/train_crafter.sh --ablation   # IDM ablation
sbatch scripts/train_crafter.sh --seed 1000  # Second seed
sbatch scripts/run_all_evals.sh              # Run all evaluations
```

## What We Changed from Two Rooms

| Component | Two Rooms | Crafter |
|-----------|-----------|---------|
| Actions | 2 continuous (nn.Identity) | 17 discrete (nn.Embedding) |
| IDM loss | MSE regression | Cross-entropy classification |
| Observations | 2-channel 65×65 | 3-channel RGB 64×64 |
| Data | On-the-fly generation | Pre-collected .npz trajectories |
| Probe | XY position | 16 game state features (health, inventory) |

## Files Added/Modified

### New files
- `eb_jepa/action_encoders.py` — Discrete action embedding encoder
- `eb_jepa/datasets/crafter/` — Dataset, normalizer, config
- `examples/ac_video_jepa/cfgs/crafter.yaml` — Crafter training config
- `scripts/collect_crafter_data.py` — Data collection
- `scripts/eval_probe.py` — Linear probe evaluation
- `scripts/eval_rollout.py` — Rollout accuracy + dreaming visualization
- `scripts/make_figures.py` — Figure generation
- `scripts/train_crafter.sh` — SLURM training script
- `scripts/run_all_evals.sh` — SLURM eval script

### Modified files
- `eb_jepa/jepa.py` — Fixed action dimension check for discrete actions
- `eb_jepa/losses.py` — Added discrete mode to InverseDynamicsLoss
- `eb_jepa/datasets/utils.py` — Added crafter branch to init_data()
- `examples/ac_video_jepa/main.py` — Added discrete action builder + probe gating
