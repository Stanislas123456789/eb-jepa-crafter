#!/bin/bash
#SBATCH --job-name=crafter-jepa
#SBATCH --account=vivatech-dreamingmachines
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH --gres=gpu:b200:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --reservation=Vivatech
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Usage:
#   sbatch scripts/train_crafter.sh               # full model (idm_coeff=1)
#   sbatch scripts/train_crafter.sh --ablation     # IDM ablation (idm_coeff=0)
#   sbatch scripts/train_crafter.sh --seed 1000    # custom seed

set -e

# Use the team's env.sh which handles aarch64 venv, paths, modules
source ~/eb_jepa/env.sh
module load python312 2>/dev/null || true

# Activate the arch-specific venv (aarch64 on compute nodes)
source $UV_PROJECT_ENVIRONMENT/bin/activate

cd ~/eb_jepa

# Parse args
SEED=1
IDM_COEFF=1
for arg in "$@"; do
    case $arg in
        --ablation) IDM_COEFF=0 ;;
        --seed) shift; SEED=$1 ;;
        --seed=*) SEED="${arg#*=}" ;;
    esac
    shift 2>/dev/null || true
done

echo "=== Crafter JEPA Training ==="
echo "  Seed: $SEED"
echo "  IDM coeff: $IDM_COEFF"
echo "  Arch: $(uname -m)"
echo "  Python: $(which python3)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Node: $(hostname)"
echo "  Date: $(date)"
echo "  EBJEPA_DSETS: $EBJEPA_DSETS"
echo ""

python3 -c "
from examples.ac_video_jepa.main import run
run(
    fname='examples/ac_video_jepa/cfgs/crafter.yaml',
    **{
        'meta.seed': $SEED,
        'model.regularizer.idm_coeff': $IDM_COEFF,
        'data.data_dir': '$EBJEPA_DSETS/crafter_trajectories',
        'logging.log_wandb': False,
    }
)
"
