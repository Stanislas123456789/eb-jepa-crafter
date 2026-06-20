#!/bin/bash
#SBATCH --job-name=crafter-eval
#SBATCH --account=vivatech-dreamingmachines
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH --gres=gpu:b200:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --reservation=Vivatech
#SBATCH --output=eval-%j.out
#SBATCH --error=eval-%j.err

set -e
source ~/eb_jepa/env.sh
module load python312 2>/dev/null || true
source $UV_PROJECT_ENVIRONMENT/bin/activate
cd ~/eb_jepa

DSETS=$EBJEPA_DSETS/crafter_trajectories
CKPTS=$EBJEPA_CKPTS/ac_video_jepa

echo "=== Crafter JEPA Evaluation ==="
echo "  Node: $(hostname), Arch: $(uname -m)"
echo "  Date: $(date)"
echo ""

# Find all checkpoint dirs
FULL_CKPT=$(find $CKPTS -path "*idm1_seed1/latest.pth.tar" | head -1)
ABLATED_CKPT=$(find $CKPTS -path "*idm0_seed1/latest.pth.tar" | head -1)
FULL_CKPT2=$(find $CKPTS -path "*idm1_seed1000/latest.pth.tar" | head -1)

echo "Full model (seed=1): $FULL_CKPT"
echo "Ablated (idm=0): $ABLATED_CKPT"
echo "Full model (seed=1000): $FULL_CKPT2"
echo ""

EVAL_DIR=eval_results
mkdir -p $EVAL_DIR

# 1. Probe evaluation on full model
if [ -n "$FULL_CKPT" ]; then
    echo ">>> Running probe eval on full model..."
    python3 scripts/eval_probe.py \
        --checkpoint_path "$FULL_CKPT" \
        --data_dir "$DSETS" \
        --output_dir "$EVAL_DIR/probe_full" \
        --epochs 10 2>&1 | tail -30
    echo ""
fi

# 2. Probe evaluation on ablated model
if [ -n "$ABLATED_CKPT" ]; then
    echo ">>> Running probe eval on ablated model..."
    python3 scripts/eval_probe.py \
        --checkpoint_path "$ABLATED_CKPT" \
        --data_dir "$DSETS" \
        --output_dir "$EVAL_DIR/probe_ablated" \
        --epochs 10 2>&1 | tail -30
    echo ""
fi

# 3. Rollout evaluation on full model
if [ -n "$FULL_CKPT" ]; then
    echo ">>> Running rollout eval on full model..."
    python3 scripts/eval_rollout.py \
        --checkpoint_path "$FULL_CKPT" \
        --data_dir "$DSETS" \
        --output_dir "$EVAL_DIR/rollout_full" \
        --num_batches 50 2>&1 | tail -20
    echo ""
fi

# 4. Rollout evaluation on ablated model
if [ -n "$ABLATED_CKPT" ]; then
    echo ">>> Running rollout eval on ablated model..."
    python3 scripts/eval_rollout.py \
        --checkpoint_path "$ABLATED_CKPT" \
        --data_dir "$DSETS" \
        --output_dir "$EVAL_DIR/rollout_ablated" \
        --num_batches 50 2>&1 | tail -20
    echo ""
fi

# 5. Generate figures
echo ">>> Generating figures..."
python3 scripts/make_figures.py \
    --results_dir "$EVAL_DIR" \
    --output_dir "$EVAL_DIR/figures" 2>&1 | tail -10

echo ""
echo "=== ALL EVALUATIONS COMPLETE ==="
echo "  Results in: $EVAL_DIR/"
ls -la $EVAL_DIR/
