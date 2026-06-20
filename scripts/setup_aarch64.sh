#!/bin/bash
#SBATCH --job-name=setup-arm
#SBATCH --account=vivatech-dreamingmachines
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=setup-arm-%j.out
#SBATCH --error=setup-arm-%j.err

set -e
echo "=== Node: $(hostname), Arch: $(uname -m) ==="

source ~/eb_jepa/env.sh
module load python312 2>/dev/null || true

cd ~/eb_jepa
rm -f uv.lock

echo ">>> Running uv sync..."
$UV_INSTALL_DIR/uv sync --dev
echo ">>> uv sync DONE"

echo ">>> Testing torch import..."
$UV_PROJECT_ENVIRONMENT/bin/python3 -c "
import torch
print(f'torch={torch.__version__}')
print(f'cuda={torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'gpus={torch.cuda.device_count()}')
    print(f'device={torch.cuda.get_device_name(0)}')
"
echo "=== SETUP COMPLETE ==="
