#!/bin/bash
#SBATCH --job-name=test-torch
#SBATCH --account=vivatech-dreamingmachines
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH --gres=gpu:b200:1
#SBATCH --time=00:05:00
#SBATCH --reservation=Vivatech
#SBATCH --output=test-torch-%j.out
#SBATCH --error=test-torch-%j.err

module load python312 2>/dev/null || true

WORKDIR=/lustre/work/vivatech-dreamingmachines/smichel
TORCH_LIB=$WORKDIR/venv/.venv/lib/python3.12/site-packages/torch/lib
NVIDIA_LIB=$WORKDIR/venv/.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$TORCH_LIB:$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cublas/lib:${LD_LIBRARY_PATH:-}

echo "=== Node: $(hostname) ==="
echo "=== nvidia-smi ==="
nvidia-smi 2>&1 | head -10
echo "=== LD_LIBRARY_PATH ==="
echo $LD_LIBRARY_PATH | tr ':' '\n' | head -10
echo "=== ldd on libtorch_global_deps.so ==="
ldd $TORCH_LIB/libtorch_global_deps.so 2>&1
echo "=== strace dlopen attempt ==="
strace -e trace=open,openat $WORKDIR/venv/.venv/bin/python3 -c "import ctypes; ctypes.CDLL('$TORCH_LIB/libtorch_global_deps.so')" 2>&1 | tail -20
echo "=== direct python torch import ==="
$WORKDIR/venv/.venv/bin/python3 -c "import torch; print(f'SUCCESS: torch {torch.__version__}, cuda={torch.cuda.is_available()}, gpus={torch.cuda.device_count()}')" 2>&1
