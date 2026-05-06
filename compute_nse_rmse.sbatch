#!/bin/bash
#SBATCH --job-name=nse_rmse
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres="gpu:h100-96:1"
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --export=NONE

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
nvidia-smi

cd /home/s/simpsone/blend2 || exit 1
echo "Working directory: $(pwd)"

export PATH="$HOME/miniforge3/envs/cfd_jax/bin:$PATH"

module load cuda/12.2 cudnn/8.9.5.29

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async

echo "=== Verifying JAX GPU ==="
python -c "
import jax
print('JAX devices:', jax.devices())
print('GPU available:', len(jax.devices('gpu')) > 0)
"

echo "=== Computing PINN-vs-CFD RMSE for NSE models ==="
python compute_nse_rmse.py \
    --models-dir ./models \
    --cfd-dir ./cfd_cache \
    --out rmse_summary.txt

echo "=== Done ==="
cat rmse_summary.txt
