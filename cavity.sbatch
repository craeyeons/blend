#!/bin/bash
#SBATCH --job-name=cavity_router
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres="gpu:h100-96:1"
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --export=NONE

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Go to your working directory
cd /home/s/simpsone/blend2 || exit 1
echo "Working directory: $(pwd)"

# Add conda env to PATH
export PATH="$HOME/miniforge3/envs/cfd_jax/bin:$PATH"

# Load CUDA module (required on Compute Canada clusters)
module load cuda/12.2 cudnn/8.9.5.29

# Set CUDA environment variables for JAX
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# TensorFlow GPU memory settings to avoid cuDNN errors
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async

echo "=== Checking CUDA version ==="
nvcc --version 2>/dev/null || echo "nvcc not found, using nvidia-smi info:"
nvidia-smi | grep "CUDA Version"

# Verify GPU is now available
echo "=== Verifying JAX GPU support ==="
python -c "
import jax
print('JAX version:', jax.__version__)
print('JAX devices:', jax.devices())
print('GPU available:', len(jax.devices('gpu')) > 0)
print('Default backend:', jax.default_backend())
"

rm -rf router_output_cavity threshold_plots_cavity

# Train router for cavity flow
echo "=== Training Router for Cavity Flow ==="
python train_router_cavity.py \
    --model-path ./models/pinn_cavity_flow.h5 \
    --output-dir ./router_output_cavity \
    --epochs 2000 \
    --beta 1.1 \
    --lambda-tv 1.0 \
    --lr 1e-3 \
    --weight-continuity 1.0 \
    --weight-momentum 1.0 \
    --N 100 \

echo "=== Router training completed ==="

python plot_router_thresholds_cavity.py \
    --router-path ./router_output_cavity/router.weights.h5 \
    --pinn-path ./models/pinn_cavity_flow.h5 \
    --threshold-start 0.1 \
    --threshold-end 0.9 \
    --num-thresholds 9 \
    --N 100

echo "=== Job completed ==="
