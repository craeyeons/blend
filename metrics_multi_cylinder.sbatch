#!/bin/bash
#SBATCH --job-name=metrics_multi
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

echo "=== Verifying JAX GPU support ==="
python -c "
import jax
print('JAX version:', jax.__version__)
print('JAX devices:', jax.devices())
print('GPU available:', len(jax.devices('gpu')) > 0)
print('Default backend:', jax.default_backend())
"

rm -rf metrics_output_multi

# ============================================================================
# Evaluate router on TRAINING configs (sanity check)
# ============================================================================
echo "=== Metrics: Training config 1 (cy=0.5) ==="
python plot_coverage_metrics.py \
    --pinn-path ./models/pinn_cylinder_Re100_0.5_0.5_0.1.h5 \
    --router-weights ./router_output_multi/router.weights.h5 \
    --compute-cfd \
    --save-cfd ./cfd_multi_train1.npz \
    --beta 1 \
    --cylinder-y 0.5 \
    --temperature 0.5 \
    --output-dir ./metrics_output_multi/train_cy0.5

echo "=== Metrics: Training config 2 (cy=0.8) ==="
python plot_coverage_metrics.py \
    --pinn-path ./models/pinn_cylinder_Re100_0.5_0.8_0.1.h5 \
    --router-weights ./router_output_multi/router.weights.h5 \
    --compute-cfd \
    --save-cfd ./cfd_multi_train2.npz \
    --beta 1 \
    --cylinder-y 0.8 \
    --temperature 0.5 \
    --output-dir ./metrics_output_multi/train_cy0.8

# ============================================================================
# Evaluate router on TEST config (unseen during training)
# ============================================================================
echo "=== Metrics: TEST config (cy=0.65, UNSEEN) ==="
python plot_coverage_metrics.py \
    --pinn-path ./models/pinn_cylinder_Re100_0.5_0.65_0.1.h5 \
    --router-weights ./router_output_multi/router.weights.h5 \
    --compute-cfd \
    --save-cfd ./cfd_multi_test1.npz \
    --beta 1 \
    --cylinder-y 0.65 \
    --temperature 0.5 \
    --output-dir ./metrics_output_multi/test_cy0.65

echo "=== All metrics generated ==="
echo "=== Job completed ==="
