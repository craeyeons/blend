#!/bin/bash
#SBATCH --job-name=multi_router
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

rm -rf router_output_multi

# Train router on multiple cylinder configs
echo "=== Training Multi-Config Router ==="
python train_router_multi.py \
    --config configs_multi_cylinder.json \
    --output-dir ./router_output_multi \
    --epochs 2000 \
    --beta 1.1 \
    --lambda-tv 1.0 \
    --lr 1e-3 \

echo "=== Multi-config router training completed ==="

# Plot thresholds for each test config using the trained router
echo "=== Plotting thresholds ==="
python plot_router_thresholds.py \
    --router-path ./router_output_multi/router.weights.h5 \
    --threshold-start 0.45 \
    --threshold-end 0.75 \
    --num-thresholds 11 \
    --cylinder-y 0.5 \
    --output-dir ./threshold_plots_multi

rm -rf metrics_output_multi

echo "=== Metrics: Multi-Cylinder TEST (unseen config) ==="
python plot_coverage_metrics.py \
    --pinn-path ./models/pinn_cylinder_Re600_0.6_0.4_0.13.h5 \
    --router-weights ./router_output_multi/router.weights.h5 \
    --compute-cfd \
    --save-cfd ./cfd_multi_test1.npz \
    --beta 1.1 \
    --cylinder-y 0.4 \
    --cylinder-x 0.6 \
    --cylinder-radius 0.13 \
    --output-dir ./metrics_output_multi/test_cy0.65 \
    --threshold 0

echo "=== Job completed ==="
