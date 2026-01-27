#!/bin/bash
#SBATCH --job-name=router_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
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

# Set environment variables
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "=== Checking CUDA version ==="
nvcc --version 2>/dev/null || echo "nvcc not found, using nvidia-smi info:"
nvidia-smi | grep "CUDA Version"

# Verify TensorFlow GPU support
echo "=== Verifying TensorFlow GPU support ==="
python -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
print('GPU devices:', tf.config.list_physical_devices('GPU'))
print('GPU available:', len(tf.config.list_physical_devices('GPU')) > 0)
"

# Train router
echo "=== Training Router ==="
python train_router.py \
    --model-path ./models/pinn_cylinder_100.0.h5 \
    --output-dir ./router_output \
    --epochs 500 \
    --beta 0.1 \
    --lambda-tv 0.01 \
    --lr 1e-4 \
    --weight-continuity 1.0 \
    --weight-momentum 1.0 \
    --nx 200 \
    --ny 100 \
    --threshold 0.5

echo "=== Router training completed ==="

# Run hybrid simulation with trained router
echo "=== Running Hybrid Simulation ==="
python run_router_hybrid.py \
    --router-path ./router_output/router_weights.h5 \
    --pinn-path ./models/pinn_cylinder_100.0.h5 \
    --output-dir ./router_hybrid_output \
    --Re 100 \
    --max-iter 100000

echo "=== Job completed ==="
