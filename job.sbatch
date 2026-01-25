#!/bin/bash
#SBATCH --job-name=cfd_sim
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres="gpu:h100-96:1"
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --export=NONE

# Add conda env to PATH
export PATH="$HOME/miniforge3/envs/cfd_jax/bin:$PATH"

# Prevent TensorFlow from hogging GPU memory
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async

# JAX CUDA settings
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Go to working directory
cd /home/s/simpsone/blend2

# Run simulation
#python main.py \
#    --simulation cylinder \
#    --mode dynamic \
#    --model-path ./models/pinn_cylinder_100.0.h5 \
#    --complexity-threshold 0.6 \
#    --merge-distance 1 \
#    --weight-strain 0.05 \
#    --weight-vorticity 0.2 \
#    --weight-momentum 2 \
#    --weight-continuity 0.5 \
#    --normalization percentile

python router_train.py --Re 100 --epochs 200
