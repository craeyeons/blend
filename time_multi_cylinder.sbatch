#!/bin/bash
#SBATCH --job-name=time_multi_cyl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres="gpu:h100-96:1"
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --export=NONE

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
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

echo "=== Timing: Multi-Cylinder TEST (unseen config) ==="
python time_cylinder.py \
    --pinn-path ./models/pinn_cylinder_Re600_0.6_0.4_0.13.h5 \
    --router-weights ./router_output_multi/router.weights.h5 \
    --beta 1 \
    --cylinder-y 0.4 \
    --cylinder-x 0.6 \
    --cylinder-radius 0.13 \
    --temperature 0.5 \
    --n-runs 3

echo "=== Job completed ==="
