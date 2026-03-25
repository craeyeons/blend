#!/bin/bash
#SBATCH --job-name=metrics_plate
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
nvidia-smi

cd /home/s/simpsone/blend2/elasticity || exit 1

export PATH="$HOME/miniforge3/envs/cfd_jax/bin:$PATH"
module load cuda/12.2 cudnn/8.9.5.29

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async

rm -rf metrics_output_plate

python plot_coverage_metrics.py \
    --problem plate_with_hole \
    --pinn-path ./models/pinn_plate_with_hole.h5 \
    --router-path ./router_output/plate_with_hole/beta_1.1/router.weights.h5 \
    --fdm-path ./results/fdm_plate_with_hole.npz \
    --beta 1.1 \
    --output-dir ./metrics_output_plate

echo "=== Metrics done ==="
