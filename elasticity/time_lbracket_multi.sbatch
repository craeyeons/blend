#!/bin/bash
#SBATCH --job-name=time_lbracket_multi
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

cd /home/s/simpsone/blend2/elasticity || exit 1
echo "Working directory: $(pwd)"

export PATH="$HOME/miniforge3/envs/cfd_jax/bin:$PATH"
module load cuda/12.2 cudnn/8.9.5.29

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async

# ============================================================
# EDIT THESE PER SUBMISSION
#   CONFIG_JSON : JSON with train/test load configs
#   ROUTER      : multi-load router weights trained on CONFIG_JSON's train split
#   SPLIT       : 'train' / 'test' / 'both'
# ============================================================
CONFIG_JSON=./configs_multi_lbracket.json
ROUTER=./router_output/l_bracket_multi/beta_1.1/router.weights.h5
SPLIT=test
TIMING_DIR=./timing_output/l_bracket_multi

echo "Config: $CONFIG_JSON  split=$SPLIT"
echo "Router: $ROUTER"

echo "=== Timing: Multi-load L-bracket elasticity ==="
python time_elasticity_multi_lbracket.py \
    --config $CONFIG_JSON \
    --router-weights $ROUTER \
    --split $SPLIT \
    --output-dir $TIMING_DIR \
    --nx 200 --ny 200 \
    --beta 1.1 \
    --n-runs 3

echo "=== Job completed ==="
