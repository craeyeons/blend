#!/bin/bash
#SBATCH --job-name=time_lbracket
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
# EDIT THESE PER SUBMISSION to match the trained load config.
# ============================================================
LOAD_EDGE=right
LOAD_MAGNITUDE=8.0
LOAD_ANGLE=45.0

TAG="${LOAD_EDGE}_m${LOAD_MAGNITUDE}_a${LOAD_ANGLE}"
TAG="${TAG//./p}"
TAG="${TAG//-/n}"

VBAR=./models/pinn_l_bracket_vbar_${TAG}.weights.h5
HBAR=./models/pinn_l_bracket_hbar_${TAG}.weights.h5
ROUTER=./router_output/l_bracket/${TAG}/beta_1.1/router.weights.h5
TIMING_DIR=./timing_output/l_bracket/${TAG}

echo "Load config: edge=$LOAD_EDGE  magnitude=$LOAD_MAGNITUDE  angle=$LOAD_ANGLE  tag=$TAG"

echo "=== Timing: L-Bracket Elasticity (decomposed PINN) ==="
python time_elasticity.py \
    --problem l_bracket \
    --pinn-path dummy \
    --pinn-vbar-path $VBAR \
    --pinn-hbar-path $HBAR \
    --router-weights $ROUTER \
    --beta 1.1 \
    --nx 200 --ny 200 \
    --x-min 0.0 --y-min 0.0 \
    --load-edge $LOAD_EDGE \
    --load-magnitude $LOAD_MAGNITUDE \
    --load-angle $LOAD_ANGLE \
    --output-dir $TIMING_DIR \
    --n-runs 3

echo "=== Job completed ==="
