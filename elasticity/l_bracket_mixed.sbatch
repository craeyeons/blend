#!/bin/bash
#SBATCH --job-name=elast_lbrac_mixed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres="gpu:h100-96:1"
#SBATCH --mem=32G
#SBATCH --time=4:00:00
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

echo "=== Verifying GPU ==="
python -c "
import tensorflow as tf
print('TF:', tf.__version__, 'GPU:', tf.config.list_physical_devices('GPU'))
"

# --- Step 1: Train mixed-variable PINN ---
echo "=== Training Mixed-Variable PINN: l_bracket ==="
python train_pinn_mixed.py \
    --problem l_bracket \
    --epochs 20000 \
    --lr 1e-4 \
    --batch-size 256 \
    --n-domain 40000 \
    --n-boundary 2000 \
    --dist-epochs 2000 \
    --part-epochs 2000 \
    --applied-stress 10.0

# --- Step 2: FDM ground truth (if not already done) ---
echo "=== Running FDM solver ==="
python run_fdm.py \
    --problem l_bracket \
    --nx 200 --ny 200 \
    --max-iter 200000

echo "=== Job completed ==="
