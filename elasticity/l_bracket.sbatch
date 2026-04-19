#!/bin/bash
#SBATCH --job-name=elast_lbrac
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

echo "=== Verifying GPU ==="
python -c "
import tensorflow as tf
print('TF:', tf.__version__, 'GPU:', tf.config.list_physical_devices('GPU'))
import jax
print('JAX:', jax.__version__, 'devices:', jax.devices())
"

# --- Step 1: Train PINN ---
echo "=== Training PINN: l_bracket ==="
python train_pinn.py \
    --problem l_bracket \
    --epochs 10000 \
    --lr 1e-3 \
    --n-domain 10000 \
    ----w-tracn-boundary 2000 \
     5.0

# --- Step 2: FDM ground truth ---
echo "=== Running FDM solver ==="
python run_fdm.py \
    --problem l_bracket \
    --nx 200 --ny 200 \
    --max-iter 200000

# --- Step 3: Train router ---
echo "=== Training Router ==="
python train_router.py \
    --problem l_bracket \
        --model-path ./models/pinn_l_bracket.weights.h5 \
    --epochs 2000 \
    --beta 1.1 \
    --lambda-tv 1.0 \
    --lr 1e-3 \
    --nx 200 --ny 200 \
    --x-min 0.0 --y-min 0.0

# --- Step 4: Threshold plots ---
echo "=== Threshold Plots ==="
python plot_router_thresholds.py \
    --problem l_bracket \
    --router-path ./router_output/l_bracket/beta_1.1/router.weights.h5 \
    --pinn-path ./models/pinn_l_bracket.weights.h5 \
    --output-dir ./threshold_plots_lbracket \
    --threshold-start -0.5 \
    --threshold-end 0.5 \
    --num-thresholds 11

# --- Step 5: Coverage metrics ---
echo "=== Coverage Metrics ==="
python plot_coverage_metrics.py \
    --problem l_bracket \
    --pinn-path ./models/pinn_l_bracket.weights.h5 \
    --router-path ./router_output/l_bracket/beta_1.1/router.weights.h5 \
    --fdm-path ./results/fdm_l_bracket.npz \
    --beta 1.1 \
    --output-dir ./metrics_output_lbracket

echo "=== Job completed ==="
