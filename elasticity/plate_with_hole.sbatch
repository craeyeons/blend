#!/bin/bash
#SBATCH --job-name=elast_plate
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
echo "=== Training PINN: plate_with_hole ==="
python train_pinn.py \
    --problem plate_with_hole \
    --epochs 10000 \
    --lr 1e-3 \
    --n-domain 10000 \
    --n-boundary 2000

# --- Step 2: FDM ground truth ---
echo "=== Running FDM solver ==="
python run_fdm.py \
    --problem plate_with_hole \
    --nx 200 --ny 200 \
    --max-iter 200000

# --- Step 3: Train router ---
echo "=== Training Router ==="
python train_router.py \
    --problem plate_with_hole \
        --model-path ./models/pinn_plate_with_hole.weights.h5 \
    --epochs 2000 \
    --beta 1.1 \
    --lambda-tv 1.0 \
    --lr 1e-3 \
    --nx 200 --ny 200

# --- Step 4: Threshold plots ---
echo "=== Threshold Plots ==="
python plot_router_thresholds.py \
    --problem plate_with_hole \
    --router-path ./router_output/plate_with_hole/beta_1.1/router.weights.h5 \
    --pinn-path ./models/pinn_plate_with_hole.weights.h5 \
    --output-dir ./threshold_plots_plate \
    --threshold-start 0.1 \
    --threshold-end 0.9 \
    --num-thresholds 9

# --- Step 5: Coverage metrics ---
echo "=== Coverage Metrics ==="
python plot_coverage_metrics.py \
    --problem plate_with_hole \
    --pinn-path ./models/pinn_plate_with_hole.weights.h5 \
    --router-path ./router_output/plate_with_hole/beta_1.1/router.weights.h5 \
    --fdm-path ./results/fdm_plate_with_hole.npz \
    --beta 1.1 \
    --output-dir ./metrics_output_plate

echo "=== Job completed ==="
