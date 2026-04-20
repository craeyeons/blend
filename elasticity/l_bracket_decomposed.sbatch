#!/bin/bash
#SBATCH --job-name=elast_lbrac_decomp
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
# EDIT THESE PER SUBMISSION to change the applied load.
#   LOAD_EDGE      : 'top'  (V-bar top edge)   or  'right' (H-bar right edge)
#   LOAD_MAGNITUDE : traction magnitude (e.g. 10.0)
#   LOAD_ANGLE     : direction in degrees, CCW from +x
#                    (e.g. 180 = pull in -x, -90 = pull in -y)
# ============================================================
LOAD_EDGE=top
LOAD_MAGNITUDE=10.0
LOAD_ANGLE=180.0

# Encode load config in a short, filesystem-safe tag.
TAG="${LOAD_EDGE}_m${LOAD_MAGNITUDE}_a${LOAD_ANGLE}"
TAG="${TAG//./p}"   # replace '.' with 'p' so filenames are cleaner
TAG="${TAG//-/n}"   # replace leading '-' in angle with 'n'

VBAR=./models/pinn_l_bracket_vbar_${TAG}.weights.h5
HBAR=./models/pinn_l_bracket_hbar_${TAG}.weights.h5
FDM_NPZ=./results/fdm_l_bracket_${TAG}.npz
ROUTER_DIR=./router_output/l_bracket/${TAG}/beta_1.1
METRICS_DIR=./metrics_output_lbracket/${TAG}
THRESH_DIR=./threshold_plots_lbracket/${TAG}

echo "Load config: edge=$LOAD_EDGE  magnitude=$LOAD_MAGNITUDE  angle=$LOAD_ANGLE  tag=$TAG"

echo "=== Verifying GPU ==="
python -c "
import tensorflow as tf
print('TF:', tf.__version__, 'GPU:', tf.config.list_physical_devices('GPU'))
import jax
print('JAX:', jax.__version__, 'devices:', jax.devices())
"

# --- Step 1: Train decomposed PINN ---
echo "=== Training Decomposed PINN: l_bracket ==="
python train_pinn_decomposed.py \
    --epochs 10000 \
    --lr 3e-4 \
    --n-domain 10000 \
    --n-boundary 2000 \
    --n-coupling 5000 \
    --n-interface 2000 \
    --w-coupling 10.0 \
    --w-trac-match 10.0 \
    --load-edge $LOAD_EDGE \
    --load-magnitude $LOAD_MAGNITUDE \
    --load-angle $LOAD_ANGLE \
    --tag $TAG

# --- Step 2: FDM ground truth ---
echo "=== Running FDM solver ==="
python run_fdm.py \
    --problem l_bracket \
    --nx 200 --ny 200 \
    --max-iter 200000 \
    --load-edge $LOAD_EDGE \
    --load-magnitude $LOAD_MAGNITUDE \
    --load-angle $LOAD_ANGLE \
    --tag $TAG

# --- Step 3: Train router ---
echo "=== Training Router ==="
python train_router.py \
    --problem l_bracket \
    --model-path dummy \
    --pinn-vbar-path $VBAR \
    --pinn-hbar-path $HBAR \
    --epochs 2000 \
    --beta 1.1 \
    --lambda-tv 1.0 \
    --lr 1e-3 \
    --nx 200 --ny 200 \
    --x-min 0.0 --y-min 0.0 \
    --load-edge $LOAD_EDGE \
    --load-magnitude $LOAD_MAGNITUDE \
    --load-angle $LOAD_ANGLE \
    --output-dir $ROUTER_DIR

# --- Step 4: Threshold plots ---
echo "=== Threshold Plots ==="
python plot_router_thresholds.py \
    --problem l_bracket \
    --router-path ${ROUTER_DIR}/router.weights.h5 \
    --pinn-vbar-path $VBAR \
    --pinn-hbar-path $HBAR \
    --output-dir $THRESH_DIR \
    --load-edge $LOAD_EDGE \
    --load-magnitude $LOAD_MAGNITUDE \
    --load-angle $LOAD_ANGLE \
    --threshold-start -0.5 \
    --threshold-end 0.5 \
    --num-thresholds 11

# --- Step 5: Coverage metrics ---
echo "=== Coverage Metrics ==="
python plot_coverage_metrics.py \
    --problem l_bracket \
    --pinn-vbar-path $VBAR \
    --pinn-hbar-path $HBAR \
    --router-path ${ROUTER_DIR}/router.weights.h5 \
    --fdm-path $FDM_NPZ \
    --beta 1.1 \
    --load-edge $LOAD_EDGE \
    --load-magnitude $LOAD_MAGNITUDE \
    --load-angle $LOAD_ANGLE \
    --output-dir $METRICS_DIR

echo "=== Job completed ==="
