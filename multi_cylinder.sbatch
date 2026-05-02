#!/bin/bash
#SBATCH --job-name=multi_router
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

# Go to your working directory
cd /home/s/simpsone/blend2 || exit 1
echo "Working directory: $(pwd)"

# Add conda env to PATH
export PATH="$HOME/miniforge3/envs/cfd_jax/bin:$PATH"

# Load CUDA module (required on Compute Canada clusters)
module load cuda/12.2 cudnn/8.9.5.29

# Set CUDA environment variables for JAX
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# TensorFlow GPU memory settings to avoid cuDNN errors
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async

echo "=== Checking CUDA version ==="
nvcc --version 2>/dev/null || echo "nvcc not found, using nvidia-smi info:"
nvidia-smi | grep "CUDA Version"

echo "=== Verifying JAX GPU support ==="
python -c "
import jax
print('JAX version:', jax.__version__)
print('JAX devices:', jax.devices())
print('GPU available:', len(jax.devices('gpu')) > 0)
print('Default backend:', jax.default_backend())
"

rm -rf router_output_multi

# Train router on multiple cylinder configs
echo "=== Training Multi-Config Router ==="
python train_router_multi.py \
    --config configs_multi_cylinder.json \
    --output-dir ./router_output_multi \
    --epochs 2000 \
    --beta 1.1 \
    --lambda-tv 1.0 \
    --lr 1e-3 \

echo "=== Multi-config router training completed ==="

# Plot thresholds at the in-distribution test geometry (matches test #1).
# Note: plot_router_thresholds.py takes --nu (not --Re); nu=0.01 == Re=100.
echo "=== Plotting thresholds ==="
python plot_router_thresholds.py \
    --router-path ./router_output_multi/router.weights.h5 \
    --pinn-path ./models/pinn_cylinder_Re100_0.6_0.55_0.11.h5 \
    --threshold-start 0.45 \
    --threshold-end 0.75 \
    --num-thresholds 11 \
    --cylinder-x 0.6 \
    --cylinder-y 0.55 \
    --cylinder-radius 0.11 \
    --inlet-velocity 1.0 \
    --nu 0.01 \
    --output-dir ./threshold_plots_multi

rm -rf metrics_output_multi

# ----------------------------------------------------------------------
# TEST #1: in-distribution geometry, matched PINN.
#   Test (x=0.6, y=0.55, r=0.11) is a true OOD-geometry interpolation:
#   each axis is bracketed by train configs, but this exact triple is
#   unseen by the router. PINN was trained at this geometry/inlet.
# ----------------------------------------------------------------------
echo "=== Metrics: TEST 1 — OOD geometry, matched PINN ==="
python plot_coverage_metrics.py \
    --pinn-path ./models/pinn_cylinder_Re100_0.6_0.55_0.11.h5 \
    --router-weights ./router_output_multi/router.weights.h5 \
    --compute-cfd \
    --save-cfd ./cfd_multi_test1.npz \
    --beta 1.1 \
    --cylinder-x 0.6 \
    --cylinder-y 0.55 \
    --cylinder-radius 0.11 \
    --inlet-velocity 1.0 \
    --Re 100 \
    --output-dir ./metrics_output_multi/test_ood

# ----------------------------------------------------------------------
# TEST #2: wrong-PINN reuse — same OOD geometry, faster inlet.
#   PINN was trained at inlet=1.0; eval runs at inlet=1.5 (dimensional
#   Re = u*L/nu = 1.5 with nu=0.01, so --Re 150 --inlet 1.5 means same
#   fluid as training, just faster flow). The PINN's velocities are
#   off by ~1.5x and pressures by ~2.25x, so the router has to reject
#   it. Mirrors Poisson's wrongpinn_shifted_src.
# ----------------------------------------------------------------------
echo "=== Metrics: TEST 2 — wrong-PINN reuse (inlet 1.0-trained, 1.5 eval) ==="
python plot_coverage_metrics.py \
    --pinn-path ./models/pinn_cylinder_Re100_0.6_0.55_0.11.h5 \
    --router-weights ./router_output_multi/router.weights.h5 \
    --compute-cfd \
    --save-cfd ./cfd_multi_test2.npz \
    --beta 1.1 \
    --cylinder-x 0.6 \
    --cylinder-y 0.55 \
    --cylinder-radius 0.11 \
    --inlet-velocity 1.5 \
    --Re 150 \
    --output-dir ./metrics_output_multi/test_wrongpinn

echo "=== Job completed ==="
