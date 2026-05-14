#!/bin/bash
#SBATCH --job-name=test_errorbars
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres="gpu:h100-96:1"
#SBATCH --mem=32G
#SBATCH --time=6:00:00
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

echo "=== Verifying GPU support ==="
python -c "
import jax, tensorflow as tf
print('JAX:', jax.__version__, 'devices:', jax.devices())
print('TF:', tf.__version__, 'GPU:', tf.config.list_physical_devices('GPU'))
"

# ============================================================
# Held-out NSE test cylinder (0.60, 0.55, 0.11) — replot
# Abstention True Loss vs Coverage  and  Hybrid RMSE vs Coverage
# with error bars across the 3 router seeds trained by
# time_multi_runs.sbatch (./time_multi_runs/router_runs/run_{0,1,2}/).
#
# - No titles on either plot
# - Green star at optimal coverage (mean across seeds, with std bars)
# - Both figures share figsize / fonts / colors / style
# ============================================================

OUTPUT_DIR=./test_errorbars_output
mkdir -p "$OUTPUT_DIR"

# Reuse the CFD reference cached by time_multi_runs if it exists.
TEST_CFD=./time_multi_runs/test_00_x0.6_y0.55_r0.11_u1.0/cfd.npz
if [ -f "$TEST_CFD" ]; then
    CFD_ARG="--cfd-path $TEST_CFD"
else
    CFD_ARG=""
fi

python plot_test_errorbars.py \
    --pinn-path ./models/pinn_cylinder_Re100_0.6_0.55_0.11.h5 \
    --router-weights \
        ./time_multi_runs/router_runs/run_0/router.weights.h5 \
        ./time_multi_runs/router_runs/run_1/router.weights.h5 \
        ./time_multi_runs/router_runs/run_2/router.weights.h5 \
    --output-dir "$OUTPUT_DIR" \
    --beta 1.1 \
    --lambda-tv 1.0 \
    --cylinder-x 0.60 --cylinder-y 0.55 --cylinder-radius 0.11 \
    --inlet-velocity 1.0 --Re 100 \
    $CFD_ARG

echo "=== Job completed ==="
