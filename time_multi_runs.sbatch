#!/bin/bash
#SBATCH --job-name=time_multi_runs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres="gpu:h100-96:1"
#SBATCH --mem=32G
#SBATCH --time=12:00:00
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

# ----------------------------------------------------------------------
# Coverage sweep for the test config at index 0
# (test_0_x0.6_y0.55_r0.11_u1.0). For each of the 3 trained routers
# (run_{0,1,2}/) sweeps the CFD coverage in 10% increments (0%..100%),
# recording the raw (RMSE, hybrid time) at every level. Both axes vary
# per seed (cell placement affects solver conditioning), so all raw
# per-seed points are saved for downstream replotting/aggregation.
# Output:
#   ./time_multi_runs_accuracy_time/test_0_..._sweep/
#     - sweep.csv   (columns: seed,target_coverage,actual_coverage,
#                    rmse,hybrid_time,cfd_time_mean)
#     - sweep.json
#     - sweep.pdf   (RMSE vs hybrid time, one line per seed)
# Routers are trained first if router_runs/run_*/ don't exist yet
# (pass EXTRA="--skip-train" to skip, or "--retrain" to force).
# ----------------------------------------------------------------------

CONFIGS=configs_multi_cylinder.json
OUTPUT_ROOT=./time_multi_runs_accuracy_time

# Pass --retrain to force retraining; default reuses any existing router_runs/run_*/
EXTRA="${EXTRA:-}"

python time_multi_runs.py \
    --config "$CONFIGS" \
    --output-dir "$OUTPUT_ROOT" \
    --epochs 500 \
    --beta 1.1 \
    --lambda-tv 1.0 \
    --Re 100 \
    --sweep \
    --sweep-role test \
    --sweep-idx 0 \
    --sweep-step 0.1 \
    --sweep-extra-coverages 0.46 \
    --sweep-seeds 0 1 \
    $EXTRA

echo "=== Job completed ==="
