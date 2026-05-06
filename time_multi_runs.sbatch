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
# Train the multi router 3x (seeds 0,1,2) and evaluate every config in
# configs_multi_cylinder.json. Outputs per config:
#   ./time_multi_runs/<split>_<idx>_x..._y..._r..._u.../
#     - summary.json
#     - regions_and_errors.pdf
#     - solution_comparison.pdf
# Top level:
#   ./time_multi_runs/summary.csv  (mean/std for all metrics)
#   ./time_multi_runs/summary.json
#   ./time_multi_runs/router_runs/run_{0,1,2}/  (trained router weights)
# ----------------------------------------------------------------------

CONFIGS=configs_multi_cylinder.json
OUTPUT_ROOT=./time_multi_runs

# Pass --retrain to force retraining; default reuses any existing router_runs/run_*/
EXTRA="${EXTRA:-}"

python time_multi_runs.py \
    --config "$CONFIGS" \
    --output-dir "$OUTPUT_ROOT" \
    --epochs 500 \
    --beta 1.1 \
    --lambda-tv 0.1 \
    --Re 100 \
    $EXTRA

echo "=== Job completed ==="
