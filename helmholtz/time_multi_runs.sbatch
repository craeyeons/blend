#!/bin/bash
#SBATCH --job-name=helm_time_multi_runs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres="gpu:h100-96:1"
#SBATCH --mem=64G
#SBATCH --time=24:00:00

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi

cd /home/s/simpsone/blend2/helmholtz || exit 1
echo "Working directory: $(pwd)"

export PATH="$HOME/miniforge3/envs/cfd_jax/bin:$PATH"
module load cuda/12.2 cudnn/8.9.5.29

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async

python -c "import skfem" 2>/dev/null || pip install --no-input scikit-fem

# ----------------------------------------------------------------------
# Train router 3x on configs_exp3.json (seeds 0,1,2), then evaluate every
# train+test config. Outputs per config:
#   ./time_multi_runs/<split>_<idx>_<name>/
#     - summary.json
#     - regions_and_errors.pdf
#     - solution_comparison.pdf
# Top level:
#   ./time_multi_runs/summary.csv (mean/std for all metrics)
#   ./time_multi_runs/summary.json
#   ./time_multi_runs/router_runs/router_helmholtz_run{0,1,2}.weights.h5
# ----------------------------------------------------------------------

CONFIGS=configs_exp3.json
OUTPUT_ROOT=./time_multi_runs

EXTRA="${EXTRA:-}"

python time_multi_runs.py \
    --configs "$CONFIGS" \
    --output-dir "$OUTPUT_ROOT" \
    --epochs 2000 \
    --beta 1.0 \
    --lambda-tv 1.0 \
    $EXTRA

echo "=== Job completed ==="
