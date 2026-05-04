#!/bin/bash
#SBATCH --job-name=time_multi_cyl
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

# ----------------------------------------------------------------------
# Loop over every config in configs_multi_cylinder.json (train + test).
# Each config has its own per-geometry PINN; all configs share the same
# multi-router weights.
#
# compute_summary_table.py invokes plot_coverage_metrics.py per config to
# produce per-config plots + npz, then aggregates everything into one
# summary_table.csv and summary_table.tex (drop-in for Table 2).
#
# Per-config outputs land in:
#   ./time_multi_runs/<role>_<idx>_x..._y..._r..._u.../
#     - solution_comparison.png   (PINN | Hybrid | CFD, titled, closed PINN contour)
#     - coverage_curve.png        (RMSE vs coverage)
#     - loss_vs_coverage.png      (training loss vs coverage, optimum starred)
#     - metrics_results.npz       (numerical results: PINN/Hybrid RMSE, coverage, times)
#     - cfd.npz                   (CFD ground-truth fields)
# ----------------------------------------------------------------------

CONFIGS=configs_multi_cylinder.json
ROUTER=./router_output_multi/router.weights.h5
OUTPUT_ROOT=./time_multi_runs

rm -rf "$OUTPUT_ROOT"

echo "=== Running coverage metrics across all configs (train + test) ==="
python compute_summary_table.py \
    --configs "$CONFIGS" \
    --router-weights "$ROUTER" \
    --output-root "$OUTPUT_ROOT" \
    --beta 1.1

echo "=== Job completed ==="
echo "Summary table: ${OUTPUT_ROOT}/summary_table.csv"
echo "LaTeX: ${OUTPUT_ROOT}/summary_table.tex"
