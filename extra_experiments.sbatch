#!/bin/bash
#SBATCH --job-name=extra_experiments
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
# Experiment 1: CFD naive-threshold sanity check
#   Mid-position cylinder; no router. Threshold R(x) = norm(|r|+|e|) > 1.1
#   to form Omega_C, run hybrid, report single hybrid RMSE vs full CFD.
# ============================================================
# echo ""
# echo "=============================================================="
# echo "=== Exp 1: CFD naive-threshold sanity (mid cylinder)"
# echo "=============================================================="
# rm -rf ./naive_threshold_output_cfd
# python run_naive_threshold_cfd.py \
#     --pinn-path ./models/pinn_cylinder_Re100_0.5_0.5_0.1.h5 \
#     --cylinder-x 0.5 --cylinder-y 0.5 --cylinder-radius 0.1 \
#     --inlet-velocity 1.0 --Re 100 \
#     --beta 1.1 \
#     --no-morph \
#     --output-dir ./naive_threshold_output_cfd

# # ============================================================
# # Experiment 2: Poisson naive-threshold sanity check
# #   src_p1 (in-distribution); same threshold rule; single number.
# # ============================================================
# echo ""
# echo "=============================================================="
# echo "=== Exp 2: Poisson naive-threshold sanity (src_p1)"
# echo "=============================================================="
# cd /home/s/simpsone/blend2/helmholtz || exit 1
# python -c "import skfem" 2>/dev/null || pip install --no-input scikit-fem
# rm -rf ./naive_threshold_output_poisson
# python run_naive_threshold_poisson.py \
#     --config-name src_p1 \
#     --configs configs_exp3.json \
#     --beta 1.1 \
#     --mesh-n 129 --nx 201 --ny 201 \
#     --sigma 0.05 --amplitude 250.0 \
#     --hole-center-x 0.5 --hole-center-y 0.5 --hole-radius 0.15 \
#     --output-dir ./naive_threshold_output_poisson
# cd /home/s/simpsone/blend2 || exit 1

# # ============================================================
# # Experiment 3: CFD wrong-PINN with trained router
# #   Test geometry: Re=100, (x=0.55, y=0.5, r=0.1)  -- a 0.05 shift
# #       from train config 3 -- router has NOT seen this case.
# #   PINN used: train config 3 checkpoint (trained at x=0.5).
# #   Router: freshly retrained on configs_multi_cylinder.json.
# # ============================================================
# # echo ""
# # echo "=============================================================="
# # echo "=== Exp 3a: Retrain multi-config router (for wrongpinn eval)"
# # echo "=============================================================="
# # rm -rf router_output_multi
# # python train_router_multi.py \
# #     --config configs_multi_cylinder.json \
# #     --output-dir ./router_output_multi \
# #     --epochs 2000 \
# #     --beta 1.1 \
# #     --lambda-tv 1.0 \
# #     --lr 1e-3

# echo ""
# echo "=============================================================="
# echo "=== Exp 3b: CFD wrong-PINN, trained router, shifted-x test"
# echo "=============================================================="
# rm -rf ./metrics_output_wrongpinn_cfd
# python plot_coverage_metrics.py \
#     --pinn-path ./models/pinn_cylinder_Re100_0.5_0.5_0.1.h5 \
#     --router-weights ./router_output_multi/router.weights.h5 \
#     --compute-cfd \
#     --save-cfd ./cfd_wrongpinn_cfd.npz \
#     --beta 1.1 \
#     --cylinder-x 0.55 --cylinder-y 0.5 --cylinder-radius 0.1 \
#     --inlet-velocity 1.0 --Re 100 \
#     --output-dir ./metrics_output_wrongpinn_cfd

# echo "=== Metrics: Multi-Cylinder TEST (unseen config) ==="
# python plot_coverage_metrics.py \
#     --pinn-path ./models/pinn_cylinder_Re600_0.6_0.4_0.13.h5 \
#     --router-weights ./router_output_multi/router.weights.h5 \
#     --compute-cfd \
#     --save-cfd ./cfd_test_cy0.65.npz \
#     --beta 1.1 \
#     --cylinder-y 0.4 \
#     --cylinder-x 0.6 \
#     --cylinder-radius 0.13 \
#     --inlet-velocity 6.0 \
#     --Re 600 \
#     --output-dir ./metrics_output_test_cy0.65

# ============================================================
# Regenerate solution_comparison.png for the held-out test config with the
# updated plotting code (identical contour levels + one shared colorbar per
# row, so PINN | Hybrid | CFD truly use the same scale).  Reuses the CFD
# solution already saved by the previous time_multi run to avoid re-solving.
# ============================================================
echo ""
echo "=============================================================="
echo "=== Regen: solution_comparison.png for held-out test config"
echo "=============================================================="
TEST_CFD=./time_multi_runs/test_00_x0.6_y0.55_r0.11_u1.0/cfd.npz
TEST_OUT=./solution_test_rerun
rm -rf "$TEST_OUT"
if [ -f "$TEST_CFD" ]; then
    CFD_ARGS="--cfd-path $TEST_CFD"
else
    CFD_ARGS="--compute-cfd --save-cfd ./cfd_test_held_out.npz"
fi
python plot_coverage_metrics.py \
    --pinn-path ./models/pinn_cylinder_Re100_0.6_0.55_0.11.h5 \
    --router-weights ./router_output_multi/router.weights.h5 \
    $CFD_ARGS \
    --beta 1.1 \
    --cylinder-x 0.6 --cylinder-y 0.55 --cylinder-radius 0.11 \
    --inlet-velocity 1.0 --Re 100 \
    --output-dir "$TEST_OUT"

# ---- Actual-hybrid RMSE-vs-coverage sweep (10% increments) + optimal ----
echo ""
echo "=== Sweep: actual hybrid solves at 10% coverage increments (test) ==="
python time_cylinder.py \
    --pinn-path ./models/pinn_cylinder_Re100_0.6_0.55_0.11.h5 \
    --router-weights ./router_output_multi/router.weights.h5 \
    --output-dir "$TEST_OUT" \
    --beta 1.1 \
    --cylinder-x 0.6 --cylinder-y 0.55 --cylinder-radius 0.11 \
    --inlet-velocity 1.0 --Re 100 \
    --n-runs 1

echo "=== Composing rmse_vs_coverage.png (sweep + optimal star) ==="
python <<PYEOF
import os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = "$TEST_OUT"
sweep = np.load(os.path.join(out, 'timing_sweep.npz'))
metrics = np.load(os.path.join(out, 'metrics_results.npz'))

cov_sweep = sweep['coverage']
rmse_sweep = sweep['rmse']
cov_opt = float(metrics['full_loss_optimal_coverage'])
rmse_opt = float(metrics['hybrid_rmse'])
beta = float(metrics['cfg_beta']) if 'cfg_beta' in metrics.files else 1.1

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(cov_sweep * 100, rmse_sweep, 'o-', linewidth=2, markersize=6,
        color='tab:blue', label='Hybrid sweep (10% increments)')
ax.scatter([cov_opt * 100], [rmse_opt], marker='*', s=260, color='green',
           zorder=6, edgecolor='black', linewidths=1,
           label=f'Optimal (cov={cov_opt*100:.0f}%)')
ax.axhline(rmse_sweep[0], color='purple', linestyle='--', linewidth=1.2,
           alpha=0.5, label=f'PINN only: {rmse_sweep[0]:.4f}')
ax.axhline(rmse_sweep[-1], color='teal', linestyle='--', linewidth=1.2,
           alpha=0.5, label=f'CFD only: {rmse_sweep[-1]:.4f}')
ax.set_xlabel('Coverage (% solved by CFD)', fontsize=13)
ax.set_ylabel('Hybrid RMSE (vs CFD)', fontsize=13)
ax.set_title(f'RMSE vs Coverage  (held-out test, β = {beta})',
             fontsize=14, fontweight='bold')
ax.set_xlim(-5, 105)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=9)
fig.tight_layout()
out_path = os.path.join(out, 'rmse_vs_coverage.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'  Saved {out_path}')
PYEOF

echo "=== Job completed ==="
