"""
Plot hybrid PINN+FEM results for Helmholtz Experiment 2.

Reads:
    {timing_dir}/timing_{tag}.json
    {timing_dir}/reference_{tag}.npz

Produces:
    {plots_dir}/hybrid_{tag}.png  (3 subplots)
    {plots_dir}/metrics_exp2_{tag}.json

Usage:
    python plot_hybrid.py --tag exp2_hole_k4pi
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from lib.hybrid import solve_hybrid_schwarz, rel_l2
from lib.fem_solver import HelmholtzSolver
from lib.network import build_pinn
from lib.router import RouterCNN, create_router_input
from lib.domains import create_square_with_hole, gaussian_source


def _rebuild_hybrid_solution(args, summary, ref):
    """Re-run the hybrid solve at threshold=0 to get u_hybrid for plotting."""
    import tensorflow as tf

    X, Y = ref['X'], ref['Y']
    layout = ref['layout']
    u_fem = ref['u_fem']
    logits = ref['logits']
    pinn_u = ref['pinn_u']
    residual = ref['residual']
    f_grid = ref['f_grid']

    cx = float(summary.get('hole_center_x', 0.5))
    cy = float(summary.get('hole_center_y', 0.5))
    r_h = float(summary.get('hole_radius', 0.15))
    xs_ = float(summary['x_s']); ys_ = float(summary['y_s'])
    sigma = float(summary.get('sigma', 0.05))
    amp = float(summary.get('amplitude', 1.0))
    k = float(summary['k'])

    def dirichlet_predicate(x, y):
        return (x - cx) ** 2 + (y - cy) ** 2 <= r_h ** 2

    def f_callable(x, y):
        r2 = (x - xs_) ** 2 + (y - ys_) ** 2
        return amp * np.exp(-r2 / (2.0 * sigma ** 2))

    def g_callable(x, y):
        return np.zeros_like(x)

    solver = HelmholtzSolver(k=k, mesh_n=args.mesh_n,
                             dirichlet_predicate=dirichlet_predicate)

    # Load PINN
    with open(os.path.join(args.pinn_meta_dir,
                           f'training_meta_{args.tag}.json')) as fh:
        meta = json.load(fh)
    pinn = build_pinn(
        num_inputs=2,
        layers=tuple(meta.get('layers', [256, 256, 256, 256, 256])),
        activation='tanh',
        input_range=((0.0, 1.0), (0.0, 1.0)),
        fourier_m=int(meta.get('fourier_m', 64)),
        fourier_scale=float(meta.get('fourier_scale', k / (2 * np.pi))),
    )
    _ = pinn(tf.zeros((1, 2)))
    pinn.load_weights(os.path.join(
        args.pinn_dir, f'pinn_helmholtz_{args.tag}.weights.h5'))

    router = RouterCNN(base_filters=args.base_filters)
    dummy = create_router_input(layout, f_grid, pinn_u, residual)
    _ = router(tf.constant(dummy, dtype=tf.float32))
    router.load_weights(os.path.join(
        args.router_dir, f'router_helmholtz_{args.tag}.weights.h5'))

    res = solve_hybrid_schwarz(solver, pinn, router, f_callable, g_callable,
                               X, Y, layout, f_grid, pinn_u, residual,
                               threshold=0.0, reuse_logits=logits)
    return res['u_grid'], u_fem, layout


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', type=str, required=True)
    p.add_argument('--timing-dir', type=str, default='./timing')
    p.add_argument('--plots-dir', type=str, default='./plots')
    p.add_argument('--pinn-dir', type=str, default='./models')
    p.add_argument('--pinn-meta-dir', type=str, default='./history')
    p.add_argument('--router-dir', type=str, default='./router_models')
    p.add_argument('--mesh-n', type=int, default=129)
    p.add_argument('--base-filters', type=int, default=32)
    args = p.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    with open(os.path.join(args.timing_dir, f'timing_{args.tag}.json')) as f:
        summary = json.load(f)
    ref = np.load(os.path.join(args.timing_dir, f'reference_{args.tag}.npz'))

    u_hybrid, u_fem, layout = _rebuild_hybrid_solution(args, summary, ref)
    X, Y = ref['X'], ref['Y']

    mask = layout > 0
    u_fem_m = np.where(mask, u_fem, np.nan)
    u_hyb_m = np.where(mask, u_hybrid, np.nan)
    err = np.where(mask, u_hybrid - u_fem, np.nan)

    sweep = summary['coverage_sweep']
    cov = np.array([s['actual_coverage_pct'] for s in sweep])
    errs = np.array([s['rel_l2_vs_fem'] for s in sweep])
    wall = np.array([s['hybrid_total_s'] for s in sweep])
    fem_mean = summary['fem_mean_s']
    speedup = fem_mean / np.maximum(wall, 1e-12)
    pinn_only = summary['pinn_rel_l2_vs_fem']

    fig = plt.figure(figsize=(18, 10))

    # Row 1: solutions
    ax1 = fig.add_subplot(2, 3, 1)
    vmax = float(np.nanmax(np.abs(u_fem_m)))
    im1 = ax1.pcolormesh(X, Y, u_fem_m, cmap='RdBu_r',
                         vmin=-vmax, vmax=vmax, shading='auto')
    ax1.set_title('Full FEM  u'); ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.pcolormesh(X, Y, u_hyb_m, cmap='RdBu_r',
                         vmin=-vmax, vmax=vmax, shading='auto')
    ax2.set_title('Hybrid  u  (threshold=0)'); ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    ax3 = fig.add_subplot(2, 3, 3)
    ev = float(np.nanmax(np.abs(err)))
    im3 = ax3.pcolormesh(X, Y, err, cmap='RdBu_r',
                         vmin=-ev, vmax=ev, shading='auto')
    rl = rel_l2(u_hybrid, u_fem)
    ax3.set_title(f'u_hybrid - u_FEM  (rel L2 = {rl:.3e})')
    ax3.set_aspect('equal')
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # Row 2: coverage curves
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(cov, errs, 'o-', color='C0', label='Hybrid rel L2')
    ax4.axhline(pinn_only, ls='--', color='C3',
                label=f'PINN-only ({pinn_only:.2e})')
    ax4.set_xlabel('Rejection coverage (%)')
    ax4.set_ylabel('Rel L2 vs full FEM')
    ax4.set_yscale('log')
    ax4.set_title('Accuracy vs coverage')
    ax4.grid(True, alpha=0.3); ax4.legend()

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(cov, speedup, 'o-', color='C2')
    ax5.axhline(1.0, ls='--', color='k', alpha=0.5, label='FEM baseline')
    ax5.set_xlabel('Rejection coverage (%)')
    ax5.set_ylabel('Speedup  (FEM / hybrid wall)')
    ax5.set_title('Speedup vs coverage')
    ax5.grid(True, alpha=0.3); ax5.legend()

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(cov, wall * 1000, 'o-', color='C1', label='Hybrid total')
    ax6.axhline(fem_mean * 1000, ls='--', color='k',
                label=f'FEM baseline ({fem_mean*1000:.1f} ms)')
    ax6.set_xlabel('Rejection coverage (%)')
    ax6.set_ylabel('Wall time (ms)')
    ax6.set_title('Wall time vs coverage')
    ax6.grid(True, alpha=0.3); ax6.legend()

    fig.suptitle(
        f'Helmholtz Exp 2  tag={args.tag}  k={summary["k"]:.3f}  '
        f'FEM baseline={fem_mean*1000:.1f} ms  '
        f'hybrid@0={summary["hybrid_mean_s"]*1000:.1f} ms  '
        f'speedup={summary["speedup"]:.2f}x')
    fig.tight_layout()
    out_png = os.path.join(args.plots_dir, f'hybrid_{args.tag}.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f'Saved plot: {out_png}')

    metrics = {
        'tag': args.tag,
        'k': summary['k'],
        'fem_baseline_ms': fem_mean * 1000,
        'hybrid_threshold0_ms': summary['hybrid_mean_s'] * 1000,
        'speedup_at_threshold0': summary['speedup'],
        'pinn_only_rel_l2': pinn_only,
        'hybrid_threshold0_rel_l2': float(rl),
        'coverage_pct': cov.tolist(),
        'hybrid_rel_l2': errs.tolist(),
        'hybrid_wall_ms': (wall * 1000).tolist(),
        'speedup': speedup.tolist(),
    }
    out_json = os.path.join(args.plots_dir, f'metrics_exp2_{args.tag}.json')
    with open(out_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'Saved metrics: {out_json}')


if __name__ == '__main__':
    main()
