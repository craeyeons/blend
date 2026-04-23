"""
Per-config plots for Exp 3 multi-config hybrid Helmholtz.

For each config (ID and OOD):
  - solution_{tag}.png:
        row 1 (u):     PINN | Hybrid | FEM
        row 2 (error): PINN - FEM | Hybrid - FEM | FEM - FEM
    Title includes k, x_s, y_s, split (ID / OOD).
  - rmse_vs_coverage_{tag}.png:
        hybrid RMSE vs FEM coverage, horizontal baseline = PINN-only RMSE.

Also saves:
  - summary_{tag_prefix}.json: table of {name, split, k, x_s, y_s,
        pinn_rmse, best_hybrid_rmse, best_hybrid_cov, fem_mean_ms,
        hybrid_at_best_ms, speedup_at_best}.

Usage:
    python plot_hybrid_multi.py --pinn-tag exp3_parametric \
        --router-tag exp3 --configs configs_exp3.json --tag-prefix exp3
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from lib.domains import create_square_with_hole, gaussian_source
from lib.fem_solver import HelmholtzSolver
from lib.network import build_parametric_pinn
from lib.router import RouterCNN, create_router_input, median_normalize
from lib.hybrid import solve_hybrid_schwarz, rmse


def _find_optimal_threshold(logits, residual, ete, layout, beta):
    solid = layout > 0
    lv = logits[solid]
    raw_sum = residual + (ete if ete is not None else 0.0)
    r_bar = median_normalize(raw_sum, layout)[solid]
    thresholds = np.unique(np.sort(lv))
    best = (None, None, np.inf)
    for thr in thresholds:
        rej = lv >= thr
        loss = beta * rej.mean() + (r_bar * (~rej)).mean()
        if loss < best[2]:
            best = (float(thr), float(rej.mean()) * 100.0, float(loss))
    return best


def _plot_solution(X, Y, layout, pinn_u, u_hybrid, u_fem, accept_mask,
                   cfg_title, out_path):
    mask = layout > 0

    def _m(a): return np.where(mask, a, np.nan)

    u_pinn_m = _m(pinn_u)
    u_hyb_m = _m(u_hybrid)
    u_fem_m = _m(u_fem)

    err_pinn = _m(pinn_u - u_fem)
    err_hyb = _m(u_hybrid - u_fem)
    err_fem = _m(np.zeros_like(u_fem))

    # Shared scale for u row and error row separately.
    u_vmax = float(np.nanmax(np.abs(u_fem_m))) + 1e-30
    all_errs = np.concatenate([np.abs(err_pinn[mask]),
                               np.abs(err_hyb[mask])])
    e_vmax = float(np.nanmax(all_errs)) + 1e-30

    reject_field = (~accept_mask.astype(bool) & mask).astype(np.float32)

    def _shade_hybrid(ax):
        ax.contourf(X, Y, reject_field, levels=[0.5, 1.5],
                    colors=['black'], alpha=0.30)
        ax.contour(X, Y, reject_field, levels=[0.5],
                   colors='lime', linewidths=1.2)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    col_titles = ['PINN', 'Hybrid (shaded=FEM)', 'FEM']
    u_data = [u_pinn_m, u_hyb_m, u_fem_m]
    err_data = [err_pinn, err_hyb, err_fem]

    for j in range(3):
        ax = axes[0, j]
        im = ax.pcolormesh(X, Y, u_data[j], cmap='RdBu_r',
                           vmin=-u_vmax, vmax=u_vmax, shading='auto')
        if j == 1:
            _shade_hybrid(ax)
        ax.set_title(f'{col_titles[j]} — u'); ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, fraction=0.046)

    err_col_titles = ['PINN - FEM', 'Hybrid - FEM', 'FEM - FEM']
    for j in range(3):
        ax = axes[1, j]
        im = ax.pcolormesh(X, Y, err_data[j], cmap='RdBu_r',
                           vmin=-e_vmax, vmax=e_vmax, shading='auto')
        if j == 1:
            _shade_hybrid(ax)
        ax.set_title(err_col_titles[j]); ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, fraction=0.046)

    rl_pinn = rmse(pinn_u, u_fem)
    rl_hyb = rmse(u_hybrid, u_fem)
    cov_pct = 100.0 * reject_field.sum() / max(mask.sum(), 1)
    fig.suptitle(f'{cfg_title}  |  '
                 f'FEM coverage={cov_pct:.1f}%  '
                 f'PINN RMSE={rl_pinn:.2e}  Hybrid RMSE={rl_hyb:.2e}')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_rmse_vs_coverage(sweep, pinn_rmse, cfg_title, out_path):
    cov = np.array([s['actual_coverage_pct'] for s in sweep])
    err = np.array([s['rmse_vs_fem'] for s in sweep])
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(cov, err, 'o-', label='Hybrid RMSE')
    ax.axhline(pinn_rmse, ls='--', color='C3',
               label=f'PINN-only baseline ({pinn_rmse:.2e})')
    ax.set_xlabel('FEM coverage (%)')
    ax.set_ylabel('RMSE vs full FEM')
    ax.set_yscale('log')
    ax.set_title(f'RMSE vs coverage — {cfg_title}')
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pinn-tag', type=str, required=True)
    p.add_argument('--router-tag', type=str, required=True)
    p.add_argument('--configs', type=str, required=True)
    p.add_argument('--tag-prefix', type=str, required=True)
    p.add_argument('--pinn-dir', type=str, default='./models')
    p.add_argument('--pinn-meta-dir', type=str, default='./history')
    p.add_argument('--router-dir', type=str, default='./router_models')
    p.add_argument('--router-history-dir', type=str, default='./router_history')
    p.add_argument('--timing-dir', type=str, default='./timing')
    p.add_argument('--plots-dir', type=str, default='./plots')
    p.add_argument('--mesh-n', type=int, default=129)
    p.add_argument('--nx', type=int, default=201)
    p.add_argument('--ny', type=int, default=201)
    p.add_argument('--base-filters', type=int, default=32)
    p.add_argument('--beta', type=float, default=None)
    args = p.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    # Router beta
    beta = args.beta
    if beta is None:
        rmeta_path = os.path.join(args.router_history_dir,
                                  f'router_meta_{args.router_tag}.json')
        with open(rmeta_path) as f:
            beta = float(json.load(f).get('beta', 0.1))

    with open(os.path.join(args.pinn_meta_dir,
                           f'training_meta_{args.pinn_tag}.json')) as f:
        pmeta = json.load(f)
    cx = float(pmeta['hole_center_x']); cy = float(pmeta['hole_center_y'])
    r_h = float(pmeta['hole_radius'])
    sigma = float(pmeta['sigma']); amplitude = float(pmeta['amplitude'])

    X, Y, layout, _, _ = create_square_with_hole(
        Nx=args.nx, Ny=args.ny,
        hole_center=(cx, cy), hole_radius=r_h)

    def dirichlet_predicate(x, y):
        return (x - cx) ** 2 + (y - cy) ** 2 <= r_h ** 2

    pinn = build_parametric_pinn(
        layers=tuple(pmeta['layers']),
        spatial_range=((0.0, 1.0), (0.0, 1.0)),
        param_range=((pmeta['k_min'], pmeta['k_max']),
                     (pmeta['xs_min'], pmeta['xs_max']),
                     (pmeta['ys_min'], pmeta['ys_max'])),
        fourier_m=int(pmeta['fourier_m']),
        fourier_scale=float(pmeta['fourier_scale']),
    )
    _ = pinn(tf.zeros((1, 5)))
    pinn.load_weights(os.path.join(
        args.pinn_dir, f'pinn_helmholtz_{args.pinn_tag}.weights.h5'))

    router = RouterCNN(base_filters=args.base_filters)
    _ = router(tf.constant(np.zeros((1, args.ny, args.nx, 5), np.float32)))
    router.load_weights(os.path.join(
        args.router_dir, f'router_helmholtz_{args.router_tag}.weights.h5'))

    with open(args.configs) as f:
        configs = json.load(f)

    table = []
    for cfg in configs:
        name = cfg['name']; split = cfg.get('split', 'id')
        k = float(cfg['k']); xs = float(cfg['x_s']); ys = float(cfg['y_s'])
        tag = f"{args.tag_prefix}_{name}"
        cfg_title = (f'{name}  ({split.upper()})  '
                     f'k={k:.3f}  x_s={xs:.3f}  y_s={ys:.3f}')

        sweep_path = os.path.join(args.timing_dir, f'sweep_{tag}.json')
        ref_path = os.path.join(args.timing_dir, f'reference_{tag}.npz')
        if not (os.path.exists(sweep_path) and os.path.exists(ref_path)):
            print(f"skipping {name}: missing timing outputs")
            continue
        with open(sweep_path) as f:
            summary = json.load(f)
        ref = np.load(ref_path)

        # Optimal threshold on decision-loss.
        opt_thr, opt_cov_pct, opt_loss = _find_optimal_threshold(
            ref['logits'], ref['residual'],
            ref['ete'] if 'ete' in ref.files else None,
            layout, beta)

        # Solver + callables for this config.
        solver = HelmholtzSolver(k=k, mesh_n=args.mesh_n,
                                 dirichlet_predicate=dirichlet_predicate)
        def f_callable(x, y, xs_=xs, ys_=ys):
            return amplitude * np.exp(-((x - xs_) ** 2 + (y - ys_) ** 2)
                                      / (2.0 * sigma ** 2))
        def g_callable(x, y):
            return np.zeros_like(x)

        # Hybrid at optimal threshold (short-circuit if coverage is extreme).
        if opt_cov_pct >= 99.5:
            u_hybrid = ref['u_fem']
            accept_mask = np.zeros_like(layout, dtype=np.int32)
            note = '(short-circuit: opt_cov≥99.5% → full FEM)'
        elif opt_cov_pct <= 0.5:
            u_hybrid = ref['pinn_u']
            accept_mask = (layout > 0).astype(np.int32)
            note = '(short-circuit: opt_cov≤0.5% → pure PINN)'
        else:
            res_opt = solve_hybrid_schwarz(
                solver, pinn, router, f_callable, g_callable,
                X, Y, layout, ref['f_grid'], ref['pinn_u'], ref['residual'],
                ete_grid=(ref['ete'] if 'ete' in ref.files else None),
                threshold=float(opt_thr), reuse_logits=ref['logits'])
            u_hybrid = res_opt['u_grid']
            accept_mask = res_opt['accept_mask']
            note = ''

        # Solution + error plot
        out_sol = os.path.join(args.plots_dir, f'solution_{tag}.png')
        _plot_solution(X, Y, layout, ref['pinn_u'], u_hybrid, ref['u_fem'],
                       accept_mask,
                       cfg_title + (f'  {note}' if note else ''),
                       out_sol)
        print(f"saved {out_sol}")

        # RMSE vs coverage plot
        out_cov = os.path.join(args.plots_dir, f'rmse_vs_coverage_{tag}.png')
        _plot_rmse_vs_coverage(summary['coverage_sweep'],
                               summary['pinn_rmse_vs_fem'],
                               cfg_title, out_cov)
        print(f"saved {out_cov}")

        # Stats for table
        cov_arr = np.array([s['actual_coverage_pct']
                            for s in summary['coverage_sweep']])
        err_arr = np.array([s['rmse_vs_fem']
                            for s in summary['coverage_sweep']])
        wall_arr = np.array([s['hybrid_total_s']
                             for s in summary['coverage_sweep']])
        i_best = int(np.argmin(err_arr))
        table.append({
            'name': name, 'split': split,
            'k': k, 'x_s': xs, 'y_s': ys,
            'pinn_rmse': float(summary['pinn_rmse_vs_fem']),
            'hybrid_rmse_at_optimal': float(rmse(u_hybrid, ref['u_fem'])),
            'optimal_coverage_pct': float(opt_cov_pct),
            'best_hybrid_rmse_in_sweep': float(err_arr[i_best]),
            'best_hybrid_cov_pct': float(cov_arr[i_best]),
            'fem_mean_ms': float(summary['fem_mean_s']) * 1000,
            'hybrid_at_best_ms': float(wall_arr[i_best]) * 1000,
            'speedup_at_best': float(summary['fem_mean_s']
                                     / max(wall_arr[i_best], 1e-12)),
        })

    out_table = os.path.join(args.plots_dir,
                             f'summary_{args.tag_prefix}.json')
    with open(out_table, 'w') as f:
        json.dump({'beta': beta, 'entries': table}, f, indent=2)
    print(f"\nSaved summary table: {out_table}")


if __name__ == '__main__':
    main()
