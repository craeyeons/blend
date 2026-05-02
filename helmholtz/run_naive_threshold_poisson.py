"""
Naive-threshold sanity check for the Poisson plate-with-hole setup.

No router. For one config (default: src_p1) build R(x) = median_normalize(
|residual| + |ete|), set Omega_C = {R(x) >= beta}, run the existing FEM
hybrid via solve_hybrid_schwarz, and dump a single hybrid RMSE vs FEM
into stats.txt.

Reuses helpers from train_router_multi.py and lib/router.py.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Repo path so the same imports as train_router_multi.py work.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.fem_solver import HelmholtzSolver
from lib.hybrid import rmse, solve_hybrid_schwarz
from train_router_multi import prepare_config  # reuses meta loading + ETE


def plot_panels(u_pinn, u_hybrid, u_fem, X, Y, layout, hole, save_path):
    """1-row x 3-col PINN/Hybrid/FEM panel using RdBu_r everywhere."""
    cx, cy, rh = hole
    mask = layout > 0

    def _m(a):
        return np.where(mask, a, np.nan)

    fields = [_m(u_pinn), _m(u_hybrid), _m(u_fem)]
    titles = ['PINN  u', 'Hybrid  u', 'FEM  u']
    stacked = np.concatenate([f[mask] for f in [u_pinn, u_hybrid, u_fem]])
    absmax = float(np.max(np.abs(stacked))) + 1e-30

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, f, t in zip(axes, fields, titles):
        im = ax.pcolormesh(X, Y, f, cmap='RdBu_r',
                           vmin=-absmax, vmax=absmax, shading='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, label='u')
        circle = plt.Circle((cx, cy), rh, color='gray', fill=True, zorder=5)
        ax.add_patch(circle)
        ax.set_aspect('equal')
        ax.set_title(t)
        ax.set_xlabel('x')
    axes[0].set_ylabel('y')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_mask(accept_mask, X, Y, layout, hole, save_path):
    """Standalone partition plot: 1 = FEM-solved (rejected), 0 = PINN-pinned.

    `accept_mask` (from solve_hybrid_schwarz) is 1 where PINN is pinned,
    so we visualise 1 - accept_mask intersected with the active region.
    """
    cx, cy, rh = hole
    mask = layout > 0
    fem_mask = (mask & (accept_mask == 0)).astype(float)
    fig, ax = plt.subplots(figsize=(7, 7))
    data = np.where(mask, fem_mask, np.nan)
    im = ax.pcolormesh(X, Y, data, cmap='RdBu_r', vmin=0.0, vmax=1.0,
                       shading='auto')
    plt.colorbar(im, ax=ax, fraction=0.046, label='1 = FEM, 0 = PINN')
    ax.contour(X, Y, fem_mask, levels=[0.5], colors='black', linewidths=1.5)
    circle = plt.Circle((cx, cy), rh, color='gray', fill=True, zorder=5)
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title('Naive-threshold partition  (R(x) >= beta)')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(
        description='Naive-threshold sanity check for Poisson plate-with-hole.'
    )
    p.add_argument('--configs', type=str, default='configs_exp3.json')
    p.add_argument('--config-name', type=str, default='src_p1',
                   help='Config name from the train split.')
    p.add_argument('--beta', type=float, default=1.1)
    p.add_argument('--mesh-n', type=int, default=129)
    p.add_argument('--nx', type=int, default=201)
    p.add_argument('--ny', type=int, default=201)
    p.add_argument('--sigma', type=float, default=0.05)
    p.add_argument('--amplitude', type=float, default=250.0)
    p.add_argument('--hole-center-x', type=float, default=0.5)
    p.add_argument('--hole-center-y', type=float, default=0.5)
    p.add_argument('--hole-radius', type=float, default=0.15)
    p.add_argument('--history-dir', type=str, default='./history')
    p.add_argument('--output-dir', type=str,
                   default='./naive_threshold_output_poisson')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Locate the config entry by name (train split first, then test).
    with open(args.configs) as fp:
        cfgs = json.load(fp)
    candidates = cfgs.get('train', []) + cfgs.get('test', [])
    entry = next((e for e in candidates if e['name'] == args.config_name), None)
    if entry is None:
        raise SystemExit(f'config name {args.config_name!r} not found in {args.configs}')

    # 2) Build grid + global hole and prepare PINN/residual/ETE.
    x = np.linspace(0.0, 1.0, args.nx)
    y = np.linspace(0.0, 1.0, args.ny)
    X, Y = np.meshgrid(x, y)
    hole = (args.hole_center_x, args.hole_center_y, args.hole_radius)
    cx, cy, rh = hole
    layout = ((X - cx) ** 2 + (Y - cy) ** 2 > rh ** 2).astype(np.float32)

    print(f'[1/4] Preparing config {entry["name"]} ...')
    cfg_data = prepare_config(entry, X, Y, layout, hole,
                              args.sigma, args.amplitude, args.history_dir)

    test_layout = cfg_data['test_layout']
    test_hole = cfg_data['test_hole']
    cx_t, cy_t, rh_t = test_hole
    xs, ys = cfg_data['x_s'], cfg_data['y_s']

    def dirichlet_pred(x_, y_):
        return (x_ - cx_t) ** 2 + (y_ - cy_t) ** 2 <= rh_t ** 2
    def f_callable(x_, y_, xs_=xs, ys_=ys):
        return args.amplitude * np.exp(-((x_ - xs_) ** 2 + (y_ - ys_) ** 2)
                                       / (2.0 * args.sigma ** 2))
    def g_callable(x_, y_):
        return np.zeros_like(x_)

    # 3) FEM reference.
    print('[2/4] Solving FEM reference ...')
    solver = HelmholtzSolver(k=cfg_data['k'], mesh_n=args.mesh_n,
                             dirichlet_predicate=dirichlet_pred)
    fem_t0 = time.perf_counter()
    u_dof, _ = solver.solve(f_callable, g_callable)
    u_fem = solver.interp_to_grid(u_dof, X, Y)
    fem_time = time.perf_counter() - fem_t0

    pinn_rmse = rmse(cfg_data['pinn_u'], u_fem)
    print(f'  FEM time {fem_time*1000:.1f} ms   PINN RMSE = {pinn_rmse:.3e}')

    # 4) Naive-threshold hybrid: pass R(x) (= cfg_data['target']) as the
    #    "logits" and beta as the threshold. solve_hybrid_schwarz forms
    #    Omega_C = {logits >= threshold}, exactly the naive rule.
    R = cfg_data['target'].numpy() if hasattr(cfg_data['target'], 'numpy') \
        else np.asarray(cfg_data['target'])

    print(f'[3/4] Hybrid with mask = (R(x) >= {args.beta:.3f}) ...')
    hyb_t0 = time.perf_counter()
    res = solve_hybrid_schwarz(
        solver, cfg_data['pinn'], None,  # router=None: we pass logits directly
        f_callable, g_callable,
        X, Y, test_layout,
        cfg_data['f_grid'], cfg_data['pinn_u'],
        cfg_data['residual'], ete_grid=cfg_data['ete'],
        threshold=float(args.beta), reuse_logits=R,
        apply_morph_opening=False,  # naive rule: no smoothing
    )
    hyb_time = time.perf_counter() - hyb_t0

    hybrid_rmse = rmse(res['u_grid'], u_fem)
    coverage_pct = float(res['coverage_pct'])

    print('[4/4] Writing stats ...')
    stats_path = os.path.join(args.output_dir, 'stats.txt')
    with open(stats_path, 'w') as fp:
        fp.write('Naive-threshold sanity check (Poisson plate-with-hole)\n')
        fp.write('======================================================\n')
        fp.write(f'config:              {entry["name"]}\n')
        fp.write(f'PINN path:           {entry["pinn_path"]}\n')
        fp.write(f'beta:                {args.beta}\n')
        fp.write(f'source (x_s, y_s):   ({xs}, {ys})\n')
        fp.write(f'hole (cx, cy, r):    ({cx_t}, {cy_t}, {rh_t})\n')
        fp.write(f'FEM coverage:        {coverage_pct:.2f}%\n')
        fp.write(f'PINN RMSE vs FEM:    {pinn_rmse:.6e}\n')
        fp.write(f'Hybrid RMSE vs FEM:  {hybrid_rmse:.6e}\n')
        fp.write(f'FEM solve time:      {fem_time:.4f} s\n')
        fp.write(f'Hybrid solve time:   {hyb_time:.4f} s\n')

    np.savez(os.path.join(args.output_dir, 'fields.npz'),
             u_pinn=cfg_data['pinn_u'], u_hybrid=res['u_grid'], u_fem=u_fem,
             accept_mask=res['accept_mask'], R=R, layout=test_layout)

    plot_panels(
        cfg_data['pinn_u'], res['u_grid'], u_fem,
        X, Y, test_layout, test_hole,
        save_path=os.path.join(args.output_dir, 'solution_comparison.png'),
    )
    plot_mask(
        res['accept_mask'], X, Y, test_layout, test_hole,
        save_path=os.path.join(args.output_dir, 'partition_mask.png'),
    )

    print(f'-> {stats_path}')
    print(f'   PINN RMSE   = {pinn_rmse:.6e}')
    print(f'   Hybrid RMSE = {hybrid_rmse:.6e}   (FEM coverage {coverage_pct:.1f}%)')


if __name__ == '__main__':
    main()
