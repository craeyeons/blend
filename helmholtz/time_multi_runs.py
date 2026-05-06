#!/usr/bin/env python3
"""Multi-config Helmholtz/Poisson timing across 3 router retrainings.

Mirrors blend/time_multi_runs.py but for FEM/PINN hybrid (configs_exp3.json).

For each config in configs:
  - Solve FEM 3x; keep one solution for RMSE.
  - Train router 3x (different seeds), save weights to
      <out>/router_runs/router_helmholtz_run{0,1,2}.weights.h5
  - For each trained router:
      * abstention loss at optimal threshold (training-loss picker)
      * FEM coverage % at optimal threshold
      * hybrid solve time (single timing per router)
      * RMSE vs FEM
  - Aggregate mean/std across 3 router runs.
  - Plots (run 0): regions_and_errors.pdf, solution_comparison.pdf
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from lib.domains import create_square_with_hole
from lib.fem_solver import HelmholtzSolver
from lib.router import RouterCNN, create_router_input
from lib.hybrid import solve_hybrid_schwarz, rmse

from train_router_multi import (
    prepare_config,
    _optimal_threshold_from_training_loss,
    DEFAULT_SIGMA, DEFAULT_AMPLITUDE, DEFAULT_HOLE,
)


N_SEEDS = 3
N_FEM_RUNS = 3


def train_routers(args):
    runs_dir = Path(args.output_dir) / 'router_runs'
    runs_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parent / 'train_router_multi.py'
    for s in range(N_SEEDS):
        tag = f"run{s}"
        weights = runs_dir / f'router_helmholtz_{tag}.weights.h5'
        if weights.exists() and not args.retrain:
            print(f"[run {s}] weights exist, skipping training.")
            continue
        cmd = [
            sys.executable, str(script),
            '--configs', args.configs,
            '--tag', tag,
            '--output-dir', str(runs_dir),
            '--plots-dir', str(runs_dir / f'plots_{tag}'),
            '--timing-dir', str(runs_dir / f'timing_{tag}'),
            '--history-dir', args.history_dir,
            '--epochs', str(args.epochs),
            '--lr', str(args.lr),
            '--beta', str(args.beta),
            '--lambda-tv', str(args.lambda_tv),
            '--base-filters', str(args.base_filters),
            '--mesh-n', str(args.mesh_n),
            '--nx', str(args.nx), '--ny', str(args.ny),
            '--sigma', str(args.sigma),
            '--amplitude', str(args.amplitude),
            '--hole-center-x', str(args.hole_center_x),
            '--hole-center-y', str(args.hole_center_y),
            '--hole-radius', str(args.hole_radius),
            '--seed', str(s),
            '--skip-analysis',
        ]
        print(f"[run {s}] training: tag={tag}")
        subprocess.run(cmd, check=True)


def _plot_regions_and_errors(X, Y, test_layout, test_hole, accept_mask,
                             pinn_u, u_hybrid, u_fem, save_path):
    cx, cy, rh = test_hole
    fluid = test_layout > 0

    err_pinn = np.abs(pinn_u - u_fem) * test_layout
    err_hyb = np.abs(u_hybrid - u_fem) * test_layout

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Region map (1=FEM, 0=PINN)
    ax = axes[0]
    region = np.full(test_layout.shape, np.nan)
    accept = accept_mask.astype(bool)
    region[fluid & accept] = 1.0   # FEM
    region[fluid & ~accept] = 0.0  # PINN
    cmap_rb = ListedColormap(['#3b82f6', '#ef4444'])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap_rb.N)
    im = ax.pcolormesh(X, Y, region, cmap=cmap_rb, norm=norm, shading='auto')
    cb = plt.colorbar(im, ax=ax, fraction=0.046, ticks=[0, 1])
    cb.ax.set_yticklabels(['PINN', 'FEM'])
    ax.add_patch(plt.Circle((cx, cy), rh, color='gray', fill=True, zorder=5))
    ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')

    err_max = float(max(np.max(err_pinn[fluid]), np.max(err_hyb[fluid]))) + 1e-30
    for ax, err, lab in zip(axes[1:], [err_pinn, err_hyb], ['PINN error', 'Hybrid error']):
        data = np.where(fluid, err, np.nan)
        im = ax.pcolormesh(X, Y, data, cmap='YlOrRd',
                           vmin=0.0, vmax=err_max, shading='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, label=lab)
        ax.add_patch(plt.Circle((cx, cy), rh, color='gray', fill=True, zorder=5))
        ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')

    plt.tight_layout()
    fig.savefig(save_path, dpi=1200, bbox_inches='tight')
    plt.close(fig)


def _plot_solution_comparison(X, Y, test_layout, test_hole, accept_mask,
                              pinn_u, u_hybrid, u_fem, save_path):
    """3 columns (PINN | Hybrid | FEM), 2 rows (solution | error). No black."""
    cx, cy, rh = test_hole
    fluid = test_layout > 0

    sols = [pinn_u, u_hybrid, u_fem]
    err_pinn = np.abs(pinn_u - u_fem) * test_layout
    err_hyb = np.abs(u_hybrid - u_fem) * test_layout
    err_fem = np.zeros_like(u_fem)
    errs = [err_pinn, err_hyb, err_fem]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Row 0: solutions, shared diverging coolwarm
    sol_stack = np.concatenate([s[fluid] for s in sols])
    smax = float(np.max(np.abs(sol_stack))) + 1e-30
    for j, s in enumerate(sols):
        ax = axes[0, j]
        data = np.where(fluid, s, np.nan)
        im = ax.pcolormesh(X, Y, data, cmap='coolwarm',
                           vmin=-smax, vmax=smax, shading='auto')
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.add_patch(plt.Circle((cx, cy), rh, color='gray', fill=True, zorder=5))
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])

    # Row 1: errors, YlOrRd (no black)
    err_max = float(max(np.max(err_pinn[fluid]), np.max(err_hyb[fluid]))) + 1e-30
    for j, e in enumerate(errs):
        ax = axes[1, j]
        data = np.where(fluid, e, np.nan)
        im = ax.pcolormesh(X, Y, data, cmap='YlOrRd',
                           vmin=0.0, vmax=err_max, shading='auto')
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.add_patch(plt.Circle((cx, cy), rh, color='gray', fill=True, zorder=5))
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])

    # Column labels
    for j, lab in enumerate(['PINN', 'Hybrid', 'FEM']):
        axes[0, j].set_xlabel(lab)
        axes[0, j].xaxis.set_label_position('top')
    axes[0, 0].set_ylabel('solution')
    axes[1, 0].set_ylabel('|err vs FEM|')

    plt.tight_layout()
    fig.savefig(save_path, dpi=1200, bbox_inches='tight')
    plt.close(fig)


def evaluate_config(cfg_data, args, runs_dir, X, Y, layout, hole, out_dir, label):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_layout = cfg_data['test_layout']
    test_hole = cfg_data['test_hole']
    cx, cy, rh = test_hole
    xs, ys = cfg_data['x_s'], cfg_data['y_s']
    k = cfg_data['k']
    pinn_u = cfg_data['pinn_u']

    def f_callable(x, y, xs_=xs, ys_=ys):
        return args.amplitude * np.exp(-((x - xs_)**2 + (y - ys_)**2)
                                       / (2.0 * args.sigma**2))
    def g_callable(x, y):
        return np.zeros_like(x)
    def dpred(x, y):
        return (x - cx)**2 + (y - cy)**2 <= rh**2

    solver = HelmholtzSolver(k=k, mesh_n=args.mesh_n, dirichlet_predicate=dpred)

    # FEM x N_FEM_RUNS
    fem_times = []
    u_fem = None
    for i in range(N_FEM_RUNS):
        t0 = time.perf_counter()
        u_dof, _ = solver.solve(f_callable, g_callable)
        u_grid = solver.interp_to_grid(u_dof, X, Y)
        fem_times.append(time.perf_counter() - t0)
        if u_fem is None:
            u_fem = u_grid
        print(f"  [{label}] FEM {i+1}/{N_FEM_RUNS}: {fem_times[-1]:.3f}s")

    abst_losses, covs, hyb_times, rmses = [], [], [], []
    plot_payload = None
    target_R = cfg_data['target'].numpy() if hasattr(cfg_data['target'], 'numpy') \
        else np.asarray(cfg_data['target'])

    for s in range(N_SEEDS):
        weights = runs_dir / f'router_helmholtz_run{s}.weights.h5'
        router = RouterCNN(base_filters=args.base_filters)
        _ = router(cfg_data['inputs'])
        router.load_weights(str(weights))
        logits = router(cfg_data['inputs'], training=False)[0, :, :, 0].numpy()

        thr, rej_pct, abst_loss = _optimal_threshold_from_training_loss(
            logits, target_R, test_layout, float(args.beta))
        # rej_pct is rejection % (= PINN %) → CFD/FEM coverage = 100 - rej_pct
        actual_cov_pct = 100.0 - rej_pct

        # Hybrid solve (timed)
        if actual_cov_pct >= 99.5:
            u_hyb = u_fem
            accept_mask = (test_layout > 0).astype(np.int32)
            hyb_t = float(np.mean(fem_times))
            err = 0.0
        elif actual_cov_pct <= 0.5:
            u_hyb = pinn_u
            accept_mask = np.zeros_like(test_layout, dtype=np.int32)
            hyb_t = 0.0
            err = float(rmse(pinn_u, u_fem))
        else:
            t0 = time.perf_counter()
            res = solve_hybrid_schwarz(
                solver, cfg_data['pinn'], router, f_callable, g_callable,
                X, Y, test_layout, cfg_data['f_grid'], cfg_data['pinn_u'],
                cfg_data['residual'], ete_grid=cfg_data['ete'],
                threshold=float(thr), reuse_logits=logits)
            hyb_t = time.perf_counter() - t0
            u_hyb = res['u_grid']
            accept_mask = res['accept_mask']
            err = float(rmse(u_hyb, u_fem))

        abst_losses.append(abst_loss)
        covs.append(actual_cov_pct)
        hyb_times.append(hyb_t)
        rmses.append(err)
        print(f"  [{label}] run {s}: cov={actual_cov_pct:.2f}% "
              f"abst_loss={abst_loss:.4f} hyb_time={hyb_t:.3f}s "
              f"RMSE={err:.4e}")

        if s == 0:
            plot_payload = dict(u_hyb=u_hyb, accept_mask=accept_mask)

    summary = {
        'label': label,
        'name': cfg_data['name'], 'k': k, 'x_s': xs, 'y_s': ys,
        'fem_time_mean': float(np.mean(fem_times)),
        'fem_time_std': float(np.std(fem_times)),
        'fem_times': fem_times,
        'hybrid_time_mean': float(np.mean(hyb_times)),
        'hybrid_time_std': float(np.std(hyb_times)),
        'hybrid_times': hyb_times,
        'abstention_loss_mean': float(np.mean(abst_losses)),
        'abstention_loss_std': float(np.std(abst_losses)),
        'abstention_losses': abst_losses,
        'fem_coverage_mean': float(np.mean(covs)),
        'fem_coverage_std': float(np.std(covs)),
        'fem_coverages': covs,
        'rmse_mean': float(np.mean(rmses)),
        'rmse_std': float(np.std(rmses)),
        'rmses': rmses,
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    p = plot_payload
    _plot_regions_and_errors(X, Y, test_layout, test_hole, p['accept_mask'],
                             pinn_u, p['u_hyb'], u_fem,
                             str(out_dir / 'regions_and_errors.pdf'))
    _plot_solution_comparison(X, Y, test_layout, test_hole, p['accept_mask'],
                              pinn_u, p['u_hyb'], u_fem,
                              str(out_dir / 'solution_comparison.pdf'))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--configs', required=True)
    p.add_argument('--output-dir', default='./time_multi_runs')
    p.add_argument('--history-dir', default='./history')
    p.add_argument('--epochs', type=int, default=2000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--lambda-tv', type=float, default=1.0)
    p.add_argument('--base-filters', type=int, default=32)
    p.add_argument('--mesh-n', type=int, default=129)
    p.add_argument('--nx', type=int, default=201)
    p.add_argument('--ny', type=int, default=201)
    p.add_argument('--sigma', type=float, default=DEFAULT_SIGMA)
    p.add_argument('--amplitude', type=float, default=DEFAULT_AMPLITUDE)
    p.add_argument('--hole-center-x', type=float, default=DEFAULT_HOLE[0])
    p.add_argument('--hole-center-y', type=float, default=DEFAULT_HOLE[1])
    p.add_argument('--hole-radius', type=float, default=DEFAULT_HOLE[2])
    p.add_argument('--retrain', action='store_true')
    p.add_argument('--skip-train', action='store_true')
    args = p.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_train:
        train_routers(args)

    runs_dir = out_root / 'router_runs'
    hole = (args.hole_center_x, args.hole_center_y, args.hole_radius)
    X, Y, layout, _, _ = create_square_with_hole(
        Nx=args.nx, Ny=args.ny,
        hole_center=(hole[0], hole[1]), hole_radius=hole[2])

    with open(args.configs) as f:
        cfgs = json.load(f)
    entries = [('train', i, e) for i, e in enumerate(cfgs.get('train', []))]
    entries += [('test', i, e) for i, e in enumerate(cfgs.get('test', []))]

    all_summaries = []
    for split, idx, e in entries:
        name = e.get('name', f'{split}_{idx}')
        label = f"{split}_{idx}_{name}"
        out_dir = out_root / label
        print(f"\n=== {label} ===")
        if not os.path.exists(e['pinn_path']):
            print(f"  SKIP: missing PINN at {e['pinn_path']}")
            continue
        cfg_data = prepare_config(e, X, Y, layout, hole,
                                  args.sigma, args.amplitude, args.history_dir)
        summary = evaluate_config(cfg_data, args, runs_dir,
                                  X, Y, layout, hole, out_dir, label)
        all_summaries.append(summary)

    import csv
    csv_path = out_root / 'summary.csv'
    keys = ['label', 'name',
            'abstention_loss_mean', 'abstention_loss_std',
            'fem_coverage_mean', 'fem_coverage_std',
            'fem_time_mean', 'fem_time_std',
            'hybrid_time_mean', 'hybrid_time_std',
            'rmse_mean', 'rmse_std']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for s in all_summaries:
            w.writerow({k: s[k] for k in keys})
    with open(out_root / 'summary.json', 'w') as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nWrote {csv_path}")


if __name__ == '__main__':
    main()
