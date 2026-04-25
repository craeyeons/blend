"""
Train one shared router over N independent single-case Helmholtz PINNs
(Exp 2 setup, one per train config), then run full analysis on every
train + test config:

    - per-config FEM reference solve
    - PINN-only RMSE vs FEM
    - coverage sweep via Schwarz hybrid (RMSE + wall time at each coverage)
    - per-config plots: solution panel, rmse-vs-coverage, coverage grid
    - summary JSON across configs

Mirrors the CFD multi-cylinder pattern (`train_router_multi.py` at repo
root). Router input is the 5-channel Helmholtz layout:
    (layout, f, pinn_u, |r|/median, |e|/median)

Usage:
    python train_router_multi.py --configs configs_exp3.json --tag exp3 \\
        --epochs 2000 --lr 1e-3 --beta 1.0 --lambda-tv 1.0
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_gpus = tf.config.list_physical_devices('GPU')
if _gpus:
    try:
        for _gpu in _gpus:
            tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass

from lib.domains import create_square_with_hole, gaussian_source
from lib.fem_solver import HelmholtzSolver
from lib.network import build_pinn
from lib.router import (RouterCNN, HelmholtzResidualComputer,
                        create_router_input, compute_ete_fft,
                        median_normalize)
from lib.hybrid import solve_hybrid_schwarz, threshold_for_coverage, rmse


N_FEM_RUNS = 5
COVERAGE_DECILES = np.linspace(0.0, 1.0, 11)

DEFAULT_SIGMA = 0.05
DEFAULT_AMPLITUDE = 250.0
DEFAULT_HOLE = (0.5, 0.5, 0.15)
DEFAULT_LAYERS = [256, 256, 256, 256, 256]
DEFAULT_FOURIER_M = 128


def _meta_path(history_dir, pinn_path):
    """history/training_meta_<tag>.json from pinn_helmholtz_<tag>.weights.h5."""
    base = os.path.basename(pinn_path)
    prefix, suffix = 'pinn_helmholtz_', '.weights.h5'
    tag = base[len(prefix):-len(suffix)] if (base.startswith(prefix)
                                             and base.endswith(suffix)) else base
    return os.path.join(history_dir, f'training_meta_{tag}.json')


def _load_scalar_pinn(pinn_path, k, history_dir='./history'):
    """Load a single-case scalar PINN by reading its training meta (for
    layers / fourier hyperparams). Falls back to Exp-2 defaults if meta
    is missing."""
    mpath = _meta_path(history_dir, pinn_path)
    if os.path.exists(mpath):
        with open(mpath) as f:
            meta = json.load(f)
        layers = tuple(meta.get('layers', DEFAULT_LAYERS))
        fourier_m = int(meta.get('fourier_m', DEFAULT_FOURIER_M))
        fourier_scale = float(meta.get('fourier_scale', k / (2.0 * np.pi)))
    else:
        print(f"  (no meta at {mpath}; using defaults + fourier_scale=k/2pi)")
        layers = tuple(DEFAULT_LAYERS)
        fourier_m = DEFAULT_FOURIER_M
        fourier_scale = k / (2.0 * np.pi)

    model = build_pinn(num_inputs=2, layers=layers, activation='tanh',
                       input_range=((0.0, 1.0), (0.0, 1.0)),
                       fourier_m=fourier_m,
                       fourier_scale=fourier_scale)
    _ = model(tf.zeros((1, 2)))
    model.load_weights(pinn_path)
    return model


def _pinn_on_grid(pinn, X, Y, layout):
    xy = np.stack([X.ravel(), Y.ravel()], axis=-1).astype(np.float32)
    u = pinn.predict(xy, batch_size=len(xy), verbose=0)
    return u.reshape(X.shape).astype(np.float32) * layout


def prepare_config(entry, X, Y, layout, sigma, amplitude, history_dir):
    """Build per-config router inputs + training target + ancillary fields."""
    name = entry['name']
    k = float(entry['k']); xs = float(entry['x_s']); ys = float(entry['y_s'])
    print(f"  preparing {name}  k={k:.4f}  x_s={xs:.3f}  y_s={ys:.3f}")

    pinn = _load_scalar_pinn(entry['pinn_path'], k, history_dir=history_dir)

    f_grid = gaussian_source(X, Y, x_s=xs, y_s=ys,
                             sigma=sigma, amplitude=amplitude)
    pinn_u = _pinn_on_grid(pinn, X, Y, layout)

    rcomp = HelmholtzResidualComputer(pinn, k)
    signed_r = rcomp.compute_signed_residual(X, Y, f_grid) * layout
    residual = np.abs(signed_r)
    ete = compute_ete_fft(signed_r, k, layout=layout)

    inputs = create_router_input(layout, f_grid, pinn_u, residual, ete=ete)
    # R = normalize(|r| + e) — each already nonneg; median-normalize the sum.
    target_R = median_normalize(residual + ete, layout).astype(np.float32)

    return {
        'name': name, 'k': k, 'x_s': xs, 'y_s': ys,
        'pinn': pinn,
        'f_grid': f_grid, 'pinn_u': pinn_u,
        'signed_r': signed_r, 'residual': residual, 'ete': ete,
        'inputs': tf.constant(inputs, dtype=tf.float32),
        'target': tf.constant(target_R, dtype=tf.float32),
    }


def _tv(r4):
    tv_h = tf.reduce_mean(tf.abs(r4[:, :, 1:, :] - r4[:, :, :-1, :]))
    tv_v = tf.reduce_mean(tf.abs(r4[:, 1:, :, :] - r4[:, :-1, :, :]))
    return tv_h + tv_v


def make_train_step(router, optimizer, layout_tf, beta, lambda_tv,
                    grad_clip=1.0):
    @tf.function
    def step(inputs, target_R):
        with tf.GradientTape() as tape:
            s = router(inputs, training=True)[0, :, :, 0]
            num = tf.reduce_sum(layout_tf) + 1e-10
            logistic = tf.reduce_sum(
                (beta * tf.math.softplus(s)
                 + target_R * tf.math.softplus(-s)) * layout_tf) / num
            s4 = tf.reshape(s * layout_tf,
                            [1, tf.shape(s)[0], tf.shape(s)[1], 1])
            tv = lambda_tv * _tv(s4)
            total = logistic + tv
        grads = tape.gradient(total, router.trainable_variables)
        if grad_clip > 0:
            grads, _ = tf.clip_by_global_norm(grads, grad_clip)
        optimizer.apply_gradients(zip(grads, router.trainable_variables))
        rejected = tf.reduce_sum(
            tf.cast(s > 0, tf.float32) * layout_tf) / num
        return total, logistic, tv, rejected
    return step


def _plot_router_output(logits, X, Y, layout, hole, title, out_path):
    mask = layout > 0
    disp = np.where(mask, logits, np.nan)
    fig, ax = plt.subplots(figsize=(7, 6))
    vmax = float(np.nanmax(np.abs(disp))) + 1e-30
    im = ax.pcolormesh(X, Y, disp, cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, shading='auto')
    cx, cy, rh = hole
    circle = plt.Circle((cx, cy), rh, color='gray', fill=True, alpha=0.8)
    ax.add_patch(circle)
    ax.set_aspect('equal'); ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_coverage_grid(logits, X, Y, layout, hole, title, out_path):
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    axes = axes.flatten()
    solid = layout > 0
    for j, cov in enumerate(COVERAGE_DECILES):
        ax = axes[j]
        if cov <= 0.0:
            thr = float(logits[solid].max()) + 1.0
        elif cov >= 1.0:
            thr = float(logits[solid].min()) - 1.0
        else:
            fluid_logits = logits[solid]
            sorted_desc = np.sort(fluid_logits)[::-1]
            n = len(sorted_desc)
            k_cov = max(1, min(int(cov * n), n))
            thr = float(sorted_desc[k_cov - 1])
        combined = np.zeros_like(logits)
        combined[~solid] = 0
        combined[solid & (logits < thr)] = 1   # PINN
        combined[solid & (logits >= thr)] = 2  # FEM
        ax.contourf(X, Y, combined, levels=[-0.5, 0.5, 1.5, 2.5],
                    colors=['lightgray', 'tab:blue', 'tab:red'], alpha=0.75)
        cx, cy, rh = hole
        ax.add_patch(plt.Circle((cx, cy), rh, color='gray', fill=True))
        ax.set_aspect('equal')
        actual = 100.0 * np.sum(solid & (logits >= thr)) / max(np.sum(solid), 1)
        ax.set_title(f'{cov*100:.0f}% (act {actual:.0f}%)', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    axes[-1].axis('off')
    legend = [Patch(facecolor='lightgray', label='hole'),
              Patch(facecolor='tab:blue', alpha=0.75, label='PINN'),
              Patch(facecolor='tab:red',  alpha=0.75, label='FEM')]
    fig.legend(handles=legend, loc='lower right', fontsize=10)
    plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_solution(X, Y, layout, pinn_u, u_hybrid, u_fem, accept_mask,
                   title, out_path):
    mask = layout > 0
    def _m(a): return np.where(mask, a, np.nan)
    u_vmax = float(np.nanmax(np.abs(_m(u_fem)))) + 1e-30
    err_pinn = _m(pinn_u - u_fem); err_hyb = _m(u_hybrid - u_fem)
    e_vmax = float(max(np.nanmax(np.abs(err_pinn)),
                       np.nanmax(np.abs(err_hyb)))) + 1e-30
    reject_field = ((~accept_mask.astype(bool)) & mask).astype(np.float32)

    def _shade(ax):
        ax.contourf(X, Y, reject_field, levels=[0.5, 1.5],
                    colors=['black'], alpha=0.30)
        ax.contour(X, Y, reject_field, levels=[0.5],
                   colors='lime', linewidths=1.2)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    row0 = [_m(pinn_u), _m(u_hybrid), _m(u_fem)]
    ttl0 = ['PINN  u', 'Hybrid  u (shaded=FEM)', 'FEM  u']
    for j in range(3):
        ax = axes[0, j]
        im = ax.pcolormesh(X, Y, row0[j], cmap='RdBu_r',
                           vmin=-u_vmax, vmax=u_vmax, shading='auto')
        if j == 1: _shade(ax)
        ax.set_aspect('equal'); ax.set_title(ttl0[j])
        plt.colorbar(im, ax=ax, fraction=0.046)
    row1 = [err_pinn, err_hyb, _m(np.zeros_like(u_fem))]
    ttl1 = ['PINN − FEM', 'Hybrid − FEM', 'FEM − FEM']
    for j in range(3):
        ax = axes[1, j]
        im = ax.pcolormesh(X, Y, row1[j], cmap='RdBu_r',
                           vmin=-e_vmax, vmax=e_vmax, shading='auto')
        if j == 1: _shade(ax)
        ax.set_aspect('equal'); ax.set_title(ttl1[j])
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_rmse_vs_coverage(sweep, pinn_rmse, title, out_path, best=None):
    cov = np.array([s['actual_coverage_pct'] for s in sweep])
    err = np.array([s['rmse_vs_fem'] for s in sweep])
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(cov, err, 'o-', label='Hybrid RMSE')
    ax.axhline(pinn_rmse, ls='--', color='C3',
               label=f'PINN-only ({pinn_rmse:.2e})')
    if best is not None:
        ax.plot([best['actual_coverage_pct']], [best['rmse_vs_fem']],
                marker='*', markersize=20, markerfacecolor='gold',
                markeredgecolor='black', markeredgewidth=1.5,
                linestyle='none', zorder=5,
                label=f"Best hybrid (cov={best['actual_coverage_pct']:.1f}%, "
                      f"RMSE={best['rmse_vs_fem']:.2e})")
    ax.set_xlabel('FEM coverage (%)')
    ax.set_ylabel('RMSE vs full FEM')
    ax.set_yscale('log')
    ax.set_title(f'RMSE vs coverage — {title}')
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def run_analysis_for_config(cfg_data, split, hole, sigma, amplitude,
                            mesh_n, X, Y, layout, router,
                            tag_prefix, plots_dir, timing_dir):
    name = cfg_data['name']
    k = cfg_data['k']; xs = cfg_data['x_s']; ys = cfg_data['y_s']
    tag = f"{tag_prefix}_{split}_{name}"
    title = f'{name} ({split.upper()})  k={k:.3f}  x_s={xs:.3f}  y_s={ys:.3f}'
    print(f"\n--- analyze {split}:{name} ---")

    cx, cy, rh = hole
    def dirichlet_pred(x, y):
        return (x - cx) ** 2 + (y - cy) ** 2 <= rh ** 2
    def f_callable(x, y, xs_=xs, ys_=ys):
        return amplitude * np.exp(-((x - xs_) ** 2 + (y - ys_) ** 2)
                                  / (2.0 * sigma ** 2))
    def g_callable(x, y):
        return np.zeros_like(x)

    solver = HelmholtzSolver(k=k, mesh_n=mesh_n,
                             dirichlet_predicate=dirichlet_pred)
    # FEM reference + timing
    fem_times = []
    u_fem = None
    for i in range(N_FEM_RUNS):
        t0 = time.perf_counter()
        u_dof, _ = solver.solve(f_callable, g_callable)
        u_grid = solver.interp_to_grid(u_dof, X, Y)
        fem_times.append(time.perf_counter() - t0)
        if i == 0:
            u_fem = u_grid
    fem_mean = float(np.mean(fem_times))
    fem_std = float(np.std(fem_times))

    pinn_u = cfg_data['pinn_u']
    pinn_rmse = rmse(pinn_u, u_fem)
    print(f"  FEM baseline {fem_mean*1000:.1f}±{fem_std*1000:.1f} ms   "
          f"PINN RMSE={pinn_rmse:.3e}")

    # Router logits
    logits = router(cfg_data['inputs'], training=False)[0, :, :, 0].numpy()

    # Coverage sweep
    sweep = []
    for cov in COVERAGE_DECILES:
        thr = threshold_for_coverage(logits, layout, float(cov))
        if cov <= 0.0:
            sweep.append({'target_coverage': 0.0, 'threshold': float(thr),
                          'actual_coverage_pct': 0.0,
                          'hybrid_total_s': 0.0,
                          'rmse_vs_fem': float(pinn_rmse),
                          'n_accepted_dofs': 0})
            continue
        if cov >= 1.0:
            sweep.append({'target_coverage': 1.0, 'threshold': float(thr),
                          'actual_coverage_pct': 100.0,
                          'hybrid_total_s': float(fem_mean),
                          'rmse_vs_fem': 0.0,
                          'n_accepted_dofs': 0})
            continue
        t0 = time.perf_counter()
        res = solve_hybrid_schwarz(
            solver, cfg_data['pinn'], router, f_callable, g_callable,
            X, Y, layout, cfg_data['f_grid'], cfg_data['pinn_u'],
            cfg_data['residual'], ete_grid=cfg_data['ete'],
            threshold=float(thr), reuse_logits=logits)
        wall = time.perf_counter() - t0
        err = rmse(res['u_grid'], u_fem)
        sweep.append({'target_coverage': float(cov),
                      'threshold': float(thr),
                      'actual_coverage_pct': float(res['coverage_pct']),
                      'hybrid_total_s': float(wall),
                      'rmse_vs_fem': float(err),
                      'n_accepted_dofs': int(res['n_accepted_dofs'])})
        print(f"    tgt={cov*100:5.1f}%  act={res['coverage_pct']:5.1f}%  "
              f"wall={wall*1000:6.1f}ms  RMSE={err:.3e}")

    # Pick best hybrid (lowest RMSE, excluding coverage=1.0 which is full FEM).
    interior = [s for s in sweep
                if s['target_coverage'] not in (0.0, 1.0)]
    best = min(interior, key=lambda s: s['rmse_vs_fem']) if interior else sweep[0]
    # Re-solve at best threshold for the solution plot.
    thr = best['threshold']
    res_best = solve_hybrid_schwarz(
        solver, cfg_data['pinn'], router, f_callable, g_callable,
        X, Y, layout, cfg_data['f_grid'], cfg_data['pinn_u'],
        cfg_data['residual'], ete_grid=cfg_data['ete'],
        threshold=float(thr), reuse_logits=logits)

    # Plots
    _plot_router_output(logits, X, Y, layout, hole,
                        f'Router logits — {title}',
                        os.path.join(plots_dir, f'router_{tag}.png'))
    _plot_coverage_grid(logits, X, Y, layout, hole,
                        f'Coverage deciles — {title}',
                        os.path.join(plots_dir, f'coverage_grid_{tag}.png'))
    _plot_solution(X, Y, layout, pinn_u, res_best['u_grid'], u_fem,
                   res_best['accept_mask'],
                   f'{title}  |  cov={best["actual_coverage_pct"]:.1f}%  '
                   f'PINN={pinn_rmse:.2e}  Hybrid={best["rmse_vs_fem"]:.2e}',
                   os.path.join(plots_dir, f'solution_{tag}.png'))
    _plot_rmse_vs_coverage(sweep, pinn_rmse, title,
                           os.path.join(plots_dir, f'rmse_vs_coverage_{tag}.png'),
                           best=best)

    # Save sweep + reference arrays
    with open(os.path.join(timing_dir, f'sweep_{tag}.json'), 'w') as fp:
        json.dump({'name': name, 'split': split, 'tag': tag,
                   'k': k, 'x_s': xs, 'y_s': ys,
                   'fem_mean_s': fem_mean, 'fem_std_s': fem_std,
                   'pinn_rmse_vs_fem': float(pinn_rmse),
                   'best_hybrid': best,
                   'coverage_sweep': sweep}, fp, indent=2)
    np.savez(os.path.join(timing_dir, f'reference_{tag}.npz'),
             X=X, Y=Y, layout=layout, u_fem=u_fem, pinn_u=pinn_u,
             residual=cfg_data['residual'], ete=cfg_data['ete'],
             f_grid=cfg_data['f_grid'], logits=logits)

    return {'name': name, 'split': split, 'tag': tag,
            'k': k, 'x_s': xs, 'y_s': ys,
            'pinn_rmse': float(pinn_rmse),
            'fem_mean_ms': fem_mean * 1000,
            'best_hybrid_rmse': float(best['rmse_vs_fem']),
            'best_hybrid_cov_pct': float(best['actual_coverage_pct']),
            'best_hybrid_ms': float(best['hybrid_total_s']) * 1000,
            'speedup_at_best': float(fem_mean / max(best['hybrid_total_s'], 1e-12))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--configs', type=str, required=True)
    p.add_argument('--tag', type=str, required=True)
    p.add_argument('--output-dir', type=str, default='./router_output_multi')
    p.add_argument('--plots-dir', type=str, default=None)
    p.add_argument('--timing-dir', type=str, default=None)
    p.add_argument('--history-dir', type=str, default='./history')
    p.add_argument('--epochs', type=int, default=2000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr-min', type=float, default=1e-5)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--lambda-tv', type=float, default=1.0)
    p.add_argument('--base-filters', type=int, default=32)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--mesh-n', type=int, default=129)
    p.add_argument('--nx', type=int, default=201)
    p.add_argument('--ny', type=int, default=201)
    p.add_argument('--sigma', type=float, default=DEFAULT_SIGMA)
    p.add_argument('--amplitude', type=float, default=DEFAULT_AMPLITUDE)
    p.add_argument('--hole-center-x', type=float, default=DEFAULT_HOLE[0])
    p.add_argument('--hole-center-y', type=float, default=DEFAULT_HOLE[1])
    p.add_argument('--hole-radius', type=float, default=DEFAULT_HOLE[2])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--skip-train', action='store_true',
                   help='Skip router training; load existing weights at '
                        '<output-dir>/router_helmholtz_<tag>.weights.h5 and '
                        'go straight to analysis.')
    p.add_argument('--skip-analysis', action='store_true',
                   help='Train router and exit; skip per-config FEM + '
                        'coverage sweep + plots.')
    args = p.parse_args()

    out_dir = args.output_dir
    plots_dir = args.plots_dir or os.path.join(out_dir, 'plots')
    timing_dir = args.timing_dir or os.path.join(out_dir, 'timing')
    for d in (out_dir, plots_dir, timing_dir):
        os.makedirs(d, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    tf.random.set_seed(args.seed)

    hole = (args.hole_center_x, args.hole_center_y, args.hole_radius)
    X, Y, layout, _, _ = create_square_with_hole(
        Nx=args.nx, Ny=args.ny,
        hole_center=(hole[0], hole[1]), hole_radius=hole[2])
    layout_tf = tf.constant(layout.astype(np.float32))

    with open(args.configs) as f:
        cfg = json.load(f)
    train_entries = cfg['train']
    test_entries = cfg.get('test', [])

    print(f"\n[1/5] Preparing {len(train_entries)} train + "
          f"{len(test_entries)} test configs")
    train_data = []
    for e in train_entries:
        train_data.append(prepare_config(e, X, Y, layout,
                                         args.sigma, args.amplitude,
                                         args.history_dir))
    test_data = []
    for e in test_entries:
        if not os.path.exists(e['pinn_path']):
            print(f"  SKIP test {e['name']}: no weights at {e['pinn_path']}")
            continue
        test_data.append(prepare_config(e, X, Y, layout,
                                        args.sigma, args.amplitude,
                                        args.history_dir))

    # Router + optimizer
    print(f"\n[2/5] Initializing router (base_filters={args.base_filters})")
    router = RouterCNN(base_filters=args.base_filters)
    _ = router(train_data[0]['inputs'])
    print(f"  Parameters: {router.count_params():,}")
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    beta = tf.constant(args.beta, dtype=tf.float32)
    ltv = tf.constant(args.lambda_tv, dtype=tf.float32)
    step_fn = make_train_step(router, optimizer, layout_tf,
                              beta, ltv, grad_clip=args.grad_clip)

    router_path = os.path.join(out_dir, f'router_helmholtz_{args.tag}.weights.h5')

    if args.skip_train:
        print(f"\n[3/5] --skip-train: loading existing router from {router_path}")
        if not os.path.exists(router_path):
            raise FileNotFoundError(
                f"--skip-train set but no router weights at {router_path}")
        router.load_weights(router_path)
        train_time_s = 0.0
        history = {'total': [], 'logistic': [], 'tv': [], 'reject': [], 'lr': []}
        # Skip the training loop body entirely.
        for_loop_count = 0
    else:
        for_loop_count = args.epochs

    # Training loop (cycle configs; cosine LR)
    print(f"\n[3/5] Training router for {for_loop_count} epochs "
          f"on {len(train_data)} configs")
    t0 = time.perf_counter()
    if not args.skip_train:
        history = {'total': [], 'logistic': [], 'tv': [], 'reject': [], 'lr': []}
    for epoch in range(for_loop_count):
        progress = epoch / max(args.epochs - 1, 1)
        cur_lr = args.lr_min + 0.5 * (args.lr - args.lr_min) * (
            1 + np.cos(np.pi * progress))
        optimizer.learning_rate.assign(cur_lr)

        order = rng.permutation(len(train_data))
        ep_totals = []
        last_log = last_tv = last_rej = 0.0
        for idx in order:
            d = train_data[idx]
            total, logistic, tv, rej = step_fn(d['inputs'], d['target'])
            ep_totals.append(float(total))
            last_log = float(logistic); last_tv = float(tv); last_rej = float(rej)
        history['total'].append(float(np.mean(ep_totals)))
        history['logistic'].append(last_log)
        history['tv'].append(last_tv)
        history['reject'].append(last_rej)
        history['lr'].append(float(cur_lr))
        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1}/{args.epochs}  "
                  f"mean_L={history['total'][-1]:.4f}  "
                  f"log={last_log:.4f}  tv={last_tv:.4f}  "
                  f"rej={last_rej*100:.1f}%  lr={cur_lr:.2e}")
    if not args.skip_train:
        train_time_s = time.perf_counter() - t0
        print(f"Router trained in {train_time_s:.1f}s")
        router.save_weights(router_path)
        np.savez(os.path.join(out_dir, f'router_history_{args.tag}.npz'),
                 **{k: np.array(v) for k, v in history.items()})
        meta_out = {
            'tag': args.tag, 'configs': args.configs,
            'n_train_configs': len(train_data),
            'n_test_configs': len(test_data),
            'epochs': args.epochs, 'lr': args.lr, 'lr_min': args.lr_min,
            'beta': float(args.beta), 'lambda_tv': float(args.lambda_tv),
            'base_filters': args.base_filters, 'grad_clip': args.grad_clip,
            'mesh_n': args.mesh_n, 'nx': args.nx, 'ny': args.ny,
            'sigma': args.sigma, 'amplitude': args.amplitude,
            'hole_center_x': args.hole_center_x,
            'hole_center_y': args.hole_center_y,
            'hole_radius': args.hole_radius,
            'train_time_s': train_time_s,
            'timestamp': datetime.now().isoformat(),
        }
        with open(os.path.join(out_dir, f'router_meta_{args.tag}.json'), 'w') as f:
            json.dump(meta_out, f, indent=2)
        print(f"  saved router weights: {router_path}")

        # Quick loss-history plot
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(history['total'], label='total')
        ax.plot(history['logistic'], label='logistic', alpha=0.7)
        ax.plot(history['tv'], label='tv', alpha=0.7)
        ax.set_xlabel('epoch'); ax.set_ylabel('loss'); ax.legend()
        ax.set_title(f'Router training — {args.tag}')
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f'training_history_{args.tag}.png'),
                    dpi=150)
        plt.close(fig)

    if args.skip_analysis:
        print("\n[4/5] --skip-analysis: stopping after router training.")
        return

    # Full analysis on train + test
    print(f"\n[4/5] Running analysis (FEM ref + coverage sweep + plots)")
    summary_rows = []
    for d in train_data:
        summary_rows.append(run_analysis_for_config(
            d, 'train', hole, args.sigma, args.amplitude, args.mesh_n,
            X, Y, layout, router, args.tag, plots_dir, timing_dir))
    for d in test_data:
        summary_rows.append(run_analysis_for_config(
            d, 'test', hole, args.sigma, args.amplitude, args.mesh_n,
            X, Y, layout, router, args.tag, plots_dir, timing_dir))

    # Summary CSV + Markdown + JSON
    print(f"\n[5/5] Writing summary")
    fields = ['name', 'split', 'k', 'x_s', 'y_s',
              'pinn_rmse', 'best_hybrid_rmse', 'best_hybrid_cov_pct',
              'fem_mean_ms', 'best_hybrid_ms', 'speedup_at_best']
    csv_path = os.path.join(out_dir, f'summary_{args.tag}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: row[k] for k in fields})

    md_path = os.path.join(out_dir, f'summary_{args.tag}.md')
    with open(md_path, 'w') as f:
        f.write(f'# Exp 3 summary — {args.tag}\n\n')
        f.write('| name | split | k | x_s | y_s | PINN RMSE | Hybrid RMSE | '
                'Cov% | FEM ms | Hybrid ms | Speedup |\n')
        f.write('|---|---|---|---|---|---|---|---|---|---|---|\n')
        for r in summary_rows:
            f.write(f"| {r['name']} | {r['split']} | {r['k']:.3f} | "
                    f"{r['x_s']:.3f} | {r['y_s']:.3f} | "
                    f"{r['pinn_rmse']:.2e} | {r['best_hybrid_rmse']:.2e} | "
                    f"{r['best_hybrid_cov_pct']:.1f} | "
                    f"{r['fem_mean_ms']:.1f} | {r['best_hybrid_ms']:.1f} | "
                    f"{r['speedup_at_best']:.2f}x |\n")

    with open(os.path.join(out_dir, f'summary_{args.tag}.json'), 'w') as f:
        json.dump({'tag': args.tag, 'entries': summary_rows}, f, indent=2)

    print(f"\nDone. Results under {out_dir}/")
    print(f"  weights:    router_helmholtz_{args.tag}.weights.h5")
    print(f"  summaries:  {csv_path}, {md_path}")
    print(f"  plots:      {plots_dir}/")
    print(f"  timing:     {timing_dir}/")


if __name__ == '__main__':
    main()
