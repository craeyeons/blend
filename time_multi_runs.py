#!/usr/bin/env python3
"""Multi-config NSE timing + evaluation across 3 router retrainings.

Pipeline:
    1. For seed s in {0,1,2}: train router (subprocess into train_router_multi.py)
       writing weights to <output-dir>/router_runs/run_<s>/.
    2. For each train+test config in configs_multi_cylinder.json:
         a. Solve CFD 3x; keep one solution for RMSE.
         b. For each of the 3 trained routers:
              - compute hybrid solution; time it once.
              - true abstention loss at optimal threshold.
              - CFD coverage % at optimal threshold.
              - RMSE vs CFD.
         c. Aggregate mean/std over the 3 router runs.
    3. Per config: dump summary JSON + 2 plots
       (regions_and_errors.pdf, solution_comparison.pdf) using run 0.
    4. Top-level summary CSV across all configs.

Usage:
    python time_multi_runs.py --config configs_multi_cylinder.json \
        --output-dir ./time_multi_runs --epochs 500 --beta 1.0 --lambda-tv 0.1

Skip retraining (reuse existing router_runs/run_*/router.weights.h5):
    --skip-train
"""
import argparse
import contextlib
import io
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
from scipy.ndimage import binary_erosion, binary_dilation

from lib.router import (
    RouterCNN, PINNResidualComputer, compute_pinn_residual_field,
    create_router_input, compute_bc_error_field, solve_error_transport,
    create_cylinder_setup,
)
from lib.cylinder_flow import CylinderFlowSimulation, CylinderFlowHybridSimulation
from cylinder_network import Network as CylinderNetwork
from plot_coverage_metrics import (
    plot_solution_comparison, plot_regions_and_errors, compute_l2_error_field,
)


N_SEEDS = 3
N_CFD_RUNS = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gauge_free_rmse(u_p, v_p, p_p, u_r, v_r, p_r, valid, dx, dy, gradp_scale):
    px, py = np.gradient(p_p, dy, dx)
    pxr, pyr = np.gradient(p_r, dy, dx)
    err = ((u_p - u_r) ** 2 + (v_p - v_r) ** 2
           + ((px - pxr) ** 2 + (py - pyr) ** 2) / gradp_scale)
    return float(np.sqrt(np.mean(err[valid])))


def _interface_ring(mask, iters=1):
    m = mask.astype(bool)
    return binary_dilation(m, iterations=iters) & ~binary_erosion(m, iterations=iters)


def _find_optimal(residual_field, router_output, layout, beta, n_points=200):
    fluid = layout > 0
    R = residual_field[fluid]
    s = router_output[fluid]
    n = len(R)
    order = np.argsort(s)[::-1]
    R_sorted = R[order]
    s_sorted = s[order]
    cov = np.linspace(0, 1, n_points)
    loss = np.zeros(n_points)
    for i, c in enumerate(cov):
        k = int(c * n)
        rl = (1 - c) * np.mean(R_sorted[k:]) if (n - k) > 0 else 0.0
        loss[i] = beta * c + rl
    j = int(np.argmin(loss))
    c_opt = cov[j]
    k = int(c_opt * n)
    if k == 0:
        thresh = s_sorted[0] + 0.001
    elif k >= n:
        thresh = s_sorted[-1] - 0.001
    else:
        thresh = s_sorted[k - 1]
    return float(thresh), float(c_opt), float(loss[j])


def _load_pinn(pinn_path):
    net = CylinderNetwork()
    m = net.build(num_inputs=2, layers=[48, 48, 48, 48],
                  activation='tanh', num_outputs=3)
    m.load_weights(pinn_path)
    return m


def _setup(cfg, args):
    cx = cfg.get('cylinder_x', 0.5)
    cy = cfg.get('cylinder_y', 0.5)
    cr = cfg.get('cylinder_radius', 0.1)
    u_inlet = cfg.get('inlet_velocity', 1.0)
    pinn_path = cfg['pinn_path']

    X, Y, layout, bc_mask, bc_u, bc_v, bc_p = create_cylinder_setup(
        Nx=args.nx, Ny=args.ny,
        x_domain=(args.x_min, args.x_max),
        y_domain=(args.y_min, args.y_max),
        cylinder_center=(cx, cy),
        cylinder_radius=cr,
        inlet_velocity=u_inlet,
    )
    pinn = _load_pinn(pinn_path)
    xy = np.stack([X.flatten(), Y.flatten()], -1).astype(np.float32)
    uvp = pinn.predict(xy, batch_size=len(xy), verbose=0)
    pu = uvp[:, 0].reshape(X.shape).astype(np.float32) * layout
    pv = uvp[:, 1].reshape(X.shape).astype(np.float32) * layout
    pp = uvp[:, 2].reshape(X.shape).astype(np.float32) * layout

    bc_err = compute_bc_error_field(bc_mask, bc_u, bc_v, pu, pv, layout)
    nu = 1.0 / args.Re
    ete = solve_error_transport(pu, pv, bc_err, layout, nu=nu,
                                x_domain=(args.x_min, args.x_max),
                                y_domain=(args.y_min, args.y_max))
    res = compute_pinn_residual_field(pinn, X, Y, layout, bc_mask, bc_u, bc_v, nu=nu)
    inputs = create_router_input(layout, bc_mask, bc_u, bc_v, bc_p,
                                 pu, pv, pp, ete, res)

    rcomp = PINNResidualComputer(pinn, nu=nu, rho=1.0)
    Xt = tf.constant(X, dtype=tf.float32)
    Yt = tf.constant(Y, dtype=tf.float32)
    pde = rcomp.compute_total_residual_with_bc(
        Xt, Yt, tf.constant(bc_mask, dtype=tf.float32),
        tf.constant(bc_u, dtype=tf.float32),
        tf.constant(bc_v, dtype=tf.float32),
        {'continuity': 1.0, 'momentum': 1.0},
    ).numpy() * layout
    rfield = pde + ete
    med = np.median(rfield[layout > 0])
    if med > 1e-10:
        rfield = rfield / med

    return dict(
        X=X, Y=Y, layout=layout, bc_mask=bc_mask,
        bc_u=bc_u, bc_v=bc_v, bc_p=bc_p,
        pinn=pinn, pu=pu, pv=pv, pp=pp,
        inputs=tf.constant(inputs, dtype=tf.float32),
        residual_field=rfield,
        cx=cx, cy=cy, cr=cr, u_inlet=u_inlet, nu=nu,
    )


def _make_cfd(setup, args):
    return CylinderFlowSimulation(
        Re=args.Re, N=args.ny, max_iter=args.max_iter, tol=args.tol,
        x_domain=(args.x_min, args.x_max),
        y_domain=(args.y_min, args.y_max),
        cylinder_center=(setup['cx'], setup['cy']),
        cylinder_radius=setup['cr'],
        inlet_velocity=setup['u_inlet'],
    )


def _make_hybrid(setup, mask, args):
    def uv_func(net, xy):
        uvp = net.predict(xy, batch_size=len(xy), verbose=0)
        return uvp[..., 0], uvp[..., 1]
    return CylinderFlowHybridSimulation(
        network=setup['pinn'], uv_func=uv_func, mask=mask,
        Re=args.Re, N=args.ny, max_iter=args.max_iter, tol=args.tol,
        x_domain=(args.x_min, args.x_max),
        y_domain=(args.y_min, args.y_max),
        cylinder_center=(setup['cx'], setup['cy']),
        cylinder_radius=setup['cr'],
        inlet_velocity=setup['u_inlet'],
    )


def _load_router(weights_path, setup, args):
    r = RouterCNN(base_filters=args.base_filters, temperature=0.5)
    _ = r(setup['inputs'])
    r.load_weights(weights_path)
    return r


def _router_output(router, setup):
    return router(setup['inputs'], training=False)[0, :, :, 0].numpy()


# ---------------------------------------------------------------------------
# Training (subprocess)
# ---------------------------------------------------------------------------

def train_routers(args):
    runs_dir = Path(args.output_dir) / 'router_runs'
    runs_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parent / 'train_router_multi.py'
    for s in range(N_SEEDS):
        out = runs_dir / f'run_{s}'
        weights = out / 'router.weights.h5'
        if weights.exists() and not args.retrain:
            print(f"[run {s}] weights exist, skipping training.")
            continue
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(script),
            '--config', args.config,
            '--output-dir', str(out),
            '--epochs', str(args.epochs),
            '--beta', str(args.beta),
            '--lambda-tv', str(args.lambda_tv),
            '--lr', str(args.lr),
            '--nx', str(args.nx), '--ny', str(args.ny),
            '--x-min', str(args.x_min), '--x-max', str(args.x_max),
            '--y-min', str(args.y_min), '--y-max', str(args.y_max),
            '--nu', str(1.0 / args.Re), '--rho', '1.0',
            '--base-filters', str(args.base_filters),
            '--seed', str(s),
        ]
        print(f"[run {s}] training: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Per-config evaluation
# ---------------------------------------------------------------------------

def evaluate_config(setup, args, runs_dir, label, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- CFD x N_CFD_RUNS ----
    cfd_times = []
    u_cfd = v_cfd = p_cfd = None
    for i in range(N_CFD_RUNS):
        sim = _make_cfd(setup, args)
        with contextlib.redirect_stdout(io.StringIO()):
            t0 = time.perf_counter()
            u, v, p = sim.solve()
            t1 = time.perf_counter()
        cfd_times.append(t1 - t0)
        u_cfd, v_cfd, p_cfd = np.array(u), np.array(v), np.array(p)
        print(f"  [{label}] CFD {i+1}/{N_CFD_RUNS}: {cfd_times[-1]:.3f}s")

    layout = setup['layout']
    fluid = layout > 0
    dx = float(setup['X'][0, 1] - setup['X'][0, 0])
    dy = float(setup['Y'][1, 0] - setup['Y'][0, 0])
    pxc, pyc = np.gradient(p_cfd, dy, dx)
    gradp_scale = float(np.max(pxc[fluid]**2 + pyc[fluid]**2)) + 1e-10
    interior = binary_erosion(fluid, iterations=1)

    # Residual field for abstention-loss optimum
    rfield = setup['residual_field']

    abst_losses, covs, hyb_times, rmses = [], [], [], []
    plot_run_idx = 0  # use run 0 for the per-config plots
    plot_payload = None

    for s in range(N_SEEDS):
        weights = runs_dir / f'run_{s}' / 'router.weights.h5'
        router = _load_router(str(weights), setup, args)
        s_field = _router_output(router, setup)

        thresh, c_opt, abst_loss = _find_optimal(rfield, s_field, layout, args.beta)
        cfd_mask = (s_field >= thresh).astype(np.int32) * layout.astype(np.int32)
        actual_cov = float(np.sum(cfd_mask) / np.sum(layout))

        sim = _make_hybrid(setup, cfd_mask, args)
        with contextlib.redirect_stdout(io.StringIO()):
            t0 = time.perf_counter()
            uh, vh, ph = sim.solve()
            t1 = time.perf_counter()
        hyb_times.append(t1 - t0)
        uh, vh, ph = np.array(uh), np.array(vh), np.array(ph)

        ring = _interface_ring(cfd_mask.astype(bool) & fluid, iters=1)
        valid = interior & ~ring
        rmse = _gauge_free_rmse(uh, vh, ph, u_cfd, v_cfd, p_cfd,
                                valid, dx, dy, gradp_scale)

        abst_losses.append(abst_loss)
        covs.append(actual_cov)
        rmses.append(rmse)

        print(f"  [{label}] run {s}: cov={actual_cov*100:.2f}% "
              f"abst_loss={abst_loss:.4f} hyb_time={hyb_times[-1]:.3f}s "
              f"RMSE={rmse:.4f}")

        if s == plot_run_idx:
            plot_payload = dict(
                cfd_mask=cfd_mask, uh=uh, vh=vh, ph=ph,
                u_cfd=u_cfd, v_cfd=v_cfd, p_cfd=p_cfd,
            )

    summary = {
        'label': label,
        'cfd_time_mean': float(np.mean(cfd_times)),
        'cfd_time_std': float(np.std(cfd_times)),
        'cfd_times': cfd_times,
        'hybrid_time_mean': float(np.mean(hyb_times)),
        'hybrid_time_std': float(np.std(hyb_times)),
        'hybrid_times': hyb_times,
        'abstention_loss_mean': float(np.mean(abst_losses)),
        'abstention_loss_std': float(np.std(abst_losses)),
        'abstention_losses': abst_losses,
        'cfd_coverage_mean': float(np.mean(covs)),
        'cfd_coverage_std': float(np.std(covs)),
        'cfd_coverages': covs,
        'rmse_mean': float(np.mean(rmses)),
        'rmse_std': float(np.std(rmses)),
        'rmses': rmses,
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Plots — use plot run 0
    p = plot_payload
    plot_regions_and_errors(
        setup['pu'], setup['pv'], setup['pp'],
        p['uh'], p['vh'], p['ph'],
        p['u_cfd'], p['v_cfd'], p['p_cfd'],
        setup['X'], setup['Y'], setup['layout'], p['cfd_mask'],
        (setup['cx'], setup['cy']), setup['cr'],
        save_path=str(out_dir / 'regions_and_errors.pdf'),
    )
    plot_solution_comparison(
        setup['pu'], setup['pv'], setup['pp'],
        p['u_cfd'], p['v_cfd'], p['p_cfd'],
        p['uh'], p['vh'], p['ph'],
        setup['X'], setup['Y'], setup['layout'], p['cfd_mask'],
        (setup['cx'], setup['cy']), setup['cr'],
        save_path=str(out_dir / 'solution_comparison.pdf'),
    )
    return summary


# ---------------------------------------------------------------------------
# Coverage sweep (accuracy + time vs coverage, per seed)
# ---------------------------------------------------------------------------

def _plot_sweep(rows, cfd_time_mean, label, save_path):
    """RMSE-vs-time tradeoff, one line per seed (raw points, no aggregation).
    Both axes vary per seed, so seeds are drawn as separate curves rather
    than a band — replot from sweep.csv if you want a different aggregation."""
    seeds = sorted({r['seed'] for r in rows})
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for s in seeds:
        sr = sorted((r for r in rows if r['seed'] == s),
                    key=lambda r: r['actual_coverage'])
        t = [r['hybrid_time'] for r in sr]
        rmse = [r['rmse'] for r in sr]
        ax.plot(t, rmse, '-o', label=f"seed {s}", zorder=2)
    ax.axvline(cfd_time_mean, ls='--', color='gray', lw=1,
               label=f"full CFD ({cfd_time_mean:.1f}s)")
    ax.set_xlabel("hybrid solve time (s)")
    ax.set_ylabel("RMSE vs CFD")
    ax.set_title(label)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def sweep_config(setup, args, runs_dir, label, out_dir, coverages):
    """For one config: sweep CFD coverage in fixed increments across all
    seeds, recording the raw (RMSE, hybrid time) at each coverage level for
    each seed.

    Coverage c selects the top-c fraction of fluid cells by router score as
    the CFD region (exact cell count, tie-safe). c=0 is pure PINN, c=1 is
    full CFD. Both RMSE and time vary per seed (placement affects solver
    conditioning), so every raw per-seed point is written to sweep.csv for
    downstream replotting/aggregation."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- CFD reference x N_CFD_RUNS (for RMSE target + timing baseline) ----
    cfd_times = []
    u_cfd = v_cfd = p_cfd = None
    for i in range(N_CFD_RUNS):
        sim = _make_cfd(setup, args)
        with contextlib.redirect_stdout(io.StringIO()):
            t0 = time.perf_counter()
            u, v, p = sim.solve()
            t1 = time.perf_counter()
        cfd_times.append(t1 - t0)
        u_cfd, v_cfd, p_cfd = np.array(u), np.array(v), np.array(p)
        print(f"  [{label}] CFD {i+1}/{N_CFD_RUNS}: {cfd_times[-1]:.3f}s")
    cfd_time_mean = float(np.mean(cfd_times))

    layout = setup['layout']
    fluid = layout > 0
    fluid_idx = np.argwhere(fluid)
    n_fluid = len(fluid_idx)
    dx = float(setup['X'][0, 1] - setup['X'][0, 0])
    dy = float(setup['Y'][1, 0] - setup['Y'][0, 0])
    pxc, pyc = np.gradient(p_cfd, dy, dx)
    gradp_scale = float(np.max(pxc[fluid]**2 + pyc[fluid]**2)) + 1e-10
    interior = binary_erosion(fluid, iterations=1)

    rows = []
    for s in args.sweep_seeds:
        weights = runs_dir / f'run_{s}' / 'router.weights.h5'
        router = _load_router(str(weights), setup, args)
        s_field = _router_output(router, setup)
        # rank fluid cells by router score, highest first
        order = np.argsort(s_field[fluid])[::-1]
        ranked = fluid_idx[order]

        for cov in coverages:
            k = int(round(cov * n_fluid))
            k = max(0, min(n_fluid, k))
            cfd_mask = np.zeros_like(layout, dtype=np.int32)
            if k > 0:
                sel = ranked[:k]
                cfd_mask[sel[:, 0], sel[:, 1]] = 1
            actual_cov = float(k / n_fluid)

            sim = _make_hybrid(setup, cfd_mask, args)
            with contextlib.redirect_stdout(io.StringIO()):
                t0 = time.perf_counter()
                uh, vh, ph = sim.solve()
                t1 = time.perf_counter()
            hyb_time = t1 - t0
            uh, vh, ph = np.array(uh), np.array(vh), np.array(ph)

            ring = _interface_ring(cfd_mask.astype(bool) & fluid, iters=1)
            valid = interior & ~ring
            rmse = _gauge_free_rmse(uh, vh, ph, u_cfd, v_cfd, p_cfd,
                                    valid, dx, dy, gradp_scale)

            rows.append({
                'seed': s,
                'target_coverage': float(cov),
                'actual_coverage': actual_cov,
                'rmse': rmse,
                'hybrid_time': hyb_time,
                'cfd_time_mean': cfd_time_mean,
            })
            print(f"  [{label}] seed {s} cov={actual_cov*100:5.1f}% "
                  f"RMSE={rmse:.4f} hyb_time={hyb_time:.3f}s")

    import csv
    csv_path = out_dir / 'sweep.csv'
    keys = ['seed', 'target_coverage', 'actual_coverage',
            'rmse', 'hybrid_time', 'cfd_time_mean']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / 'sweep.json', 'w') as f:
        json.dump({'label': label, 'rows': rows}, f, indent=2)
    _plot_sweep(rows, cfd_time_mean, label, str(out_dir / 'sweep.pdf'))
    print(f"\nWrote {csv_path} and sweep.pdf")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output-dir', default='./time_multi_runs')
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--lambda-tv', type=float, default=1.0)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--nx', type=int, default=200)
    p.add_argument('--ny', type=int, default=100)
    p.add_argument('--x-min', type=float, default=0.0)
    p.add_argument('--x-max', type=float, default=2.0)
    p.add_argument('--y-min', type=float, default=0.0)
    p.add_argument('--y-max', type=float, default=1.0)
    p.add_argument('--Re', type=float, default=100.0)
    p.add_argument('--max-iter', type=int, default=200000)
    p.add_argument('--tol', type=float, default=1e-6)
    p.add_argument('--base-filters', type=int, default=32)
    p.add_argument('--retrain', action='store_true',
                   help='Force retraining even if weights exist.')
    p.add_argument('--skip-train', action='store_true',
                   help='Skip training step (reuse existing router_runs/).')
    p.add_argument('--sweep', action='store_true',
                   help='Coverage-sweep mode: for one config, record '
                        '(RMSE, hybrid time) per seed at fixed coverage steps.')
    p.add_argument('--sweep-role', default='test',
                   help='Config role to sweep (default: test).')
    p.add_argument('--sweep-idx', type=int, default=0,
                   help='Config index within the role to sweep (default: 0).')
    p.add_argument('--sweep-step', type=float, default=0.1,
                   help='Coverage increment for the sweep (default: 0.1).')
    p.add_argument('--sweep-extra-coverages', type=float, nargs='*', default=[],
                   help='Extra coverage fractions to merge into the grid '
                        '(e.g. 0.46 for the abstention-optimal point).')
    p.add_argument('--sweep-seeds', type=int, nargs='+', default=[0, 1, 2],
                   help='Router seeds to evaluate in sweep mode.')
    args = p.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_train:
        train_routers(args)

    runs_dir = out_root / 'router_runs'
    with open(args.config) as f:
        cfgs = json.load(f)

    if args.sweep:
        pool = cfgs.get(args.sweep_role, [])
        if not (0 <= args.sweep_idx < len(pool)):
            raise SystemExit(
                f"--sweep-idx {args.sweep_idx} out of range for role "
                f"'{args.sweep_role}' ({len(pool)} configs).")
        cfg = pool[args.sweep_idx]
        label = (f"{args.sweep_role}_{args.sweep_idx}_x{cfg.get('cylinder_x',0.5)}"
                 f"_y{cfg.get('cylinder_y',0.5)}_r{cfg.get('cylinder_radius',0.1)}"
                 f"_u{cfg.get('inlet_velocity',1.0)}_sweep")
        out_dir = out_root / label
        grid = np.arange(0.0, 1.0 + 1e-9, args.sweep_step)
        coverages = np.round(
            sorted(set(list(grid) + list(args.sweep_extra_coverages))), 6)
        print(f"\n=== SWEEP {label} | coverages={list(coverages)} ===")
        setup = _setup(cfg, args)
        sweep_config(setup, args, runs_dir, label, out_dir, coverages)
        return

    all_cfgs = [('train', i, c) for i, c in enumerate(cfgs.get('train', []))]
    all_cfgs += [('test', i, c) for i, c in enumerate(cfgs.get('test', []))]

    all_summaries = []
    for role, idx, cfg in all_cfgs:
        label = f"{role}_{idx}_x{cfg.get('cylinder_x',0.5)}_y{cfg.get('cylinder_y',0.5)}_r{cfg.get('cylinder_radius',0.1)}_u{cfg.get('inlet_velocity',1.0)}"
        out_dir = out_root / label
        print(f"\n=== {label} ===")
        setup = _setup(cfg, args)
        summary = evaluate_config(setup, args, runs_dir, label, out_dir)
        all_summaries.append(summary)

    # CSV aggregate
    import csv
    csv_path = out_root / 'summary.csv'
    keys = ['label',
            'abstention_loss_mean', 'abstention_loss_std',
            'cfd_coverage_mean', 'cfd_coverage_std',
            'cfd_time_mean', 'cfd_time_std',
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
