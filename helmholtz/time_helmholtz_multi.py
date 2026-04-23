"""
Multi-config timing + coverage sweep for Exp 3 parametric Helmholtz.

For every config in configs JSON: load FEM reference, build router inputs
via the parametric PINN, run a coverage sweep, and time the hybrid path.

Per config, saves:
    {output_dir}/sweep_{prefix}_{name}.json
    {output_dir}/reference_{prefix}_{name}.npz

Usage:
    python time_helmholtz_multi.py --pinn-tag exp3_parametric \
        --router-tag exp3 --configs configs_exp3.json --tag-prefix exp3
"""

import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

from lib.domains import create_square_with_hole, gaussian_source
from lib.fem_solver import HelmholtzSolver
from lib.network import build_parametric_pinn
from lib.router import (RouterCNN, create_router_input, compute_ete_fft)
from lib.hybrid import (solve_hybrid_schwarz, threshold_for_coverage, rmse)

from train_router_multi import ParametricResidualComputer


N_RUNS = 5


def _pinn_on_grid(pinn, X, Y, k, xs, ys, layout):
    xy = np.stack([X.ravel(), Y.ravel()], axis=-1).astype(np.float32)
    N = xy.shape[0]
    params = np.stack([np.full(N, k, dtype=np.float32),
                       np.full(N, xs, dtype=np.float32),
                       np.full(N, ys, dtype=np.float32)], axis=-1)
    xyp = np.concatenate([xy, params], axis=-1)
    u = pinn.predict(xyp, batch_size=len(xyp), verbose=0)
    return u.reshape(X.shape) * layout


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pinn-tag', type=str, required=True)
    p.add_argument('--router-tag', type=str, required=True)
    p.add_argument('--configs', type=str, required=True)
    p.add_argument('--tag-prefix', type=str, required=True)
    p.add_argument('--pinn-dir', type=str, default='./models')
    p.add_argument('--pinn-meta-dir', type=str, default='./history')
    p.add_argument('--router-dir', type=str, default='./router_models')
    p.add_argument('--fem-dir', type=str, default='./results')
    p.add_argument('--output-dir', type=str, default='./timing')
    p.add_argument('--mesh-n', type=int, default=129)
    p.add_argument('--nx', type=int, default=201)
    p.add_argument('--ny', type=int, default=201)
    p.add_argument('--base-filters', type=int, default=32)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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
    dummy = np.zeros((1, args.ny, args.nx, 5), dtype=np.float32)
    _ = router(tf.constant(dummy))
    router.load_weights(os.path.join(
        args.router_dir, f'router_helmholtz_{args.router_tag}.weights.h5'))

    with open(args.configs) as f:
        configs = json.load(f)

    for cfg in configs:
        name = cfg['name']
        k = float(cfg['k']); xs = float(cfg['x_s']); ys = float(cfg['y_s'])
        split = cfg.get('split', 'id')
        tag = f"{args.tag_prefix}_{name}"
        print(f"\n=== {name}  k={k:.4f}  x_s={xs:.3f}  y_s={ys:.3f}  "
              f"split={split} ===")

        # Load FEM reference.
        ref_path = os.path.join(args.fem_dir, f'fem_helmholtz_{tag}.npz')
        ref = np.load(ref_path)
        u_fem = ref['u_fem']

        # Config-specific solver (A depends on k).
        solver = HelmholtzSolver(k=k, mesh_n=args.mesh_n,
                                 dirichlet_predicate=dirichlet_predicate)

        def f_callable(x, y, xs_=xs, ys_=ys):
            return amplitude * np.exp(-((x - xs_) ** 2 + (y - ys_) ** 2)
                                      / (2.0 * sigma ** 2))
        def g_callable(x, y):
            return np.zeros_like(x)

        # PINN + residual + ETE channels (config-specific, router-input build).
        t_inf0 = time.perf_counter()
        pinn_u = _pinn_on_grid(pinn, X, Y, k, xs, ys, layout)
        t_pinn = time.perf_counter() - t_inf0

        rcomp = ParametricResidualComputer(pinn, k, xs, ys)
        f_grid = gaussian_source(X, Y, x_s=xs, y_s=ys,
                                 sigma=sigma, amplitude=amplitude)
        t0 = time.perf_counter()
        signed_r = rcomp.compute_signed_residual(X, Y, f_grid) * layout
        residual = np.abs(signed_r)
        t_res = time.perf_counter() - t0
        t0 = time.perf_counter()
        ete = compute_ete_fft(signed_r, k, layout=layout)
        t_ete = time.perf_counter() - t0

        inputs = create_router_input(layout, f_grid, pinn_u, residual, ete=ete)
        t0 = time.perf_counter()
        logits = router(tf.constant(inputs, dtype=tf.float32),
                        training=False)[0, :, :, 0].numpy()
        t_router = time.perf_counter() - t0

        # PINN-only RMSE vs FEM.
        pinn_rmse = rmse(pinn_u, u_fem)
        print(f"  PINN RMSE = {pinn_rmse:.3e}  "
              f"(pinn_grid={t_pinn:.3f}s  residual={t_res:.3f}s  "
              f"ete={t_ete:.3f}s  router={t_router:.3f}s)")

        # FEM baseline timing.
        fem_times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            u_dof, _ = solver.solve(f_callable, g_callable)
            u_grid = solver.interp_to_grid(u_dof, X, Y)
            fem_times.append(time.perf_counter() - t0)
        fem_mean = float(np.mean(fem_times))
        fem_std = float(np.std(fem_times))
        print(f"  FEM baseline {fem_mean*1000:.1f} +- {fem_std*1000:.1f} ms")

        # Coverage sweep.
        coverage_targets = np.linspace(0.0, 1.0, 11)
        sweep = []
        for cov in coverage_targets:
            thr = threshold_for_coverage(logits, layout, float(cov))
            if cov <= 0.0:
                err = pinn_rmse
                sweep.append({'target_coverage': 0.0,
                              'threshold': float(thr),
                              'actual_coverage_pct': 0.0,
                              'hybrid_total_s': 0.0,
                              'rmse_vs_fem': float(err),
                              'n_accepted_dofs': 0})
                continue
            if cov >= 1.0:
                sweep.append({'target_coverage': 1.0,
                              'threshold': float(thr),
                              'actual_coverage_pct': 100.0,
                              'hybrid_total_s': float(fem_mean),
                              'rmse_vs_fem': 0.0,
                              'n_accepted_dofs': 0})
                continue
            t0 = time.perf_counter()
            res = solve_hybrid_schwarz(
                solver, pinn, router, f_callable, g_callable,
                X, Y, layout, f_grid, pinn_u, residual,
                ete_grid=ete, threshold=float(thr), reuse_logits=logits)
            wall = time.perf_counter() - t0
            err = rmse(res['u_grid'], u_fem)
            sweep.append({'target_coverage': float(cov),
                          'threshold': float(thr),
                          'actual_coverage_pct': float(res['coverage_pct']),
                          'hybrid_total_s': float(wall),
                          'rmse_vs_fem': float(err),
                          'n_accepted_dofs': int(res['n_accepted_dofs'])})
            print(f"    target={cov*100:5.1f}% "
                  f"actual={res['coverage_pct']:5.1f}% "
                  f"wall={wall*1000:6.1f}ms  RMSE={err:.3e}")

        summary = {
            'name': name, 'split': split,
            'k': k, 'x_s': xs, 'y_s': ys,
            'sigma': sigma, 'amplitude': amplitude,
            'fem_mean_s': fem_mean, 'fem_std_s': fem_std,
            'pinn_grid_s': t_pinn, 'residual_s': t_res,
            'ete_s': t_ete, 'router_s': t_router,
            'pinn_rmse_vs_fem': pinn_rmse,
            'coverage_sweep': sweep,
        }
        with open(os.path.join(args.output_dir,
                               f'sweep_{tag}.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        np.savez(os.path.join(args.output_dir, f'reference_{tag}.npz'),
                 X=X, Y=Y, layout=layout, u_fem=u_fem,
                 pinn_u=pinn_u, residual=residual, ete=ete,
                 f_grid=f_grid, logits=logits)

    print("\nDone.")


if __name__ == '__main__':
    main()
