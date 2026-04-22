"""
Time PINN+FEM hybrid vs full-FEM baseline for Helmholtz Experiment 2.

Measures:
  * full-FEM baseline solve time (N_RUNS=10)
  * hybrid solve time at threshold 0 (N_RUNS=10), broken into
    PINN inference / router inference / FEM spsolve
  * coverage sweep: 11 rejection percentages in [0, 100], run once each,
    record (coverage%, hybrid solve time, rel L2 vs full-FEM)

Usage:
    python time_helmholtz.py --k 12.566 --tag exp2_hole_k4pi
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

from lib.domains import (create_square, create_square_with_hole,
                         gaussian_source, manufactured_solution)
from lib.network import build_pinn
from lib.fem_solver import HelmholtzSolver
from lib.router import (RouterCNN, HelmholtzResidualComputer,
                        create_router_input)
from lib.hybrid import (solve_hybrid_schwarz, threshold_for_coverage,
                        rel_l2)


N_RUNS = 10
N_WARMUP = 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--k', type=float, required=True)
    p.add_argument('--tag', type=str, required=True)
    p.add_argument('--domain', type=str, default='square_hole',
                   choices=['square', 'square_hole'])
    p.add_argument('--source', type=str, default='gaussian',
                   choices=['manufactured', 'gaussian'])
    p.add_argument('--x-s', type=float, default=0.7)
    p.add_argument('--y-s', type=float, default=0.5)
    p.add_argument('--sigma', type=float, default=0.05)
    p.add_argument('--amplitude', type=float, default=1.0)
    p.add_argument('--hole-center-x', type=float, default=0.5)
    p.add_argument('--hole-center-y', type=float, default=0.5)
    p.add_argument('--hole-radius', type=float, default=0.15)
    p.add_argument('--nx', type=int, default=201)
    p.add_argument('--ny', type=int, default=201)
    p.add_argument('--mesh-n', type=int, default=129)
    p.add_argument('--pinn-dir', type=str, default='./models')
    p.add_argument('--pinn-meta-dir', type=str, default='./history')
    p.add_argument('--router-dir', type=str, default='./router_models')
    p.add_argument('--results-dir', type=str, default='./results')
    p.add_argument('--output-dir', type=str, default='./timing')
    p.add_argument('--base-filters', type=int, default=32)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"TIMING  tag={args.tag}  k={args.k:.4f}  N_RUNS={N_RUNS}")
    print("=" * 60)

    # Grid + layout
    if args.domain == 'square_hole':
        X, Y, layout, _, _ = create_square_with_hole(
            Nx=args.nx, Ny=args.ny,
            hole_center=(args.hole_center_x, args.hole_center_y),
            hole_radius=args.hole_radius)
        cx, cy, r_h = args.hole_center_x, args.hole_center_y, args.hole_radius
        def dirichlet_predicate(x, y):
            return (x - cx) ** 2 + (y - cy) ** 2 <= r_h ** 2
    else:
        X, Y, layout, _, _ = create_square(Nx=args.nx, Ny=args.ny)
        dirichlet_predicate = None

    # Source + Dirichlet BC callables for FEM
    if args.source == 'gaussian':
        f_grid = gaussian_source(X, Y, x_s=args.x_s, y_s=args.y_s,
                                 sigma=args.sigma, amplitude=args.amplitude)
        def f_callable(x, y):
            r2 = (x - args.x_s) ** 2 + (y - args.y_s) ** 2
            return args.amplitude * np.exp(-r2 / (2.0 * args.sigma ** 2))
        def g_callable(x, y):
            return np.zeros_like(x)
    else:
        _, f_grid = manufactured_solution(X, Y, args.k)
        def f_callable(x, y):
            return (args.k ** 2) * np.sin(args.k * x) * np.sin(args.k * y)
        def g_callable(x, y):
            return np.sin(args.k * x) * np.sin(args.k * y)

    # Solver
    solver = HelmholtzSolver(k=args.k, mesh_n=args.mesh_n,
                             dirichlet_predicate=dirichlet_predicate)

    # PINN
    meta_path = os.path.join(args.pinn_meta_dir, f'training_meta_{args.tag}.json')
    with open(meta_path) as f:
        pinn_meta = json.load(f)
    layers_cfg = pinn_meta.get('layers', [256, 256, 256, 256, 256])
    fourier_m = int(pinn_meta.get('fourier_m', 64))
    fourier_scale = float(pinn_meta.get('fourier_scale', args.k / (2 * np.pi)))
    pinn = build_pinn(num_inputs=2, layers=tuple(layers_cfg),
                      activation='tanh',
                      input_range=((0.0, 1.0), (0.0, 1.0)),
                      fourier_m=fourier_m, fourier_scale=fourier_scale)
    _ = pinn(tf.zeros((1, 2)))
    pinn.load_weights(os.path.join(
        args.pinn_dir, f'pinn_helmholtz_{args.tag}.weights.h5'))

    # PINN prediction on grid
    xy = np.stack([X.ravel(), Y.ravel()], axis=-1).astype(np.float32)
    pinn_u = pinn.predict(xy, batch_size=len(xy), verbose=0).reshape(X.shape)
    pinn_u = pinn_u * layout

    # PDE residual (precomputed for router input; fixed for all thresholds)
    residual = HelmholtzResidualComputer(pinn, args.k).compute_residual(
        X, Y, f_grid) * layout

    # Router
    router = RouterCNN(base_filters=args.base_filters)
    dummy_inputs = create_router_input(layout, f_grid, pinn_u, residual)
    _ = router(tf.constant(dummy_inputs, dtype=tf.float32))
    router.load_weights(os.path.join(
        args.router_dir, f'router_helmholtz_{args.tag}.weights.h5'))

    # Precompute router logits once (fixed for the config; threshold
    # sweeps reuse these via `reuse_logits`).
    logits = router(tf.constant(dummy_inputs, dtype=tf.float32),
                    training=False)[0, :, :, 0].numpy()

    # ---------- Warmup ----------
    print("\nWarmup...")
    for _ in range(N_WARMUP):
        solver.solve(f_callable, g_callable)
        solve_hybrid_schwarz(solver, pinn, router, f_callable, g_callable,
                             X, Y, layout, f_grid, pinn_u, residual,
                             threshold=0.0, reuse_logits=logits)

    # ---------- FEM baseline loop ----------
    # Time solve + interp_to_grid end-to-end so this matches the hybrid
    # wall clock (which also returns a grid solution).
    print("\nFEM baseline loop (solve + interp)...")
    fem_times = []
    u_fem_reference = None
    for i in range(N_RUNS):
        t0 = time.perf_counter()
        u_dof, _ = solver.solve(f_callable, g_callable)
        u_grid = solver.interp_to_grid(u_dof, X, Y)
        t_total = time.perf_counter() - t0
        fem_times.append(t_total)
        if u_fem_reference is None:
            u_fem_reference = u_grid
    fem_mean, fem_std = float(np.mean(fem_times)), float(np.std(fem_times))
    print(f"  mean={fem_mean:.4f}s  std={fem_std:.4f}s")

    # ---------- Hybrid loop at threshold 0 ----------
    print("\nHybrid loop (threshold=0)...")
    hyb_total = []
    hyb_pinn = []  # includes pinn_pin_time_s (vertex evaluation)
    hyb_router = []
    hyb_solve = []
    for i in range(N_RUNS):
        t0 = time.perf_counter()
        # Time the router call fresh here (don't reuse_logits)
        res = solve_hybrid_schwarz(solver, pinn, router, f_callable, g_callable,
                                   X, Y, layout, f_grid, pinn_u, residual,
                                   threshold=0.0)
        total = time.perf_counter() - t0
        hyb_total.append(total)
        hyb_router.append(res['router_time_s'])
        hyb_pinn.append(res['pinn_pin_time_s'])
        hyb_solve.append(res['solve_time_s'])
    hyb_mean = float(np.mean(hyb_total))
    hyb_std = float(np.std(hyb_total))
    print(f"  mean total={hyb_mean:.4f}s  std={hyb_std:.4f}s  "
          f"speedup={fem_mean / max(hyb_mean, 1e-12):.2f}x")
    print(f"  router={np.mean(hyb_router):.4f}s  "
          f"pin={np.mean(hyb_pinn):.4f}s  solve={np.mean(hyb_solve):.4f}s")

    # ---------- Coverage sweep ----------
    print("\nCoverage sweep (11 targets)...")
    coverage_targets = np.linspace(0.0, 1.0, 11)
    sweep = []
    # Also evaluate PINN-only rel L2 vs FEM reference for the plot.
    pinn_rel_l2 = rel_l2(pinn_u, u_fem_reference)
    for cov in coverage_targets:
        thr = threshold_for_coverage(logits, layout, float(cov))
        t0 = time.perf_counter()
        res = solve_hybrid_schwarz(solver, pinn, router, f_callable, g_callable,
                                   X, Y, layout, f_grid, pinn_u, residual,
                                   threshold=thr, reuse_logits=logits)
        wall = time.perf_counter() - t0
        errl2 = rel_l2(res['u_grid'], u_fem_reference)
        sweep.append({
            'target_coverage': float(cov),
            'threshold': float(thr),
            'actual_coverage_pct': float(res['coverage_pct']),
            'hybrid_total_s': float(wall),
            'hybrid_solve_only_s': float(res['solve_time_s']),
            'rel_l2_vs_fem': float(errl2),
            'n_accepted_dofs': int(res['n_accepted_dofs']),
        })
        print(f"  target={cov*100:5.1f}%  "
              f"actual={res['coverage_pct']:5.1f}%  "
              f"solve={res['solve_time_s']*1000:6.1f}ms  "
              f"rel_L2={errl2:.3e}")

    summary = {
        'k': args.k,
        'tag': args.tag,
        'domain': args.domain,
        'source': args.source,
        'x_s': args.x_s, 'y_s': args.y_s,
        'fem_mean_s': fem_mean, 'fem_std_s': fem_std,
        'hybrid_mean_s': hyb_mean, 'hybrid_std_s': hyb_std,
        'hybrid_router_mean_s': float(np.mean(hyb_router)),
        'hybrid_pin_mean_s': float(np.mean(hyb_pinn)),
        'hybrid_solve_mean_s': float(np.mean(hyb_solve)),
        'speedup': fem_mean / max(hyb_mean, 1e-12),
        'pinn_rel_l2_vs_fem': pinn_rel_l2,
        'coverage_sweep': sweep,
    }

    out_path = os.path.join(args.output_dir, f'timing_{args.tag}.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved timing summary: {out_path}")

    # Also persist the FEM reference grid for the plot script.
    ref_path = os.path.join(args.output_dir, f'reference_{args.tag}.npz')
    np.savez(ref_path, X=X, Y=Y, layout=layout,
             u_fem=u_fem_reference, logits=logits,
             pinn_u=pinn_u, residual=residual, f_grid=f_grid)
    print(f"Saved reference arrays: {ref_path}")


if __name__ == '__main__':
    main()
