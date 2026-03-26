#!/usr/bin/env python3
"""
Time FDM and hybrid solvers for 2D linear elasticity.

Hybrid timing includes the full "meta equation":
    hybrid_time = PINN_inference + router_inference + FDM_solve

Runs multiple trials at the optimal threshold.
"""

import argparse
import contextlib
import io
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'no-latex'])

from lib.network import Network
from lib.domains import create_plate_with_hole, create_l_bracket
from lib.solver import ElasticitySolver
from lib.router import (
    RouterCNN,
    PINNResidualComputer,
    create_router_input,
    compute_bc_error_field,
    solve_error_transport,
)


def pinn_predict(pinn_model, X, Y, layout, E=1.0, nu=0.3):
    """Run PINN inference and return ux, uy, von_mises fields."""
    xy_flat = np.stack([X.flatten(), Y.flatten()], axis=-1).astype(np.float32)
    pinn_out = pinn_model.predict(xy_flat, batch_size=len(xy_flat), verbose=0)
    pinn_ux = pinn_out[:, 0].reshape(X.shape).astype(np.float32) * layout
    pinn_uy = pinn_out[:, 1].reshape(X.shape).astype(np.float32) * layout

    dx = (X[0, -1] - X[0, 0]) / (X.shape[1] - 1)
    dy = (Y[-1, 0] - Y[0, 0]) / (Y.shape[0] - 1)
    C11 = E / (1.0 - nu ** 2)
    C12 = nu * E / (1.0 - nu ** 2)
    C66 = E / (2.0 * (1.0 + nu))

    exx = np.zeros_like(pinn_ux)
    eyy = np.zeros_like(pinn_uy)
    exy = np.zeros_like(pinn_ux)
    exx[1:-1, 1:-1] = (pinn_ux[1:-1, 2:] - pinn_ux[1:-1, :-2]) / (2 * dx)
    eyy[1:-1, 1:-1] = (pinn_uy[2:, 1:-1] - pinn_uy[:-2, 1:-1]) / (2 * dy)
    exy[1:-1, 1:-1] = 0.5 * (
        (pinn_ux[2:, 1:-1] - pinn_ux[:-2, 1:-1]) / (2 * dy)
        + (pinn_uy[1:-1, 2:] - pinn_uy[1:-1, :-2]) / (2 * dx)
    )
    sxx = C11 * exx + C12 * eyy
    syy = C12 * exx + C11 * eyy
    sxy = 2 * C66 * exy
    pinn_vm = np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3 * sxy ** 2).astype(np.float32) * layout

    return pinn_ux, pinn_uy, pinn_vm


def router_predict(router, pinn_ux, pinn_uy, pinn_vm, layout,
                   disp_bc_mask, bc_ux, bc_uy, trac_bc_mask, threshold=0.0,
                   E=1.0, nu=0.3, x_domain=(0.0, 2.0), y_domain=(0.0, 2.0)):
    """Run router inference and return output + mask."""
    bc_error = compute_bc_error_field(disp_bc_mask, bc_ux, bc_uy, pinn_ux, pinn_uy, layout)
    error_transport = solve_error_transport(
        bc_error, layout, E=E, nu=nu,
        x_domain=x_domain, y_domain=y_domain,
    )
    inputs = create_router_input(
        layout, disp_bc_mask, bc_ux, bc_uy, trac_bc_mask,
        pinn_ux, pinn_uy, pinn_vm, error_transport,
    )
    router_output = router(tf.constant(inputs, dtype=tf.float32),
                           training=False).numpy().squeeze()
    mask = (router_output > threshold).astype(np.int32) * layout.astype(np.int32)
    return router_output, mask


def find_optimal_threshold(residual_field, router_output, layout, beta, n_points=200):
    """Find optimal threshold by sweeping thresholds on router logits."""
    fluid_mask = layout > 0
    residuals = residual_field[fluid_mask]
    logits = router_output[fluid_mask]
    if logits.size == 0:
        return 0.0, 0.0

    lo, hi = float(logits.min()), float(logits.max())
    margin = max(0.1, (hi - lo) * 0.05)
    thresholds = np.linspace(lo - margin, hi + margin, max(n_points, 500))

    best_loss = float('inf')
    best_cov = 0.0
    best_t = 0.0

    for t in thresholds:
        m = logits > t
        cov = np.mean(m)
        pinn_res = np.mean(residuals[~m]) if (~m).sum() > 0 else 0.0
        loss = beta * cov + (1 - cov) * pinn_res
        if loss < best_loss:
            best_loss = loss
            best_cov = cov
            best_t = float(t)

    return best_t, best_cov


def compute_residual_field(pinn_model, X, Y, layout, E=1.0, nu=0.3, bc_error=None):
    """Compute physics residual field with median normalization."""
    residual_computer = PINNResidualComputer(pinn_model, E, nu)
    X_tf = tf.constant(X, dtype=tf.float32)
    Y_tf = tf.constant(Y, dtype=tf.float32)

    eq_x, eq_y = residual_computer.compute_residuals(X_tf, Y_tf)
    residual_field = (np.array(eq_x) + np.array(eq_y)) * layout

    if bc_error is not None:
        residual_field = residual_field + bc_error

    fluid_vals = residual_field[layout > 0]
    median = np.median(fluid_vals)
    residual_field = residual_field / (median + 1e-10) * layout
    return residual_field


def main():
    parser = argparse.ArgumentParser(
        description='Time FDM and hybrid solvers (elasticity)'
    )
    parser.add_argument('--problem', type=str, default='plate_with_hole',
                        choices=['plate_with_hole', 'l_bracket'])
    parser.add_argument('--pinn-path', type=str,
                        default='./models/pinn_plate_with_hole.weights.h5')
    parser.add_argument('--router-weights', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./timing_output')

    parser.add_argument('--nx', type=int, default=200)
    parser.add_argument('--ny', type=int, default=200)
    parser.add_argument('--max-iter', type=int, default=200000)
    parser.add_argument('--tol', type=float, default=1e-8)

    parser.add_argument('--x-min', type=float, default=-2.0)
    parser.add_argument('--x-max', type=float, default=2.0)
    parser.add_argument('--y-min', type=float, default=-2.0)
    parser.add_argument('--y-max', type=float, default=2.0)

    # Plate with hole
    parser.add_argument('--hole-x', type=float, default=0.0)
    parser.add_argument('--hole-y', type=float, default=0.0)
    parser.add_argument('--hole-radius', type=float, default=0.5)
    parser.add_argument('--applied-stress', type=float, default=10.0)

    # L-bracket
    parser.add_argument('--corner-x', type=float, default=1.0)
    parser.add_argument('--corner-y', type=float, default=1.0)

    # Material
    parser.add_argument('--E', type=float, default=1.0)
    parser.add_argument('--nu', type=float, default=0.3)

    # Router
    parser.add_argument('--layers', type=int, nargs='+', default=[128, 128, 128, 128])
    parser.add_argument('--base-filters', type=int, default=32)
    parser.add_argument('--beta', type=float, default=1.1)

    # Timing
    parser.add_argument('--n-runs', type=int, default=3)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    N_RUNS = args.n_runs

    if args.problem == 'l_bracket' and args.x_min == -2.0:
        args.x_min = 0.0
        args.y_min = 0.0

    # ================================================================
    # SETUP (not timed)
    # ================================================================
    print("=" * 60)
    print(f"SOLVER TIMING — ELASTICITY ({args.problem})")
    print("=" * 60)
    print(f"  PINN:   {args.pinn_path}")
    print(f"  Router: {args.router_weights}")
    print(f"  Grid:   {args.nx} x {args.ny}")
    print(f"  E={args.E}, nu={args.nu}")
    print(f"  Runs:   {N_RUNS}")
    print()

    # Domain
    if args.problem == 'plate_with_hole':
        X, Y, layout, disp_bc_mask, trac_bc_mask, bc_ux, bc_uy, bc_tx, bc_ty = \
            create_plate_with_hole(
                Nx=args.nx, Ny=args.ny,
                x_domain=(args.x_min, args.x_max),
                y_domain=(args.y_min, args.y_max),
                hole_center=(args.hole_x, args.hole_y),
                hole_radius=args.hole_radius,
                applied_stress=args.applied_stress,
            )
    else:
        X, Y, layout, disp_bc_mask, trac_bc_mask, bc_ux, bc_uy, bc_tx, bc_ty = \
            create_l_bracket(
                Nx=args.nx, Ny=args.ny,
                x_domain=(args.x_min, args.x_max),
                y_domain=(args.y_min, args.y_max),
                corner_x=args.corner_x,
                corner_y=args.corner_y,
                applied_stress=args.applied_stress,
            )

    # PINN
    network = Network()
    input_range = [(args.x_min, args.x_max), (args.y_min, args.y_max)]
    pinn_model = network.build(num_inputs=2, layers=args.layers,
                               activation='tanh', num_outputs=2,
                               input_range=input_range)
    pinn_model.load_weights(args.pinn_path)

    # Initial PINN predictions (for router setup)
    pinn_ux, pinn_uy, pinn_vm = pinn_predict(pinn_model, X, Y, layout,
                                              E=args.E, nu=args.nu)

    # Router
    bc_error = compute_bc_error_field(disp_bc_mask, bc_ux, bc_uy,
                                      pinn_ux, pinn_uy, layout)
    error_transport = solve_error_transport(
        bc_error, layout, E=args.E, nu=args.nu,
        x_domain=(args.x_min, args.x_max),
        y_domain=(args.y_min, args.y_max),
    )
    inputs = create_router_input(
        layout, disp_bc_mask, bc_ux, bc_uy, trac_bc_mask,
        pinn_ux, pinn_uy, pinn_vm, error_transport,
    )
    router = RouterCNN(base_filters=args.base_filters)
    _ = router(tf.constant(inputs, dtype=tf.float32))
    router.load_weights(args.router_weights)
    router_output = router(tf.constant(inputs, dtype=tf.float32),
                           training=False).numpy().squeeze()

    # Residual field for optimal threshold
    print("Computing residual field for optimal threshold...")
    residual_field = compute_residual_field(
        pinn_model, X, Y, layout, E=args.E, nu=args.nu, bc_error=bc_error
    )

    optimal_threshold, optimal_coverage = find_optimal_threshold(
        residual_field, router_output, layout, args.beta
    )
    fdm_mask_opt = (router_output > optimal_threshold).astype(np.int32) \
        * layout.astype(np.int32)
    actual_coverage = np.sum(fdm_mask_opt) / np.sum(layout)

    print(f"  Optimal threshold: {optimal_threshold:.6f}")
    print(f"  Optimal coverage:  {optimal_coverage * 100:.2f}%")
    print(f"  FDM coverage:      {actual_coverage * 100:.2f}%")
    print()

    # ================================================================
    # WARMUP (JIT compilation, not counted)
    # ================================================================
    print("--- Warmup: FDM solve (short) ---")
    with contextlib.redirect_stdout(io.StringIO()):
        solver = ElasticitySolver(
            E=args.E, nu=args.nu,
            x_domain=(args.x_min, args.x_max),
            y_domain=(args.y_min, args.y_max),
            Nx=args.nx, Ny=args.ny,
            max_iter=100, tol=args.tol,
        )
        solver.solve(layout, disp_bc_mask, bc_ux, bc_uy,
                     trac_bc_mask, bc_tx, bc_ty)
    print("  done")

    print("--- Warmup: Hybrid solve (short) ---")
    with contextlib.redirect_stdout(io.StringIO()):
        solver = ElasticitySolver(
            E=args.E, nu=args.nu,
            x_domain=(args.x_min, args.x_max),
            y_domain=(args.y_min, args.y_max),
            Nx=args.nx, Ny=args.ny,
            max_iter=100, tol=args.tol,
        )
        solver.solve(layout, disp_bc_mask, bc_ux, bc_uy,
                     trac_bc_mask, bc_tx, bc_ty,
                     fdm_mask=fdm_mask_opt,
                     initial_ux=pinn_ux, initial_uy=pinn_uy)
    print("  done")

    # TF warmup
    xy_flat = np.stack([X.flatten(), Y.flatten()], axis=-1).astype(np.float32)
    _ = pinn_model.predict(xy_flat[:1], verbose=0)
    _ = router(tf.constant(inputs, dtype=tf.float32), training=False)
    print("  TF warmup done\n")

    # ================================================================
    # TIME FDM (ground truth)
    # ================================================================
    print(f"--- FDM timing ({N_RUNS} runs) ---")
    fdm_times = []
    for i in range(N_RUNS):
        with contextlib.redirect_stdout(io.StringIO()):
            solver = ElasticitySolver(
                E=args.E, nu=args.nu,
                x_domain=(args.x_min, args.x_max),
                y_domain=(args.y_min, args.y_max),
                Nx=args.nx, Ny=args.ny,
                max_iter=args.max_iter, tol=args.tol,
            )
            t0 = time.perf_counter()
            ux_fdm, uy_fdm, sxx, syy, sxy = solver.solve(
                layout, disp_bc_mask, bc_ux, bc_uy,
                trac_bc_mask, bc_tx, bc_ty,
            )
            t1 = time.perf_counter()
        fdm_times.append(t1 - t0)
        print(f"  FDM run {i + 1}/{N_RUNS}: {fdm_times[-1]:.4f} s")

    # Keep last FDM as ground truth
    disp_fdm = np.sqrt(ux_fdm ** 2 + uy_fdm ** 2)

    # ================================================================
    # TIME HYBRID (full pipeline: PINN + router + hybrid FDM)
    # ================================================================
    print(f"\n--- Hybrid timing ({N_RUNS} runs, full pipeline) ---")
    hybrid_times = []
    pinn_times = []
    router_times = []
    solve_times = []

    for i in range(N_RUNS):
        t_total_start = time.perf_counter()

        # 1. PINN inference
        t_pinn_start = time.perf_counter()
        h_pinn_ux, h_pinn_uy, h_pinn_vm = pinn_predict(
            pinn_model, X, Y, layout, E=args.E, nu=args.nu
        )
        t_pinn_end = time.perf_counter()

        # 2. Router inference
        t_router_start = time.perf_counter()
        h_router_output, h_mask = router_predict(
            router, h_pinn_ux, h_pinn_uy, h_pinn_vm, layout,
            disp_bc_mask, bc_ux, bc_uy, trac_bc_mask,
            threshold=optimal_threshold,
            E=args.E, nu=args.nu,
            x_domain=(args.x_min, args.x_max),
            y_domain=(args.y_min, args.y_max),
        )
        t_router_end = time.perf_counter()

        # 3. Hybrid FDM solve
        with contextlib.redirect_stdout(io.StringIO()):
            solver = ElasticitySolver(
                E=args.E, nu=args.nu,
                x_domain=(args.x_min, args.x_max),
                y_domain=(args.y_min, args.y_max),
                Nx=args.nx, Ny=args.ny,
                max_iter=args.max_iter, tol=args.tol,
            )
            t_solve_start = time.perf_counter()
            ux_hyb, uy_hyb, _, _, _ = solver.solve(
                layout, disp_bc_mask, bc_ux, bc_uy,
                trac_bc_mask, bc_tx, bc_ty,
                fdm_mask=h_mask,
                initial_ux=h_pinn_ux, initial_uy=h_pinn_uy,
            )
            t_solve_end = time.perf_counter()

        t_total_end = time.perf_counter()

        pinn_times.append(t_pinn_end - t_pinn_start)
        router_times.append(t_router_end - t_router_start)
        solve_times.append(t_solve_end - t_solve_start)
        hybrid_times.append(t_total_end - t_total_start)

        print(f"  Hybrid run {i + 1}/{N_RUNS}: {hybrid_times[-1]:.4f}s "
              f"(PINN: {pinn_times[-1]:.4f}s, Router: {router_times[-1]:.4f}s, "
              f"Solve: {solve_times[-1]:.4f}s)")

    # ================================================================
    # RESULTS
    # ================================================================
    fdm_mean = np.mean(fdm_times)
    fdm_std = np.std(fdm_times)
    hyb_mean = np.mean(hybrid_times)
    hyb_std = np.std(hybrid_times)
    speedup = fdm_mean / hyb_mean if hyb_mean > 0 else float('inf')

    print()
    print("=" * 60)
    print("TIMING RESULTS")
    print("=" * 60)
    print(f"  FDM      : {fdm_mean:.4f} +/- {fdm_std:.4f} s")
    print(f"  Hybrid   : {hyb_mean:.4f} +/- {hyb_std:.4f} s (full pipeline)")
    print(f"    PINN   : {np.mean(pinn_times):.4f} +/- {np.std(pinn_times):.4f} s")
    print(f"    Router : {np.mean(router_times):.4f} +/- {np.std(router_times):.4f} s")
    print(f"    Solve  : {np.mean(solve_times):.4f} +/- {np.std(solve_times):.4f} s")
    print(f"  Speedup  : {speedup:.2f}x")
    print(f"  Coverage : {actual_coverage * 100:.2f}%")
    print(f"  Threshold: {optimal_threshold:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
