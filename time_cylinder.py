#!/usr/bin/env python3
"""
Time CFD and hybrid solvers for cylinder flow scenarios.

Only times sim.solve() calls. Router/PINN loading is excluded.
Runs multiple trials for the optimal threshold, plus a single-run
coverage sweep to produce a coverage-vs-RMSE-and-time plot.
"""

import argparse
import contextlib
import io
import os
import time

import numpy as np
import tensorflow as tf

# Configure TensorFlow GPU memory growth
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

from lib.router import (
    RouterCNN,
    PINNResidualComputer,
    create_router_input,
    compute_bc_error_field,
    solve_error_transport,
    create_cylinder_setup,
)
from lib.cylinder_flow import CylinderFlowSimulation, CylinderFlowHybridSimulation
from cylinder_network import Network as CylinderNetwork


def compute_uv_direct(network, xy):
    uvp = network.predict(xy, batch_size=len(xy), verbose=0)
    return uvp[..., 0], uvp[..., 1]


def find_optimal_threshold(residual_field, router_output, layout, beta, n_points=200):
    """Find optimal threshold from target loss curve (no TV)."""
    fluid_mask = layout > 0
    residuals = residual_field[fluid_mask]
    logits = router_output[fluid_mask]
    n_fluid = len(residuals)

    sorted_idx = np.argsort(logits)[::-1]
    sorted_residuals = residuals[sorted_idx]
    sorted_logits = logits[sorted_idx]

    coverage = np.linspace(0, 1, n_points)
    loss = np.zeros(n_points)

    for i, cov in enumerate(coverage):
        n_cfd = int(cov * n_fluid)
        cfd_cost = beta * cov
        if n_fluid - n_cfd > 0:
            residual_loss = (1 - cov) * np.mean(sorted_residuals[n_cfd:])
        else:
            residual_loss = 0.0
        loss[i] = cfd_cost + residual_loss

    optimal_idx = np.argmin(loss)
    optimal_coverage = coverage[optimal_idx]
    n_cfd_optimal = int(optimal_coverage * n_fluid)

    if 0 < n_cfd_optimal < n_fluid:
        optimal_threshold = sorted_logits[n_cfd_optimal - 1]
    elif n_cfd_optimal == 0:
        optimal_threshold = sorted_logits[0] + 0.001 if n_fluid > 0 else 1.0
    else:
        optimal_threshold = sorted_logits[-1] - 0.001 if n_fluid > 0 else 0.0

    return optimal_threshold, optimal_coverage


def threshold_for_coverage(router_output, layout, target_cov):
    """Return the threshold that achieves approximately target_cov CFD coverage."""
    fluid_logits = router_output[layout > 0]
    n_fluid = len(fluid_logits)
    sorted_logits = np.sort(fluid_logits)[::-1]
    n_cfd = int(target_cov * n_fluid)
    n_cfd = max(0, min(n_cfd, n_fluid - 1))
    if n_cfd == 0:
        return sorted_logits[0] + 1.0
    return sorted_logits[n_cfd - 1]


def make_cfd_sim(args):
    return CylinderFlowSimulation(
        Re=args.Re, N=args.ny, max_iter=args.max_iter, tol=args.tol,
        x_domain=(args.x_min, args.x_max),
        y_domain=(args.y_min, args.y_max),
        cylinder_center=(args.cylinder_x, args.cylinder_y),
        cylinder_radius=args.cylinder_radius,
        inlet_velocity=args.inlet_velocity,
    )


def make_hybrid_sim(args, pinn_model, cfd_mask):
    return CylinderFlowHybridSimulation(
        network=pinn_model, uv_func=compute_uv_direct, mask=cfd_mask,
        Re=args.Re, N=args.ny, max_iter=args.max_iter, tol=args.tol,
        x_domain=(args.x_min, args.x_max),
        y_domain=(args.y_min, args.y_max),
        cylinder_center=(args.cylinder_x, args.cylinder_y),
        cylinder_radius=args.cylinder_radius,
        inlet_velocity=args.inlet_velocity,
    )


def main():
    parser = argparse.ArgumentParser(description='Time CFD and hybrid solvers (cylinder)')
    parser.add_argument('--pinn-path', required=True)
    parser.add_argument('--router-weights', required=True)
    parser.add_argument('--output-dir', type=str, default='./timing_output')
    parser.add_argument('--nx', type=int, default=200)
    parser.add_argument('--ny', type=int, default=100)
    parser.add_argument('--x-min', type=float, default=0.0)
    parser.add_argument('--x-max', type=float, default=2.0)
    parser.add_argument('--y-min', type=float, default=0.0)
    parser.add_argument('--y-max', type=float, default=1.0)
    parser.add_argument('--cylinder-x', type=float, default=0.5)
    parser.add_argument('--cylinder-y', type=float, default=0.5)
    parser.add_argument('--cylinder-radius', type=float, default=0.1)
    parser.add_argument('--inlet-velocity', type=float, default=1.0)
    parser.add_argument('--Re', type=float, default=100)
    parser.add_argument('--max-iter', type=int, default=200000)
    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--base-filters', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--lambda-tv', type=float, default=0.01)
    parser.add_argument('--n-runs', type=int, default=3)
    parser.add_argument('--n-coverages', type=int, default=10,
                        help='Number of coverage levels for the sweep (each run once)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    N_RUNS = args.n_runs
    nu = 1.0 / args.Re

    # ================================================================
    # SETUP (not timed)
    # ================================================================
    print("=" * 60)
    print("SOLVER TIMING — CYLINDER FLOW")
    print("=" * 60)
    print(f"  PINN:     {args.pinn_path}")
    print(f"  Router:   {args.router_weights}")
    print(f"  Grid:     {args.nx} x {args.ny}")
    print(f"  Cylinder: ({args.cylinder_x}, {args.cylinder_y}), r={args.cylinder_radius}")
    print(f"  Re={args.Re}, inlet_vel={args.inlet_velocity}")
    print(f"  Runs:     {N_RUNS}")
    print()

    # Domain
    X, Y, layout, bc_mask, bc_u, bc_v, bc_p = create_cylinder_setup(
        Nx=args.nx, Ny=args.ny,
        x_domain=(args.x_min, args.x_max),
        y_domain=(args.y_min, args.y_max),
        cylinder_center=(args.cylinder_x, args.cylinder_y),
        cylinder_radius=args.cylinder_radius,
        inlet_velocity=args.inlet_velocity,
    )

    # PINN
    network = CylinderNetwork()
    pinn_model = network.build(
        num_inputs=2, layers=[48, 48, 48, 48],
        activation='tanh', num_outputs=3,
    )
    pinn_model.load_weights(args.pinn_path)

    # PINN predictions
    xy_flat = np.stack([X.flatten(), Y.flatten()], axis=-1).astype(np.float32)
    pinn_uvp = pinn_model.predict(xy_flat, batch_size=len(xy_flat), verbose=0)
    pinn_u = pinn_uvp[:, 0].reshape(X.shape).astype(np.float32) * layout
    pinn_v = pinn_uvp[:, 1].reshape(X.shape).astype(np.float32) * layout
    pinn_p = pinn_uvp[:, 2].reshape(X.shape).astype(np.float32) * layout

    # Router input + inference
    bc_error_local = compute_bc_error_field(
        bc_mask, bc_u, bc_v, pinn_u, pinn_v, layout
    )
    error_transport = solve_error_transport(
        pinn_u, pinn_v, bc_error_local, layout, nu=nu,
        x_domain=(args.x_min, args.x_max),
        y_domain=(args.y_min, args.y_max),
    )
    inputs = create_router_input(layout, bc_mask, bc_u, bc_v, bc_p,
                                 pinn_u, pinn_v, pinn_p, error_transport)
    router = RouterCNN(base_filters=args.base_filters, temperature=args.temperature)
    _ = router(inputs)
    router.load_weights(args.router_weights)
    router_output = router(inputs, training=False)[0, :, :, 0].numpy()

    # Compute residual field for optimal threshold
    print("Computing residual field for optimal threshold...")
    residual_computer = PINNResidualComputer(pinn_model, nu=nu, rho=1.0)
    X_tf = tf.constant(X, dtype=tf.float32)
    Y_tf = tf.constant(Y, dtype=tf.float32)
    bc_mask_tf = tf.constant(bc_mask, dtype=tf.float32)
    bc_u_tf = tf.constant(bc_u, dtype=tf.float32)
    bc_v_tf = tf.constant(bc_v, dtype=tf.float32)
    pde_residual = residual_computer.compute_total_residual_with_bc(
        X_tf, Y_tf, bc_mask_tf, bc_u_tf, bc_v_tf,
        {'continuity': 1.0, 'momentum': 1.0}
    ).numpy() * layout
    residual_field = pde_residual + error_transport
    fluid_vals = residual_field[layout > 0]
    median_r = np.median(fluid_vals)
    if median_r > 1e-10:
        residual_field = residual_field / median_r

    optimal_threshold, optimal_coverage = find_optimal_threshold(
        residual_field, router_output, layout, args.beta
    )
    cfd_mask_opt = (router_output >= optimal_threshold).astype(np.int32) * layout.astype(np.int32)
    actual_coverage = np.sum(cfd_mask_opt) / np.sum(layout)

    print(f"  Optimal threshold: {optimal_threshold:.6f}")
    print(f"  CFD coverage:      {actual_coverage * 100:.2f}%")
    print()

    # ================================================================
    # WARMUP (JIT compilation, not counted)
    # ================================================================
    print("--- Warmup: CFD solve ---")
    with contextlib.redirect_stdout(io.StringIO()):
        make_cfd_sim(args).solve()
    print("  done")

    print("--- Warmup: Hybrid solve ---")
    with contextlib.redirect_stdout(io.StringIO()):
        make_hybrid_sim(args, pinn_model, cfd_mask_opt).solve()
    print("  done\n")

    # ================================================================
    # TIME CFD (ground truth)
    # ================================================================
    cfd_times = []
    for i in range(N_RUNS):
        sim = make_cfd_sim(args)
        with contextlib.redirect_stdout(io.StringIO()):
            t0 = time.perf_counter()
            u_cfd, v_cfd, p_cfd = sim.solve()
            t1 = time.perf_counter()
        cfd_times.append(t1 - t0)
        print(f"  CFD  run {i + 1}/{N_RUNS}: {cfd_times[-1]:.4f} s")

    # Keep last CFD solution as ground truth for RMSE
    u_cfd = np.array(u_cfd)
    v_cfd = np.array(v_cfd)
    p_cfd = np.array(p_cfd)
    cfd_vel_mag = np.sqrt(u_cfd**2 + v_cfd**2)
    fluid_mask = layout > 0

    # ================================================================
    # TIME HYBRID (optimal threshold, repeated)
    # ================================================================
    hybrid_times = []
    for i in range(N_RUNS):
        sim = make_hybrid_sim(args, pinn_model, cfd_mask_opt)
        with contextlib.redirect_stdout(io.StringIO()):
            t0 = time.perf_counter()
            sim.solve()
            t1 = time.perf_counter()
        hybrid_times.append(t1 - t0)
        print(f"  Hybrid run {i + 1}/{N_RUNS}: {hybrid_times[-1]:.4f} s")

    # ================================================================
    # COVERAGE SWEEP (single run each)
    # ================================================================
    print(f"\n--- Coverage sweep ({args.n_coverages} levels, 1 run each) ---")
    target_coverages = np.linspace(0, 1, args.n_coverages + 2)[1:-1]  # exclude 0% and 100%

    sweep_cov = [0.0]  # start with PINN-only
    sweep_time = [0.0]  # PINN inference is ~instant relative to CFD
    pinn_vel_mag = np.sqrt(pinn_u**2 + pinn_v**2)
    rmse_pinn = np.sqrt(np.mean((pinn_vel_mag[fluid_mask] - cfd_vel_mag[fluid_mask])**2))
    sweep_rmse = [rmse_pinn]

    for target_cov in target_coverages:
        thresh = threshold_for_coverage(router_output, layout, target_cov)
        mask = (router_output >= thresh).astype(np.int32) * layout.astype(np.int32)
        cov = np.sum(mask) / np.sum(layout)

        sim = make_hybrid_sim(args, pinn_model, mask)
        with contextlib.redirect_stdout(io.StringIO()):
            t0 = time.perf_counter()
            uh, vh, ph = sim.solve()
            t1 = time.perf_counter()
        elapsed = t1 - t0

        uh, vh = np.array(uh), np.array(vh)
        hyb_vel = np.sqrt(uh**2 + vh**2)
        rmse = np.sqrt(np.mean((hyb_vel[fluid_mask] - cfd_vel_mag[fluid_mask])**2))

        sweep_cov.append(cov)
        sweep_time.append(elapsed)
        sweep_rmse.append(rmse)
        print(f"  cov={cov*100:5.1f}%  time={elapsed:.4f}s  RMSE={rmse:.6f}")

    # Add CFD-only (100% coverage)
    sweep_cov.append(1.0)
    sweep_time.append(np.mean(cfd_times))
    sweep_rmse.append(0.0)

    sweep_cov = np.array(sweep_cov)
    sweep_time = np.array(sweep_time)
    sweep_rmse = np.array(sweep_rmse)

    # ================================================================
    # PLOT: Coverage vs RMSE & Time
    # ================================================================
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_rmse = 'tab:blue'
    color_time = 'tab:red'

    ax1.set_xlabel('Coverage (% CFD)')
    ax1.set_ylabel('RMSE (vs CFD)', color=color_rmse)
    ax1.plot(sweep_cov * 100, sweep_rmse, 'o-', color=color_rmse, linewidth=2, markersize=6,
             label='RMSE')
    ax1.tick_params(axis='y', labelcolor=color_rmse)
    ax1.set_xlim(-5, 105)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Solve Time (s)', color=color_time)
    ax2.plot(sweep_cov * 100, sweep_time, 's-', color=color_time, linewidth=2, markersize=6,
             label='Time')
    ax2.tick_params(axis='y', labelcolor=color_time)

    # Mark optimal point
    ax1.axvline(x=actual_coverage * 100, color='green', linestyle='--', linewidth=1.5,
                alpha=0.7, label=f'Optimal ({actual_coverage*100:.0f}%)')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=9)

    ax1.set_title(f'Coverage vs RMSE & Solve Time — Cylinder (Re={args.Re})')
    fig.tight_layout()

    plot_path = os.path.join(args.output_dir, 'coverage_time_rmse.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved plot to {plot_path}")

    # Save sweep data
    np.savez(os.path.join(args.output_dir, 'timing_sweep.npz'),
             coverage=sweep_cov, time=sweep_time, rmse=sweep_rmse)

    # ================================================================
    # RESULTS
    # ================================================================
    cfd_mean = np.mean(cfd_times)
    cfd_std = np.std(cfd_times)
    hyb_mean = np.mean(hybrid_times)
    hyb_std = np.std(hybrid_times)
    speedup = cfd_mean / hyb_mean if hyb_mean > 0 else float('inf')

    print()
    print("=" * 60)
    print("TIMING RESULTS")
    print("=" * 60)
    print(f"  CFD     : {cfd_mean:.4f} +/- {cfd_std:.4f} s  {cfd_times}")
    print(f"  Hybrid  : {hyb_mean:.4f} +/- {hyb_std:.4f} s  {hybrid_times}")
    print(f"  Speedup : {speedup:.2f}x")
    print(f"  Coverage: {actual_coverage * 100:.2f}%")
    print(f"  Threshold: {optimal_threshold:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
