#!/usr/bin/env python3
"""
Time CFD and hybrid solvers for cylinder flow scenarios.

Only times sim.solve() calls. Router/PINN loading is excluded.
Runs multiple trials for the optimal threshold.
"""

import argparse
import contextlib
import io
import os
import time

import numpy as np
import cv2
import tensorflow as tf
from scipy.ndimage import binary_erosion, binary_dilation


def _gauge_free_rmse(u_p, v_p, p_p, u_r, v_r, p_r,
                     valid_mask, dx, dy, gradp_scale):
    """Velocity + gauge-invariant pressure-gradient RMSE on valid cells.

    Replaces (p - p_ref) with (∇p - ∇p_ref) so a constant pressure offset
    (PINN vs CFD gauge) does not contribute. The gradient term is
    normalized by the CFD pressure-gradient scale so it is comparable to
    the velocity terms.
    """
    px, py = np.gradient(p_p, dy, dx)
    pxr, pyr = np.gradient(p_r, dy, dx)
    err = ((u_p - u_r)**2 + (v_p - v_r)**2
           + ((px - pxr)**2 + (py - pyr)**2) / gradp_scale)
    return float(np.sqrt(np.mean(err[valid_mask])))


def _interface_ring(mask, iterations=1):
    """Ring of `iterations` cells on both sides of a binary mask boundary."""
    m = mask.astype(bool)
    return binary_dilation(m, iterations=iterations) & \
           ~binary_erosion(m, iterations=iterations)

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
    compute_pinn_residual_field,
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


def apply_morph_open(mask, layout, kernel_size):
    """Apply morphological opening to smooth the binary mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return opened.astype(np.int32) * layout.astype(np.int32)


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


def compute_router_output_and_ete(router, pinn_model, X, Y,
                                  layout, bc_mask, bc_u, bc_v, bc_p,
                                  pinn_u, pinn_v, pinn_p,
                                  x_min, x_max, y_min, y_max, nu):
    """Compute router output together with channel-8 error transport field."""
    bc_error_local = compute_bc_error_field(
        bc_mask, bc_u, bc_v, pinn_u, pinn_v, layout
    )
    error_transport = solve_error_transport(
        pinn_u, pinn_v, bc_error_local, layout, nu=nu,
        x_domain=(x_min, x_max),
        y_domain=(y_min, y_max),
    )
    pde_residual_np = compute_pinn_residual_field(
        pinn_model, X, Y, layout, bc_mask, bc_u, bc_v, nu=nu
    )
    inputs = create_router_input(layout, bc_mask, bc_u, bc_v, bc_p,
                                 pinn_u, pinn_v, pinn_p, error_transport,
                                 pde_residual_np)
    router_output = router(inputs, training=False)[0, :, :, 0].numpy()
    return router_output, error_transport


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
    parser.add_argument('--morph-kernel', type=int, default=5,
                        help='Kernel size for morphological opening of mask')
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
    pde_residual_np = compute_pinn_residual_field(
        pinn_model, X, Y, layout, bc_mask, bc_u, bc_v, nu=nu
    )
    inputs = create_router_input(layout, bc_mask, bc_u, bc_v, bc_p,
                                 pinn_u, pinn_v, pinn_p, error_transport,
                                 pde_residual_np)
    router = RouterCNN(base_filters=args.base_filters, temperature=args.temperature)
    _ = router(inputs)
    router.load_weights(args.router_weights)
    router_output, error_transport = compute_router_output_and_ete(
        router, pinn_model, X, Y,
        layout, bc_mask, bc_u, bc_v, bc_p,
        pinn_u, pinn_v, pinn_p,
        args.x_min, args.x_max, args.y_min, args.y_max, nu,
    )

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
    cfd_mask_opt = apply_morph_open(cfd_mask_opt, layout, args.morph_kernel)
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

    # Keep last CFD solution as ground truth
    u_cfd = np.array(u_cfd)
    v_cfd = np.array(v_cfd)
    p_cfd = np.array(p_cfd)

    # ================================================================
    # TIME HYBRID (optimal threshold, repeated)
    # ================================================================
    hybrid_times = []
    for i in range(N_RUNS):
        with contextlib.redirect_stdout(io.StringIO()):
            t0 = time.perf_counter()
            router_output_timed, _ = compute_router_output_and_ete(
                router, pinn_model, X, Y,
                layout, bc_mask, bc_u, bc_v, bc_p,
                pinn_u, pinn_v, pinn_p,
                args.x_min, args.x_max, args.y_min, args.y_max, nu,
            )
            cfd_mask_timed = (router_output_timed >= optimal_threshold).astype(np.int32) * layout.astype(np.int32)
            cfd_mask_timed = apply_morph_open(cfd_mask_timed, layout, args.morph_kernel)
            sim = make_hybrid_sim(args, pinn_model, cfd_mask_timed)
            uh_opt, vh_opt, ph_opt = sim.solve()
            t1 = time.perf_counter()
        hybrid_times.append(t1 - t0)
        print(f"  Hybrid run {i + 1}/{N_RUNS}: {hybrid_times[-1]:.4f} s")

    # Keep last CFD solution as ground truth for RMSE
    fluid_mask = layout > 0

    # Gauge-invariant error: compare ∇p instead of p. Exclude a 1-cell ring
    # around the fluid boundary (gradient artifacts) and, for hybrid fields,
    # a 1-cell ring around the PINN/CFD interface.
    dx = float(X[0, 1] - X[0, 0])
    dy = float(Y[1, 0] - Y[0, 0])
    pxc, pyc = np.gradient(p_cfd, dy, dx)
    gradp_scale = float(np.max(pxc[fluid_mask]**2 + pyc[fluid_mask]**2)) + 1e-10
    interior = binary_erosion(fluid_mask, iterations=1)

    def _hybrid_valid(cfd_mask):
        ring = _interface_ring(cfd_mask.astype(bool) & fluid_mask, iterations=1)
        return interior & ~ring

    # RMSE of hybrid at optimal threshold (from last timed run)
    uh_opt = np.array(uh_opt); vh_opt = np.array(vh_opt); ph_opt = np.array(ph_opt)
    rmse_opt = _gauge_free_rmse(uh_opt, vh_opt, ph_opt, u_cfd, v_cfd, p_cfd,
                                _hybrid_valid(cfd_mask_opt), dx, dy, gradp_scale)

    # ================================================================
    # COVERAGE SWEEP (single run at each 10% increment)
    # ================================================================
    print(f"\n--- Coverage sweep (10% increments, 1 run each) ---")
    target_coverages = np.arange(0.1, 1.0, 0.1)  # 10%, 20%, ..., 90%

    sweep_cov = [0.0]  # start with PINN-only
    sweep_time = [0.0]  # PINN inference is ~instant relative to CFD
    rmse_pinn = _gauge_free_rmse(pinn_u, pinn_v, pinn_p,
                                 u_cfd, v_cfd, p_cfd,
                                 interior, dx, dy, gradp_scale)
    sweep_rmse = [rmse_pinn]

    for target_cov in target_coverages:
        thresh = threshold_for_coverage(router_output, layout, target_cov)
        mask = (router_output >= thresh).astype(np.int32) * layout.astype(np.int32)
        mask = apply_morph_open(mask, layout, args.morph_kernel)
        cov = np.sum(mask) / np.sum(layout)

        sim = make_hybrid_sim(args, pinn_model, mask)
        with contextlib.redirect_stdout(io.StringIO()):
            t0 = time.perf_counter()
            uh, vh, ph = sim.solve()
            t1 = time.perf_counter()
        elapsed = t1 - t0

        uh, vh, ph = np.array(uh), np.array(vh), np.array(ph)
        rmse = _gauge_free_rmse(uh, vh, ph, u_cfd, v_cfd, p_cfd,
                                _hybrid_valid(mask), dx, dy, gradp_scale)

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
    # PLOT: Time vs RMSE
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep_time, sweep_rmse, 'o-', color='tab:blue', linewidth=2, markersize=6)

    # Annotate each point with coverage %
    for i, (t, r, c) in enumerate(zip(sweep_time, sweep_rmse, sweep_cov)):
        ax.annotate(f'{c*100:.0f}%', (t, r), textcoords='offset points',
                    xytext=(5, 5), fontsize=7)

    # Overlay optimal threshold point
    hyb_mean_time = float(np.mean(hybrid_times))
    ax.scatter([hyb_mean_time], [rmse_opt], marker='*', s=250,
               color='red', zorder=5,
               label=f'Optimal (cov={actual_coverage*100:.0f}%)')
    ax.legend(loc='best')

    ax.set_xlabel('Solve Time (s)')
    ax.set_ylabel('RMSE (vs CFD)')
    ax.set_title(f'Time vs RMSE at Coverage Increments — Cylinder (Re={args.Re})')
    fig.tight_layout()

    plot_path = os.path.join(args.output_dir, 'coverage_time_rmse.pdf')
    fig.savefig(plot_path, dpi=1200, bbox_inches='tight')
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
