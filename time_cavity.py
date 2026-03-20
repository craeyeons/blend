#!/usr/bin/env python3
"""
Time CFD and hybrid solvers for cavity flow.

Only times sim.solve() calls. Router/PINN loading is excluded.
Runs multiple trials and reports mean +/- std.
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

from lib.router import (
    RouterCNN,
    create_router_input,
    create_cavity_setup,
)
from lib.cavity_flow import CavityFlowSimulation, CavityFlowHybridSimulation
from lib.network import Network as CavityNetwork


def compute_uv_from_psi(network, xy):
    xy_tf = tf.constant(xy, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(xy_tf)
        psi_p = network(xy_tf, training=False)
        psi = psi_p[:, 0]
    grad_psi = tape.gradient(psi, xy_tf)
    u = grad_psi[:, 1].numpy()
    v = -grad_psi[:, 0].numpy()
    return u, v


def find_optimal_threshold(residual_field, router_output, layout, beta,
                           lambda_tv=0.01, lambda_entropy=0.1, n_points=200):
    """Find optimal threshold from training loss curve."""
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
        tv_approx = lambda_tv * 4 * cov * (1 - cov)
        loss[i] = cfd_cost + residual_loss + tv_approx

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


def compute_residual_field(pinn_model, X, Y, layout, nu=0.01, rho=1.0):
    from train_router_cavity import CavityPINNResidualComputer

    residual_computer = CavityPINNResidualComputer(
        pinn_model=pinn_model, nu=nu, rho=rho,
        x_domain=(X.min(), X.max()),
        y_domain=(Y.min(), Y.max()),
    )
    X_tf = tf.constant(X, dtype=tf.float32)
    Y_tf = tf.constant(Y, dtype=tf.float32)

    continuity, momentum = residual_computer.compute_residuals(X_tf, Y_tf)
    residual_field = (np.array(continuity) + np.array(momentum)) * layout

    residual_max = np.percentile(residual_field[layout > 0], 95)
    residual_field = np.minimum(residual_field / (residual_max + 1e-10), 1.0) * layout
    return residual_field


def main():
    parser = argparse.ArgumentParser(description='Time CFD and hybrid solvers (cavity)')
    parser.add_argument('--pinn-path', type=str, default='./models/pinn_cavity_flow.h5')
    parser.add_argument('--router-weights', required=True)
    parser.add_argument('--N', type=int, default=100)
    parser.add_argument('--x-min', type=float, default=0.0)
    parser.add_argument('--x-max', type=float, default=1.0)
    parser.add_argument('--y-min', type=float, default=0.0)
    parser.add_argument('--y-max', type=float, default=1.0)
    parser.add_argument('--lid-velocity', type=float, default=1.0)
    parser.add_argument('--Re', type=float, default=100)
    parser.add_argument('--max-iter', type=int, default=200000)
    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--base-filters', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--lambda-tv', type=float, default=0.01)
    parser.add_argument('--nu', type=float, default=0.01)
    parser.add_argument('--rho', type=float, default=1.0)
    parser.add_argument('--n-runs', type=int, default=3)
    args = parser.parse_args()

    N_RUNS = args.n_runs

    # ================================================================
    # SETUP (not timed)
    # ================================================================
    print("=" * 60)
    print("SOLVER TIMING — CAVITY FLOW")
    print("=" * 60)
    print(f"  PINN:   {args.pinn_path}")
    print(f"  Router: {args.router_weights}")
    print(f"  Grid:   {args.N} x {args.N}")
    print(f"  Re={args.Re}")
    print(f"  Runs:   {N_RUNS}")
    print()

    # Domain
    X, Y, layout, bc_mask, bc_u, bc_v, bc_p = create_cavity_setup(
        N=args.N,
        x_domain=(args.x_min, args.x_max),
        y_domain=(args.y_min, args.y_max),
        lid_velocity=args.lid_velocity,
    )

    # PINN
    network = CavityNetwork()
    pinn_model = network.build(
        num_inputs=2, layers=[32, 16, 16, 32],
        activation='swish', num_outputs=2,
    )
    pinn_model.load_weights(args.pinn_path)

    # Router output
    xy_flat = np.stack([X.flatten(), Y.flatten()], axis=-1).astype(np.float32)
    u_flat, v_flat = compute_uv_from_psi(pinn_model, xy_flat)
    psi_p = pinn_model.predict(xy_flat, batch_size=len(xy_flat), verbose=0)

    pinn_u = u_flat.reshape(X.shape).astype(np.float32) * layout
    pinn_v = v_flat.reshape(X.shape).astype(np.float32) * layout
    pinn_p = psi_p[:, 1].reshape(X.shape).astype(np.float32) * layout

    inputs = create_router_input(layout, bc_mask, bc_u, bc_v, bc_p,
                                 pinn_u, pinn_v, pinn_p)
    router = RouterCNN(base_filters=args.base_filters, temperature=args.temperature)
    _ = router(inputs)
    router.load_weights(args.router_weights)
    router_output = router(inputs, training=False).numpy().squeeze()

    # Residual field + optimal threshold
    residual_field = compute_residual_field(
        pinn_model, X, Y, layout, nu=args.nu, rho=args.rho,
    )

    optimal_threshold, optimal_coverage = find_optimal_threshold(
        residual_field, router_output, layout, args.beta,
        args.lambda_tv,
    )

    cfd_mask = (router_output >= optimal_threshold).astype(np.int32) * layout.astype(np.int32)
    actual_coverage = np.sum(cfd_mask) / np.sum(layout)

    print(f"  Optimal threshold: {optimal_threshold:.6f}")
    print(f"  CFD coverage:      {actual_coverage * 100:.2f}%")
    print()

    # ================================================================
    # WARMUP (JIT compilation, not counted)
    # ================================================================
    print("--- Warmup: CFD solve (JIT compile) ---")
    with contextlib.redirect_stdout(io.StringIO()):
        CavityFlowSimulation(
            Re=args.Re, N=args.N,
            max_iter=args.max_iter, tol=args.tol,
        ).solve()
    print("  done")

    print("--- Warmup: Hybrid solve (JIT compile) ---")
    with contextlib.redirect_stdout(io.StringIO()):
        CavityFlowHybridSimulation(
            network=pinn_model, uv_func=compute_uv_from_psi,
            mask=cfd_mask,
            Re=args.Re, N=args.N,
            max_iter=args.max_iter, tol=args.tol,
        ).solve()
    print("  done")
    print()

    # ================================================================
    # TIME CFD
    # ================================================================
    cfd_times = []
    for i in range(N_RUNS):
        sim = CavityFlowSimulation(
            Re=args.Re, N=args.N,
            max_iter=args.max_iter, tol=args.tol,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            t0 = time.perf_counter()
            sim.solve()
            t1 = time.perf_counter()
        elapsed = t1 - t0
        cfd_times.append(elapsed)
        print(f"  CFD  run {i + 1}/{N_RUNS}: {elapsed:.4f} s")

    # ================================================================
    # TIME HYBRID
    # ================================================================
    hybrid_times = []
    for i in range(N_RUNS):
        sim = CavityFlowHybridSimulation(
            network=pinn_model, uv_func=compute_uv_from_psi,
            mask=cfd_mask,
            Re=args.Re, N=args.N,
            max_iter=args.max_iter, tol=args.tol,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            t0 = time.perf_counter()
            sim.solve()
            t1 = time.perf_counter()
        elapsed = t1 - t0
        hybrid_times.append(elapsed)
        print(f"  Hybrid run {i + 1}/{N_RUNS}: {elapsed:.4f} s")

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
