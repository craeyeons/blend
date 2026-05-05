#!/usr/bin/env python3
"""
Time FDM and hybrid solvers for the generalised L-bracket elasticity setup
(one shared multi-load router, one decomposed PINN pair per load config).

Loops over configs in a JSON file (same format as configs_multi_lbracket.json).
Each config provides its own V-bar/H-bar PINN weights and load parameters;
a single router weight file is shared across all configs.

Usage:
    python time_elasticity_multi_lbracket.py \
        --config configs_multi_lbracket.json \
        --router-weights ./router_output/l_bracket_multi/beta_1.1/router.weights.h5 \
        --split test --n-runs 3
"""

import argparse
import contextlib
import io
import json
import os
import time

import numpy as np
import cv2
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
try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
except ImportError:
    pass

from lib.network import Network
from lib.domains import create_l_bracket
from lib.solver import ElasticitySolver, compute_stress_field
from train_pinn_decomposed import load_decomposed_pinn, blend_solutions
from lib.router import (
    RouterCNN,
    PINNResidualComputer,
    create_router_input,
    compute_bc_error_field,
    solve_error_transport,
)
from time_elasticity import (
    apply_morph_open,
    find_optimal_threshold,
    compute_residual_field,
    router_predict,
)


def resolve_load(cfg, fallback_mag):
    mag = cfg.get('load_magnitude', fallback_mag)
    angle = cfg.get('load_angle', None)
    edge = cfg.get('load_edge', 'top')
    if angle is not None:
        theta = np.deg2rad(angle)
        tx = float(mag * np.cos(theta))
        ty = float(mag * np.sin(theta))
    else:
        tx = -mag if edge == 'top' else mag
        ty = 0.0
    return edge, tx, ty, mag, angle


def time_one_config(cfg, router, args):
    edge, tx, ty, mag, angle = resolve_load(cfg, args.applied_stress)
    tag = cfg.get('tag', f'{edge}_m{mag}_a{angle}')
    print(f"\n{'=' * 60}\nCONFIG: {tag}\n  edge={edge}  mag={mag}  angle={angle}\n{'=' * 60}")

    # --- Domain ---
    X, Y, layout, dbc, tbc, bux, buy, btx, bty = create_l_bracket(
        Nx=args.nx, Ny=args.ny,
        x_domain=(args.x_min, args.x_max),
        y_domain=(args.y_min, args.y_max),
        corner_x=args.corner_x, corner_y=args.corner_y,
        applied_stress=args.applied_stress,
        fillet_radius=args.fillet_radius,
        load_edge=edge, load_tx=tx, load_ty=ty,
    )

    # --- PINN (decomposed pair from this config) ---
    model_v, model_h = load_decomposed_pinn(
        cfg['pinn_vbar_path'], cfg['pinn_hbar_path'],
        layers=args.layers, activation='tanh',
        x_min=args.x_min, x_max=args.x_max,
        y_min=args.y_min, y_max=args.y_max,
        corner_x=args.corner_x, corner_y=args.corner_y)

    # Dummy single-domain PINN only for PINNResidualComputer
    network = Network()
    pinn_model = network.build(
        num_inputs=2, layers=args.layers,
        activation='tanh', num_outputs=2,
        input_range=[(args.x_min, args.x_max), (args.y_min, args.y_max)],
        hard_bc='l_bracket',
        hard_bc_params={'corner_x': args.corner_x, 'corner_y': args.corner_y})

    dx = (args.x_max - args.x_min) / (args.nx - 1)
    dy = (args.y_max - args.y_min) / (args.ny - 1)

    # Initial PINN fields for setup
    pinn_ux, pinn_uy = blend_solutions(
        model_v, model_h, X, Y, layout, args.corner_x, args.corner_y)
    _, _, _, pinn_vm = compute_stress_field(
        pinn_ux, pinn_uy, layout, dx, dy, E=args.E, nu=args.nu)
    pinn_vm = pinn_vm.astype(np.float32) * layout

    bc_error = compute_bc_error_field(dbc, bux, buy, pinn_ux, pinn_uy, layout)
    error_transport = solve_error_transport(
        bc_error, layout, E=args.E, nu=args.nu,
        x_domain=(args.x_min, args.x_max), y_domain=(args.y_min, args.y_max))
    inputs = create_router_input(
        layout, dbc, bux, buy, tbc, pinn_ux, pinn_uy, pinn_vm, error_transport)

    router_output = router(tf.constant(inputs, dtype=tf.float32),
                           training=False).numpy().squeeze()

    # Optimal threshold for this config
    residual_field = compute_residual_field(
        pinn_model, X, Y, layout, E=args.E, nu=args.nu)
    opt_t, opt_cov = find_optimal_threshold(
        residual_field, router_output, layout, args.beta)
    fdm_mask_opt = apply_morph_open(
        (router_output >= opt_t).astype(np.int32) * layout.astype(np.int32),
        layout, args.morph_kernel)
    actual_cov = np.sum(fdm_mask_opt) / np.sum(layout)
    print(f"  opt threshold={opt_t:.4f}  target={opt_cov*100:.1f}%  actual={actual_cov*100:.1f}%")

    # --- Warmup ---
    with contextlib.redirect_stdout(io.StringIO()):
        solver = ElasticitySolver(
            E=args.E, nu=args.nu,
            x_domain=(args.x_min, args.x_max),
            y_domain=(args.y_min, args.y_max),
            Nx=args.nx, Ny=args.ny, max_iter=100, tol=args.tol)
        solver.solve(layout, dbc, bux, buy, tbc, btx, bty)
        solver.solve(layout, dbc, bux, buy, tbc, btx, bty,
                     fdm_mask=fdm_mask_opt,
                     initial_ux=pinn_ux, initial_uy=pinn_uy)
    _ = router(tf.constant(inputs, dtype=tf.float32), training=False)

    # --- FDM timing ---
    fdm_times = []
    ux_fdm = uy_fdm = None
    for i in range(args.n_runs):
        with contextlib.redirect_stdout(io.StringIO()):
            solver = ElasticitySolver(
                E=args.E, nu=args.nu,
                x_domain=(args.x_min, args.x_max),
                y_domain=(args.y_min, args.y_max),
                Nx=args.nx, Ny=args.ny,
                max_iter=args.max_iter, tol=args.tol)
            t0 = time.perf_counter()
            ux_fdm, uy_fdm, _, _, _ = solver.solve(
                layout, dbc, bux, buy, tbc, btx, bty)
            t1 = time.perf_counter()
        fdm_times.append(t1 - t0)

    # --- Hybrid timing (full pipeline) ---
    hyb_times, pinn_times, router_times, solve_times = [], [], [], []
    for i in range(args.n_runs):
        t_all_s = time.perf_counter()

        t0 = time.perf_counter()
        h_ux, h_uy = blend_solutions(
            model_v, model_h, X, Y, layout, args.corner_x, args.corner_y)
        _, _, _, h_vm = compute_stress_field(
            h_ux, h_uy, layout, dx, dy, E=args.E, nu=args.nu)
        h_vm = h_vm.astype(np.float32) * layout
        pinn_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _, h_mask = router_predict(
            router, h_ux, h_uy, h_vm, layout,
            dbc, bux, buy, tbc, threshold=opt_t,
            E=args.E, nu=args.nu,
            x_domain=(args.x_min, args.x_max),
            y_domain=(args.y_min, args.y_max),
            morph_kernel=args.morph_kernel)
        router_times.append(time.perf_counter() - t0)

        with contextlib.redirect_stdout(io.StringIO()):
            solver = ElasticitySolver(
                E=args.E, nu=args.nu,
                x_domain=(args.x_min, args.x_max),
                y_domain=(args.y_min, args.y_max),
                Nx=args.nx, Ny=args.ny,
                max_iter=args.max_iter, tol=args.tol)
            t0 = time.perf_counter()
            ux_h, uy_h, _, _, _ = solver.solve(
                layout, dbc, bux, buy, tbc, btx, bty,
                fdm_mask=h_mask,
                initial_ux=h_ux, initial_uy=h_uy)
            solve_times.append(time.perf_counter() - t0)

        hyb_times.append(time.perf_counter() - t_all_s)

    fluid = layout > 0
    disp_fdm = np.sqrt(ux_fdm ** 2 + uy_fdm ** 2)
    disp_hyb = np.sqrt(np.array(ux_h) ** 2 + np.array(uy_h) ** 2)
    rmse = float(np.sqrt(np.mean((disp_hyb[fluid] - disp_fdm[fluid]) ** 2)))

    fdm_mean = float(np.mean(fdm_times))
    hyb_mean = float(np.mean(hyb_times))
    speedup = fdm_mean / hyb_mean if hyb_mean > 0 else float('inf')

    print(f"  FDM    : {fdm_mean:.4f}s")
    print(f"  Hybrid : {hyb_mean:.4f}s  (pinn={np.mean(pinn_times):.4f}s "
          f"router={np.mean(router_times):.4f}s solve={np.mean(solve_times):.4f}s)")
    print(f"  Speedup: {speedup:.2f}x   RMSE: {rmse:.6f}")

    return {
        'tag': tag,
        'load_edge': edge, 'load_magnitude': mag, 'load_angle': angle,
        'fdm_mean_s': fdm_mean, 'fdm_std_s': float(np.std(fdm_times)),
        'hybrid_mean_s': hyb_mean, 'hybrid_std_s': float(np.std(hyb_times)),
        'pinn_mean_s': float(np.mean(pinn_times)),
        'router_mean_s': float(np.mean(router_times)),
        'solve_mean_s': float(np.mean(solve_times)),
        'speedup': speedup,
        'coverage': float(actual_cov),
        'threshold': float(opt_t),
        'rmse': rmse,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True,
                   help='JSON with train/test load configs (configs_multi_lbracket.json format)')
    p.add_argument('--router-weights', type=str, required=True,
                   help='Shared multi-load router weights')
    p.add_argument('--split', type=str, default='test',
                   choices=['train', 'test', 'both'])
    p.add_argument('--output-dir', type=str, default='./timing_output/l_bracket_multi')

    p.add_argument('--nx', type=int, default=200)
    p.add_argument('--ny', type=int, default=200)
    p.add_argument('--x-min', type=float, default=0.0)
    p.add_argument('--x-max', type=float, default=2.0)
    p.add_argument('--y-min', type=float, default=0.0)
    p.add_argument('--y-max', type=float, default=2.0)
    p.add_argument('--corner-x', type=float, default=1.0)
    p.add_argument('--corner-y', type=float, default=1.0)
    p.add_argument('--fillet-radius', type=float, default=0.04)
    p.add_argument('--applied-stress', type=float, default=10.0)

    p.add_argument('--E', type=float, default=1.0)
    p.add_argument('--nu', type=float, default=0.3)
    p.add_argument('--max-iter', type=int, default=200000)
    p.add_argument('--tol', type=float, default=1e-8)

    p.add_argument('--layers', type=int, nargs='+', default=[128, 128, 128, 128])
    p.add_argument('--base-filters', type=int, default=32)
    p.add_argument('--beta', type=float, default=1.1)
    p.add_argument('--n-runs', type=int, default=3)
    p.add_argument('--morph-kernel', type=int, default=5)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as f:
        cfgs = json.load(f)

    if args.split == 'both':
        run_cfgs = [('train', c) for c in cfgs.get('train', [])] + \
                   [('test', c) for c in cfgs.get('test', [])]
    else:
        run_cfgs = [(args.split, c) for c in cfgs.get(args.split, [])]

    # Build router once (shape-probed from first config's domain)
    router = RouterCNN(base_filters=args.base_filters)
    first_cfg = run_cfgs[0][1]
    edge0, tx0, ty0, _, _ = resolve_load(first_cfg, args.applied_stress)
    X0, Y0, lay0, dbc0, tbc0, bux0, buy0, _, _ = create_l_bracket(
        Nx=args.nx, Ny=args.ny,
        x_domain=(args.x_min, args.x_max), y_domain=(args.y_min, args.y_max),
        corner_x=args.corner_x, corner_y=args.corner_y,
        applied_stress=args.applied_stress, fillet_radius=args.fillet_radius,
        load_edge=edge0, load_tx=tx0, load_ty=ty0)
    probe = np.zeros((1, args.ny, args.nx, 9), dtype=np.float32)
    _ = router(tf.constant(probe))
    router.load_weights(args.router_weights)
    print(f"Loaded router: {args.router_weights}")

    results = []
    for split, cfg in run_cfgs:
        r = time_one_config(cfg, router, args)
        r['split'] = split
        results.append(r)

    # --- Summary ---
    out_path = os.path.join(args.output_dir, 'multi_timing_summary.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary to {out_path}")

    tags = [r['tag'] for r in results]
    speedups = [r['speedup'] for r in results]
    fig, ax = plt.subplots(figsize=(max(6, len(tags) * 0.8), 4))
    ax.bar(tags, speedups, color='tab:blue')
    ax.set_ylabel('Speedup (FDM / hybrid)')
    ax.set_title('L-bracket multi-load timing')
    ax.axhline(1.0, color='k', linestyle='--', linewidth=0.8)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'speedup_by_config.pdf'),
                dpi=1200, bbox_inches='tight')
    plt.close(fig)

    print("\n=== Summary ===")
    print(f"{'tag':<30} {'fdm':>8} {'hyb':>8} {'speedup':>8} {'rmse':>10}")
    for r in results:
        print(f"{r['tag']:<30} {r['fdm_mean_s']:>8.4f} {r['hybrid_mean_s']:>8.4f} "
              f"{r['speedup']:>8.2f} {r['rmse']:>10.6f}")


if __name__ == '__main__':
    main()
