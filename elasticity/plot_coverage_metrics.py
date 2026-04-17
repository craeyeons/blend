"""
Plot Coverage and Expected Loss Metrics for Hybrid PINN-FDM Elasticity Router.

Generates:
1. Coverage Plot: RMSE vs CFD rejection percentage (displacement L2 error)
2. Expected Loss Comparison: PINN-only, FDM-only, Hybrid
3. Solution comparison: PINN vs Hybrid vs FDM side-by-side
4. Timing summary

Usage:
    python plot_coverage_metrics.py --problem plate_with_hole \
        --pinn-path ./models/pinn_plate_with_hole.h5 \
        --router-path ./router_output/plate_with_hole/beta_0.1/router.weights.h5 \
        --fdm-path ./results/fdm_plate_with_hole.npz

    python plot_coverage_metrics.py --problem plate_with_hole --compute-fdm
"""

import argparse
import os
import time
import json
import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

import matplotlib.pyplot as plt
try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
except ImportError:
    pass
from matplotlib.colors import BoundaryNorm
import matplotlib.patches as mpatches

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


# =========================================================================
# Solution loaders
# =========================================================================

def load_pinn_solution(pinn_model, X, Y, layout):
    """Compute PINN displacement on the grid."""
    xy = np.stack([X.flatten(), Y.flatten()], axis=-1).astype(np.float32)
    out = pinn_model.predict(xy, batch_size=len(xy), verbose=0)
    ux = out[:, 0].reshape(X.shape).astype(np.float32) * layout
    uy = out[:, 1].reshape(X.shape).astype(np.float32) * layout
    return ux, uy


def compute_fdm_solution(args):
    """Run FDM solver and return fields + timing."""
    print("\n[Computing FDM Solution]")
    if args.problem == 'plate_with_hole':
        X, Y, layout, dbc, tbc, bux, buy, btx, bty = create_plate_with_hole(
            Nx=args.nx, Ny=args.ny,
            x_domain=(args.x_min, args.x_max),
            y_domain=(args.y_min, args.y_max),
            hole_center=(args.hole_x, args.hole_y),
            hole_radius=args.hole_radius,
            applied_stress=args.applied_stress,
        )
    else:
        X, Y, layout, dbc, tbc, bux, buy, btx, bty = create_l_bracket(
            Nx=args.nx, Ny=args.ny,
            x_domain=(args.x_min, args.x_max),
            y_domain=(args.y_min, args.y_max),
            corner_x=args.corner_x,
            corner_y=args.corner_y,
            applied_stress=args.applied_stress,
            fillet_radius=args.fillet_radius,
        )

    solver = ElasticitySolver(
        E=args.E, nu=args.nu,
        x_domain=(args.x_min, args.x_max),
        y_domain=(args.y_min, args.y_max),
        Nx=args.nx, Ny=args.ny,
        max_iter=args.max_iter, tol=args.tol,
    )

    t0 = time.time()
    ux, uy, sxx, syy, sxy = solver.solve(layout, dbc, bux, buy, tbc, btx, bty)
    fdm_time = time.time() - t0
    vm = solver.compute_von_mises(sxx, syy, sxy)
    print(f"  FDM solved in {fdm_time:.2f}s")
    return ux, uy, sxx, syy, sxy, vm, X, Y, layout, fdm_time


# =========================================================================
# Error and coverage
# =========================================================================

def compute_l2_error_field(ux_pred, uy_pred, ux_true, uy_true, layout):
    """Per-point L2 displacement error."""
    err = np.sqrt((ux_pred - ux_true) ** 2 + (uy_pred - uy_true) ** 2) * layout
    return err


def compute_coverage_curve(pinn_pred, fdm_truth, router_output, layout, n_points=100):
    """
    RMSE of remaining PINN points as a function of FDM coverage.

    Returns coverage, mse_scores, r2_scores.
    """
    fluid = layout > 0
    pinn_vals = pinn_pred[fluid]
    fdm_vals = fdm_truth[fluid]
    confs = router_output[fluid]

    n = len(pinn_vals)
    global_var = np.var(fdm_vals)

    idx = np.argsort(confs)[::-1]
    sq_err = (fdm_vals[idx] - pinn_vals[idx]) ** 2

    coverage = np.linspace(0, 1, n_points)
    mse = np.zeros(n_points)
    r2 = np.zeros(n_points)

    for i, cov in enumerate(coverage):
        n_fdm = int(cov * n)
        n_pinn = n - n_fdm
        if n_pinn > 0:
            mse[i] = np.mean(sq_err[n_fdm:])
            r2[i] = 1 - mse[i] / max(global_var, 1e-10)
        else:
            mse[i] = 0.0
            r2[i] = 1.0
    return coverage, mse, r2


def compute_expected_losses(residual_field, router_output, layout, beta):
    """Expected losses for PINN-only, FDM-only, hybrid at various thresholds."""
    fluid = layout > 0
    residuals = residual_field[fluid]
    logits = router_output[fluid]

    loss_pinn_only = np.mean(residuals)
    loss_fdm_only = beta

    default_threshold = 0.0
    cfd_mask = logits > default_threshold
    coverage_hybrid = np.mean(cfd_mask)
    pinn_pts = ~cfd_mask
    pinn_res = np.mean(residuals[pinn_pts]) if pinn_pts.sum() > 0 else 0.0
    loss_hybrid = beta * coverage_hybrid + (1 - coverage_hybrid) * pinn_res

    lo, hi = float(logits.min()), float(logits.max())
    margin = max(0.1, (hi - lo) * 0.05)
    thresholds = np.linspace(lo - margin, hi + margin, 500)

    best_loss, best_cov, best_t = float('inf'), 0.0, 0.0
    covs, losses = [], []
    for t in thresholds:
        m = logits > t
        c = np.mean(m)
        pr = np.mean(residuals[~m]) if (~m).sum() > 0 else 0.0
        L = beta * c + (1 - c) * pr
        covs.append(c)
        losses.append(L)
        if L < best_loss:
            best_loss, best_cov, best_t = L, c, t

    return {
        'loss_pinn_only': loss_pinn_only,
        'loss_fdm_only': loss_fdm_only,
        'loss_hybrid': loss_hybrid,
        'coverage_hybrid': coverage_hybrid,
        'default_threshold': default_threshold,
        'optimal_coverage': best_cov,
        'optimal_loss': best_loss,
        'optimal_threshold': best_t,
        'all_thresholds': thresholds,
        'all_coverages': np.array(covs),
        'all_losses': np.array(losses),
    }


# =========================================================================
# Plotting
# =========================================================================

def plot_fdm_solution(ux, uy, vm, sxx, syy, sxy, X, Y, layout,
                      show_hole=None, save_path=None):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fields = [
        (ux, 'ux', 'RdBu_r'), (uy, 'uy', 'RdBu_r'), (vm, 'von Mises', 'hot'),
        (sxx, 'sigma_xx', 'RdBu_r'), (syy, 'sigma_yy', 'RdBu_r'), (sxy, 'sigma_xy', 'RdBu_r'),
    ]
    for ax, (f, title, cmap) in zip(axes.flat, fields):
        masked = np.ma.masked_where(layout == 0, f)
        cf = ax.contourf(X, Y, masked, levels=50, cmap=cmap)
        plt.colorbar(cf, ax=ax)
        if show_hole:
            ax.add_patch(plt.Circle(show_hole[:2], show_hole[2], color='gray', fill=True))
        ax.set_aspect('equal'); ax.set_title(title)
    plt.suptitle('FDM Ground Truth', fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved FDM plot to {save_path}")
    plt.close(fig)


def plot_coverage_curve(coverage, rmse_scores, results, beta, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(coverage * 100, rmse_scores, 'k-', linewidth=2.5, label='RMSE (PINN vs FDM)')
    ax.plot(0, rmse_scores[0], 'o', color='purple', markersize=12, zorder=5,
            label=f'All PINN: RMSE={rmse_scores[0]:.4f}')
    ax.plot(100, rmse_scores[-1], 'o', color='teal', markersize=12, zorder=5,
            label=f'All FDM: RMSE={rmse_scores[-1]:.4f}')
    opt_cov = results['optimal_coverage'] * 100
    idx = int(results['optimal_coverage'] * (len(coverage) - 1))
    ax.axvline(x=opt_cov, color='red', linestyle='--', alpha=0.5,
               label=f'Optimal: {opt_cov:.1f}% FDM')
    ax.axhline(y=0.0, color='green', linestyle='--', linewidth=1.5, alpha=0.3)
    ax.set_xlabel('Coverage (% solved by FDM)', fontsize=14)
    ax.set_ylabel('RMSE', fontsize=14)
    ax.set_xlim(-5, 105); ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_title('Coverage vs RMSE', fontsize=14)
    ax.legend(loc='upper right', fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved coverage plot to {save_path}")
    plt.close(fig)


def plot_expected_loss(results, beta, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(results['all_coverages'] * 100, results['all_losses'], 'b-', linewidth=2,
            label='Hybrid (sweep)')
    ax.axhline(y=results['loss_pinn_only'], color='purple', linestyle='--',
               linewidth=2, label=f"PINN only: {results['loss_pinn_only']:.4f}")
    ax.axhline(y=results['loss_fdm_only'], color='teal', linestyle='--',
               linewidth=2, label=f"FDM only (beta): {results['loss_fdm_only']:.4f}")
    ax.plot(results['optimal_coverage'] * 100, results['optimal_loss'], 'r*',
            markersize=15, zorder=5,
            label=f"Optimal: cov={results['optimal_coverage']*100:.1f}%, L={results['optimal_loss']:.4f}")
    ax.set_xlabel('FDM Coverage (%)', fontsize=14)
    ax.set_ylabel('Expected Loss', fontsize=14)
    ax.set_title(f'Expected Loss Comparison (beta={beta})', fontsize=14)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved expected loss plot to {save_path}")
    plt.close(fig)


def plot_solution_comparison(ux_pinn, uy_pinn, ux_fdm, uy_fdm,
                              ux_hybrid, uy_hybrid, X, Y, layout,
                              cfd_mask, show_hole=None, save_path=None):
    """Side-by-side grid: PINN / Hybrid / FDM for ux, uy, |u|, and error."""
    fluid = layout > 0

    def _quantile_levels(values, n_levels=25, qmin=1.0, qmax=99.0, force_zero_min=False):
        vals = values[np.isfinite(values)]
        if vals.size == 0:
            return np.linspace(0.0, 1.0, n_levels)

        lo = np.percentile(vals, qmin)
        hi = np.percentile(vals, qmax)
        if force_zero_min:
            lo = 0.0

        if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
            lo = float(np.min(vals)) if np.isfinite(np.min(vals)) else 0.0
            hi = float(np.max(vals)) if np.isfinite(np.max(vals)) else 1.0
            if np.isclose(lo, hi):
                hi = lo + 1e-12

        clipped = np.clip(vals, lo, hi)
        levels = np.quantile(clipped, np.linspace(0.0, 1.0, n_levels))
        levels = np.unique(levels)

        if levels.size < 5:
            levels = np.linspace(lo, hi, n_levels)

        return levels

    mag_p = np.sqrt(ux_pinn ** 2 + uy_pinn ** 2)
    mag_h = np.sqrt(ux_hybrid ** 2 + uy_hybrid ** 2)
    mag_f = np.sqrt(ux_fdm ** 2 + uy_fdm ** 2)

    err_p = np.sqrt((ux_pinn - ux_fdm) ** 2 + (uy_pinn - uy_fdm) ** 2)
    err_h = np.sqrt((ux_hybrid - ux_fdm) ** 2 + (uy_hybrid - uy_fdm) ** 2)
    err_fdm = np.zeros_like(mag_f)

    # rows: ux, uy, |u|, error;  cols: PINN, Hybrid, FDM
    rows = [
        ('$u_x$',    [ux_pinn, ux_hybrid, ux_fdm]),
        ('$u_y$',    [uy_pinn, uy_hybrid, uy_fdm]),
        ('$|u|$',    [mag_p, mag_h, mag_f]),
        ('Error',    [err_p, err_h, err_fdm]),
    ]
    col_titles = ['PINN', 'Hybrid', 'FDM']

    nrows, ncols = len(rows), 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))

    for i, (row_label, fields) in enumerate(rows):
        # Shared color mapping per row (same scale across all three columns)
        if row_label == 'Error':
            vals = np.concatenate([f[fluid] for f in fields[:2]])  # skip zero FDM error
            levels = _quantile_levels(vals, n_levels=28, qmin=0.0, qmax=99.5, force_zero_min=True)
            cmap = 'coolwarm'
        else:
            # Use all three fields' combined range for shared scale
            all_vals = np.concatenate([f[fluid] for f in fields])
            levels = _quantile_levels(all_vals, n_levels=28, qmin=0.5, qmax=99.5)
            cmap = 'coolwarm'

        norm = BoundaryNorm(levels, ncolors=256, clip=True)
        line_levels = levels[::max(1, len(levels) // 8)]
        line_levels = np.unique(line_levels)
        if line_levels.size < 2:
            line_levels = levels

        for j, (field, col_title) in enumerate(zip(fields, col_titles)):
            ax = axes[i, j]
            masked = np.ma.masked_where(layout == 0, field)
            cf = ax.contourf(X, Y, masked, levels=levels, cmap=cmap,
                             norm=norm, extend='both')
            cs = ax.contour(X, Y, masked, levels=line_levels,
                            colors='k', linewidths=0.3, alpha=0.45)
            ax.clabel(cs, inline=True, fontsize=6, fmt='%.2e')
            plt.colorbar(cf, ax=ax)
            if show_hole:
                ax.add_patch(plt.Circle(show_hole[:2], show_hole[2],
                                        color='gray', fill=True))
            ax.contour(X, Y, cfd_mask.astype(float), levels=[0.5],
                       colors='lime', linewidths=1.5, linestyles='--')
            ax.set_aspect('equal')
            if i == 0:
                ax.set_title(col_title, fontsize=14, fontweight='bold')
            if j == 0:
                ax.set_ylabel(row_label, fontsize=13, fontweight='bold')

    fig.suptitle('Solution Comparison', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved comparison to {save_path}")
    plt.close(fig)


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Coverage metrics for hybrid PINN-FDM elasticity'
    )
    parser.add_argument('--problem', type=str, default='plate_with_hole',
                        choices=['plate_with_hole', 'l_bracket'])
    parser.add_argument('--pinn-path', type=str,
                        default='./models/pinn_plate_with_hole.h5')
    parser.add_argument('--router-path', type=str,
                        default='./router_output/plate_with_hole/beta_0.1/router.weights.h5')
    parser.add_argument('--fdm-path', type=str, default=None,
                        help='Path to precomputed FDM .npz')
    parser.add_argument('--compute-fdm', action='store_true')
    parser.add_argument('--output-dir', type=str, default='./coverage_output')

    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--nx', type=int, default=200)
    parser.add_argument('--ny', type=int, default=200)
    parser.add_argument('--x-min', type=float, default=-2.0)
    parser.add_argument('--x-max', type=float, default=2.0)
    parser.add_argument('--y-min', type=float, default=-2.0)
    parser.add_argument('--y-max', type=float, default=2.0)
    parser.add_argument('--hole-x', type=float, default=0.0)
    parser.add_argument('--hole-y', type=float, default=0.0)
    parser.add_argument('--hole-radius', type=float, default=0.5)
    parser.add_argument('--applied-stress', type=float, default=10.0)
    parser.add_argument('--corner-x', type=float, default=1.0)
    parser.add_argument('--corner-y', type=float, default=1.0)
    parser.add_argument('--fillet-radius', type=float, default=0.04,
                        help='Fillet radius at L-bracket re-entrant corner (0=sharp)')
    parser.add_argument('--E', type=float, default=1.0)
    parser.add_argument('--nu', type=float, default=0.3)
    parser.add_argument('--max-iter', type=int, default=200000)
    parser.add_argument('--tol', type=float, default=1e-8)
    parser.add_argument('--layers', type=int, nargs='+', default=[128, 128, 128, 128])
    parser.add_argument('--base-filters', type=int, default=32)

    args = parser.parse_args()

    if args.problem == 'l_bracket' and args.x_min == -2.0:
        args.x_min, args.y_min = 0.0, 0.0

    os.makedirs(args.output_dir, exist_ok=True)
    timings = {}

    # --- Domain setup ---
    if args.problem == 'plate_with_hole':
        X, Y, layout, dbc, tbc, bux, buy, btx, bty = create_plate_with_hole(
            Nx=args.nx, Ny=args.ny,
            x_domain=(args.x_min, args.x_max), y_domain=(args.y_min, args.y_max),
            hole_center=(args.hole_x, args.hole_y), hole_radius=args.hole_radius,
            applied_stress=args.applied_stress)
        show_hole = (args.hole_x, args.hole_y, args.hole_radius)
    else:
        X, Y, layout, dbc, tbc, bux, buy, btx, bty = create_l_bracket(
            Nx=args.nx, Ny=args.ny,
            x_domain=(args.x_min, args.x_max), y_domain=(args.y_min, args.y_max),
            corner_x=args.corner_x, corner_y=args.corner_y,
            applied_stress=args.applied_stress,
            fillet_radius=args.fillet_radius)
        show_hole = None

    # --- PINN ---
    print("[1] Loading PINN...")
    network = Network()
    input_range = [(args.x_min, args.x_max), (args.y_min, args.y_max)]
    pinn_model = network.build(num_inputs=2, layers=args.layers,
                               activation='tanh', num_outputs=2,
                               input_range=input_range)
    pinn_model.load_weights(args.pinn_path)

    t0 = time.time()
    ux_pinn, uy_pinn = load_pinn_solution(pinn_model, X, Y, layout)
    timings['pinn_inference_s'] = time.time() - t0
    print(f"  PINN inference: {timings['pinn_inference_s']:.3f}s")

    # --- FDM ---
    print("[2] FDM solution...")
    if args.fdm_path and os.path.exists(args.fdm_path):
        data = np.load(args.fdm_path)
        ux_fdm, uy_fdm = data['ux'], data['uy']
        vm_fdm = data['von_mises']
        sxx_fdm, syy_fdm, sxy_fdm = data['sxx'], data['syy'], data['sxy']
        timings['fdm_solve_s'] = float(data.get('solve_time', 0))
        print(f"  Loaded from {args.fdm_path}")
    elif args.compute_fdm:
        (ux_fdm, uy_fdm, sxx_fdm, syy_fdm, sxy_fdm,
         vm_fdm, _, _, _, fdm_t) = compute_fdm_solution(args)
        timings['fdm_solve_s'] = fdm_t
    else:
        print("  ERROR: provide --fdm-path or --compute-fdm")
        return

    plot_fdm_solution(ux_fdm, uy_fdm, vm_fdm, sxx_fdm, syy_fdm, sxy_fdm,
                      X, Y, layout, show_hole,
                      save_path=os.path.join(args.output_dir, 'fdm_solution.png'))

    # --- Router ---
    print("[3] Loading router...")
    router = RouterCNN(base_filters=args.base_filters)

    # Build router input
    bc_error = compute_bc_error_field(dbc, bux, buy, ux_pinn, uy_pinn, layout)

    # Compute PINN von Mises for input channel
    dx = (args.x_max - args.x_min) / (args.nx - 1)
    dy = (args.y_max - args.y_min) / (args.ny - 1)
    C11 = args.E / (1.0 - args.nu ** 2)
    C12 = args.nu * args.E / (1.0 - args.nu ** 2)
    C66 = args.E / (2.0 * (1.0 + args.nu))
    exx = np.zeros_like(ux_pinn)
    eyy = np.zeros_like(uy_pinn)
    exy = np.zeros_like(ux_pinn)
    exx[1:-1, 1:-1] = (ux_pinn[1:-1, 2:] - ux_pinn[1:-1, :-2]) / (2 * dx)
    eyy[1:-1, 1:-1] = (uy_pinn[2:, 1:-1] - uy_pinn[:-2, 1:-1]) / (2 * dy)
    exy[1:-1, 1:-1] = 0.5 * ((ux_pinn[2:, 1:-1] - ux_pinn[:-2, 1:-1]) / (2 * dy)
                               + (uy_pinn[1:-1, 2:] - uy_pinn[1:-1, :-2]) / (2 * dx))
    sx = C11 * exx + C12 * eyy
    sy = C12 * exx + C11 * eyy
    sxy_p = 2 * C66 * exy
    pinn_vm = np.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3 * sxy_p ** 2).astype(np.float32) * layout

    error_transport = solve_error_transport(
        bc_error, layout, E=args.E, nu=args.nu,
        x_domain=(args.x_min, args.x_max),
        y_domain=(args.y_min, args.y_max),
    )
    inputs = create_router_input(layout, dbc, bux, buy, tbc,
                                  ux_pinn, uy_pinn, pinn_vm, error_transport)
    _ = router(inputs)
    router.load_weights(args.router_path)

    t0 = time.time()
    r = router(tf.constant(inputs, dtype=tf.float32), training=False)
    r = r[0, :, :, 0].numpy()
    timings['router_inference_s'] = time.time() - t0

    # --- Coverage metrics ---
    print("[4] Computing metrics...")
    disp_mag_pinn = np.sqrt(ux_pinn ** 2 + uy_pinn ** 2)
    disp_mag_fdm = np.sqrt(ux_fdm ** 2 + uy_fdm ** 2)
    coverage, mse_scores, r2_scores = compute_coverage_curve(
        disp_mag_pinn, disp_mag_fdm, r, layout)
    rmse_scores = np.sqrt(mse_scores)

    # PDE residual for expected loss
    rc = PINNResidualComputer(pinn_model, E=args.E, nu=args.nu)
    X_tf = tf.constant(X, dtype=tf.float32)
    Y_tf = tf.constant(Y, dtype=tf.float32)
    res = rc.compute_total_residual(X_tf, Y_tf).numpy()
    res_flat = res.flatten()
    med = np.sort(res_flat)[len(res_flat) // 2]
    res_norm = res / (med + 1e-10)

    results = compute_expected_losses(res_norm, r, layout, args.beta)

    plot_coverage_curve(coverage, rmse_scores, results, args.beta,
                        save_path=os.path.join(args.output_dir, 'coverage_rmse.png'))
    plot_expected_loss(results, args.beta,
                       save_path=os.path.join(args.output_dir, 'expected_loss.png'))

    # --- Hybrid solution ---
    print("[5] Building hybrid solution...")
    opt_t = results['optimal_threshold']
    cfd_mask = (r >= opt_t).astype(np.int32) * layout.astype(np.int32)

    hybrid_solver = ElasticitySolver(
        E=args.E, nu=args.nu,
        x_domain=(args.x_min, args.x_max),
        y_domain=(args.y_min, args.y_max),
        Nx=args.nx, Ny=args.ny,
        max_iter=args.max_iter, tol=args.tol,
    )
    t0 = time.time()
    ux_hybrid, uy_hybrid, _, _, _ = hybrid_solver.solve(
        layout, dbc, bux, buy,
        tbc, btx, bty,
        fdm_mask=cfd_mask,
        initial_ux=ux_pinn, initial_uy=uy_pinn,
    )
    timings['hybrid_solve_s'] = time.time() - t0

    plot_solution_comparison(ux_pinn, uy_pinn, ux_fdm, uy_fdm,
                              ux_hybrid, uy_hybrid, X, Y, layout, cfd_mask,
                              show_hole,
                              save_path=os.path.join(args.output_dir, 'solution_comparison.png'))

    # --- Error field ---
    err_pinn = compute_l2_error_field(ux_pinn, uy_pinn, ux_fdm, uy_fdm, layout)
    err_hybrid = compute_l2_error_field(ux_hybrid, uy_hybrid, ux_fdm, uy_fdm, layout)

    # Hybrid total = PINN inference + router inference + hybrid FDM solve
    timings['hybrid_total_s'] = (timings.get('pinn_inference_s', 0)
                                  + timings.get('router_inference_s', 0)
                                  + timings.get('hybrid_solve_s', 0))

    fluid = layout > 0
    metrics = {
        'problem': args.problem,
        'beta': args.beta,
        'grid': f'{args.nx}x{args.ny}',
        'pinn_rmse': float(np.sqrt(np.mean(err_pinn[fluid] ** 2))),
        'hybrid_rmse': float(np.sqrt(np.mean(err_hybrid[fluid] ** 2))),
        'optimal_threshold': float(opt_t),
        'optimal_coverage_pct': float(results['optimal_coverage'] * 100),
        'loss_pinn_only': float(results['loss_pinn_only']),
        'loss_fdm_only': float(results['loss_fdm_only']),
        'loss_hybrid': float(results['optimal_loss']),
        **{k: float(v) for k, v in timings.items()},
    }

    # --- Save timing & metrics ---
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
