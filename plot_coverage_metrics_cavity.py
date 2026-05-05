"""
Plot Coverage and Expected Loss Metrics for Hybrid PINN-CFD Router (CAVITY FLOW).

This script generates two key plots:
1. Coverage Plot: Accuracy (L2 loss vs CFD ground truth) as a function of 
   rejection percentage (fraction sent to CFD solution)
   
2. Expected Loss Comparison: Compare expected true loss for:
   - ONLY PINN
   - ONLY iterative solution (CFD) with cost β per node
   - Hybrid system using the router

Usage:
    python plot_coverage_metrics_cavity.py --pinn-path <path> --router-weights <path>
    python plot_coverage_metrics_cavity.py --compute-cfd  # Compute CFD solution if not available
"""

import argparse
import os
import time
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'no-latex'])
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

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
    compute_pinn_residual_field,
    create_router_input,
    compute_bc_error_field,
    solve_error_transport,
    create_cavity_setup,
)
from lib.cavity_flow import CavityFlowSimulation, CavityFlowHybridSimulation
from lib.network import Network as CavityNetwork


def compute_uv_from_psi(network, xy):
    """
    Compute (u, v) from stream function: u = ∂ψ/∂y, v = -∂ψ/∂x
    """
    xy_tf = tf.constant(xy, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(xy_tf)
        psi_p = network(xy_tf, training=False)
        psi = psi_p[:, 0]
    grad_psi = tape.gradient(psi, xy_tf)
    u = grad_psi[:, 1].numpy()   # ∂ψ/∂y
    v = -grad_psi[:, 0].numpy()  # -∂ψ/∂x
    return u, v


def load_pinn_solution(pinn_model, X, Y, layout):
    """
    Compute PINN solution on the grid.
    
    Parameters:
    -----------
    pinn_model : tf.keras.Model
        Pre-trained PINN model (outputs psi, p)
    X, Y : ndarray
        Coordinate grids (N, N)
    layout : ndarray
        Fluid domain mask (all 1s for cavity)
        
    Returns:
    --------
    u_pinn, v_pinn, p_pinn : ndarray
        PINN predictions on the grid
    """
    xy_flat = np.stack([X.flatten(), Y.flatten()], axis=-1).astype(np.float32)
    
    # Get velocities from stream function
    u_flat, v_flat = compute_uv_from_psi(pinn_model, xy_flat)
    
    # Get pressure
    psi_p = pinn_model.predict(xy_flat, batch_size=len(xy_flat), verbose=0)
    
    u_pinn = u_flat.reshape(X.shape).astype(np.float32)
    v_pinn = v_flat.reshape(X.shape).astype(np.float32)
    p_pinn = psi_p[:, 1].reshape(X.shape).astype(np.float32)
    # Shift PINN pressure so p(0,0)=0 to match CFD reference
    p_pinn = p_pinn - p_pinn[0, 0]

    # Mask out obstacle regions (no obstacles in cavity)
    u_pinn = u_pinn * layout
    v_pinn = v_pinn * layout
    p_pinn = p_pinn * layout
    
    return u_pinn, v_pinn, p_pinn


def compute_cfd_solution(args):
    """
    Compute CFD solution using iterative solver.
    """
    print("\n[Computing CFD Solution]")
    print("  This may take a while...")
    
    sim = CavityFlowSimulation(
        Re=args.Re,
        N=args.N,
        max_iter=args.max_iter,
        tol=args.tol
    )
    
    start_time = time.time()
    u_cfd, v_cfd, p_cfd = sim.solve()
    cfd_time = time.time() - start_time
    
    print(f"  ✓ CFD solution computed in {cfd_time:.2f} seconds")
    
    return u_cfd, v_cfd, p_cfd, sim.X, sim.Y, cfd_time


def compute_l2_error_field(u_pred, v_pred, p_pred, u_true, v_true, p_true,
                            layout, X=None, Y=None, interface_mask=None):
    """Per-point gauge-invariant L2 error.

    Uses |∇p - ∇p_true|² (gauge-free) instead of (p - p_true)². Excludes a
    1-cell ring around the fluid boundary and, if `interface_mask` is given
    (hybrid PINN/CFD split), a 1-cell ring around that interface.
    """
    from scipy.ndimage import binary_erosion, binary_dilation

    error_u = (u_pred - u_true) ** 2
    error_v = (v_pred - v_true) ** 2

    if X is not None and Y is not None:
        dx = float(X[0, 1] - X[0, 0])
        dy = float(Y[1, 0] - Y[0, 0])
    else:
        dx = dy = 1.0

    fluid = layout > 0
    pxp, pyp = np.gradient(p_pred, dy, dx)
    pxt, pyt = np.gradient(p_true, dy, dx)
    gradp_scale = float(np.max(pxt[fluid] ** 2 + pyt[fluid] ** 2)) + 1e-10
    error_p = ((pxp - pxt) ** 2 + (pyp - pyt) ** 2) / gradp_scale

    error_field = np.sqrt(error_u + error_v + error_p) * layout

    valid = binary_erosion(fluid, iterations=1)
    if interface_mask is not None:
        m = interface_mask.astype(bool) & fluid
        ring = binary_dilation(m, iterations=1) & ~binary_erosion(m, iterations=1)
        valid = valid & ~ring
    error_field = np.where(valid, error_field, 0.0)
    return error_field


def plot_cfd_solution(u_cfd, v_cfd, p_cfd, X, Y, layout, save_path=None):
    """
    Plot the CFD solution fields for cavity flow.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Velocity magnitude
    vel_mag = np.sqrt(u_cfd**2 + v_cfd**2)
    
    # Mask obstacle regions
    u_plot = np.ma.masked_where(layout == 0, u_cfd)
    v_plot = np.ma.masked_where(layout == 0, v_cfd)
    p_plot = np.ma.masked_where(layout == 0, p_cfd)
    vel_plot = np.ma.masked_where(layout == 0, vel_mag)
    
    # Plot velocity magnitude
    ax = axes[0, 0]
    cf = ax.contourf(X, Y, vel_plot, levels=50, cmap='rainbow')
    plt.colorbar(cf, ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('CFD: Velocity Magnitude |u|')
    ax.set_aspect('equal')
    
    # Plot u-velocity
    ax = axes[0, 1]
    cf = ax.contourf(X, Y, u_plot, levels=50, cmap='rainbow')
    plt.colorbar(cf, ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('CFD: u-velocity')
    ax.set_aspect('equal')
    
    # Plot v-velocity
    ax = axes[1, 0]
    cf = ax.contourf(X, Y, v_plot, levels=50, cmap='rainbow')
    plt.colorbar(cf, ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('CFD: v-velocity')
    ax.set_aspect('equal')
    
    # Plot pressure
    ax = axes[1, 1]
    cf = ax.contourf(X, Y, p_plot, levels=50, cmap='rainbow')
    plt.colorbar(cf, ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('CFD: Pressure')
    ax.set_aspect('equal')
    
    plt.suptitle('CFD Ground Truth Solution (Lid-Driven Cavity)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=1200, bbox_inches='tight')
        print(f"  Saved CFD solution plot to {save_path}")
    
    plt.close(fig)
    
    return fig


def compute_hybrid_solution(pinn_model, router_output, layout, threshold, args):
    """
    Compute actual hybrid solution by running CFD in CFD regions with PINN boundary conditions.
    """
    # Create binary mask from router output with morphological opening
    cfd_mask = (router_output >= threshold).astype(np.int32)
    cfd_mask = cfd_mask * layout.astype(np.int32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (args.morph_kernel, args.morph_kernel))
    cfd_mask = cv2.morphologyEx(cfd_mask.astype(np.uint8), cv2.MORPH_OPEN,
                                kernel).astype(np.int32) * layout.astype(np.int32)

    cfd_fraction = np.sum(cfd_mask) / np.sum(layout) * 100
    print(f"  CFD region: {cfd_fraction:.1f}%")
    print(f"  PINN region: {100 - cfd_fraction:.1f}%")
    
    # Create hybrid simulation
    sim = CavityFlowHybridSimulation(
        network=pinn_model,
        uv_func=compute_uv_from_psi,
        mask=cfd_mask,
        Re=args.Re,
        N=args.N,
        max_iter=args.max_iter,
        tol=args.tol
    )
    
    # Solve and time it
    solve_start = time.time()
    u_hybrid, v_hybrid, p_hybrid = sim.solve()
    solve_time = time.time() - solve_start
    
    # Convert to numpy if needed
    u_hybrid = np.array(u_hybrid)
    v_hybrid = np.array(v_hybrid)
    p_hybrid = np.array(p_hybrid)
    
    return u_hybrid, v_hybrid, p_hybrid, cfd_mask, solve_time


def plot_hybrid_solution(u_hybrid, v_hybrid, p_hybrid, X, Y, layout, cfd_mask,
                         threshold, coverage, save_path=None, title=None, show_info=True):
    """
    Plot the hybrid solution fields with CFD/PINN region overlay.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Velocity magnitude
    vel_mag = np.sqrt(u_hybrid**2 + v_hybrid**2)
    
    # Mask obstacle regions
    u_plot = np.ma.masked_where(layout == 0, u_hybrid)
    v_plot = np.ma.masked_where(layout == 0, v_hybrid)
    p_plot = np.ma.masked_where(layout == 0, p_hybrid)
    vel_plot = np.ma.masked_where(layout == 0, vel_mag)
    
    # Plot velocity magnitude
    ax = axes[0, 0]
    cf = ax.contourf(X, Y, vel_plot, levels=50, cmap='rainbow')
    plt.colorbar(cf, ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Hybrid: Velocity Magnitude |u|')
    ax.set_aspect('equal')
    
    # Plot u-velocity
    ax = axes[0, 1]
    cf = ax.contourf(X, Y, u_plot, levels=50, cmap='rainbow')
    plt.colorbar(cf, ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Hybrid: u-velocity')
    ax.set_aspect('equal')
    
    # Plot v-velocity
    ax = axes[0, 2]
    cf = ax.contourf(X, Y, v_plot, levels=50, cmap='rainbow')
    plt.colorbar(cf, ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Hybrid: v-velocity')
    ax.set_aspect('equal')
    
    # Plot pressure
    ax = axes[1, 0]
    cf = ax.contourf(X, Y, p_plot, levels=50, cmap='rainbow')
    plt.colorbar(cf, ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Hybrid: Pressure')
    ax.set_aspect('equal')
    
    # Plot CFD/PINN region map
    ax = axes[1, 1]
    region_map = np.where(layout == 0, 0.5, np.where(cfd_mask, 1.0, 0.0))
    im = ax.imshow(region_map, extent=[X.min(), X.max(), Y.min(), Y.max()],
                   origin='lower', cmap='RdYlBu', vmin=0, vmax=1, aspect='equal')
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 0.5, 1])
    cbar.ax.set_yticklabels(['PINN', 'N/A', 'CFD'])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Solver Region Map (threshold={threshold:.4f})')
    ax.set_aspect('equal')
    
    # Add text info panel
    ax = axes[1, 2]
    ax.axis('off')
    if show_info:
        info_text = f"""Hybrid Solution Summary

    Optimal Threshold: {threshold:.6f}
    CFD Coverage: {coverage*100:.2f}%
    PINN Coverage: {(1-coverage)*100:.2f}%

    Legend:
    • Blue regions: PINN solver
    • Red regions: CFD solver

    The hybrid solution uses CFD for regions
    where router_output > threshold, and
    PINN elsewhere.
    """
        ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    suptitle = title if title is not None else 'Hybrid PINN-CFD Solution (Cavity Flow)'
    plt.suptitle(suptitle, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=1200, bbox_inches='tight')
        print(f"  Saved hybrid solution plot to {save_path}")
    
    plt.close(fig)
    
    return fig


def plot_solution_comparison(u_pinn, v_pinn, p_pinn,
                             u_cfd, v_cfd, p_cfd,
                             u_hybrid, v_hybrid, p_hybrid,
                             X, Y, layout, cfd_mask,
                             save_path=None):
    """5x3 panel: rows = p, u, v, |velocity|, error; cols = PINN, Hybrid, CFD.

    Each row shares a colour scale across columns. Pressure is gauge-aligned
    via fluid-median subtraction. The hybrid column is shaded to mark the
    CFD-solved subdomain.
    """
    from matplotlib.colors import Normalize

    fluid = layout > 0

    p_pinn_c = p_pinn - np.median(p_pinn[fluid])
    p_hyb_c = p_hybrid - np.median(p_hybrid[fluid])
    p_cfd_c = p_cfd - np.median(p_cfd[fluid])

    vel_pinn = np.sqrt(u_pinn**2 + v_pinn**2)
    vel_hyb = np.sqrt(u_hybrid**2 + v_hybrid**2)
    vel_cfd = np.sqrt(u_cfd**2 + v_cfd**2)

    err_pinn = compute_l2_error_field(u_pinn, v_pinn, p_pinn,
                                      u_cfd, v_cfd, p_cfd,
                                      layout, X=X, Y=Y)
    err_hyb = compute_l2_error_field(u_hybrid, v_hybrid, p_hybrid,
                                     u_cfd, v_cfd, p_cfd,
                                     layout, X=X, Y=Y,
                                     interface_mask=cfd_mask)
    err_cfd = np.zeros_like(u_cfd)

    rows = [
        ('p',     [p_pinn_c, p_hyb_c, p_cfd_c],  'coolwarm', False),
        ('u',     [u_pinn,   u_hybrid, u_cfd],   'coolwarm', False),
        ('v',     [v_pinn,   v_hybrid, v_cfd],   'coolwarm', False),
        ('|u|',   [vel_pinn, vel_hyb, vel_cfd],  'viridis',  True),
        ('error', [err_pinn, err_hyb, err_cfd],  'magma',    True),
    ]
    col_titles = ['PINN', 'Hybrid', 'CFD']

    fig, axes = plt.subplots(len(rows), 3, figsize=(16, 4 * len(rows)))

    for i, (label, fields, cmap, nonneg) in enumerate(rows):
        stacked = np.concatenate([f[fluid] for f in fields])
        if nonneg:
            vmin, vmax = 0.0, float(np.max(stacked))
        else:
            absmax = float(np.max(np.abs(stacked)))
            vmin, vmax = -absmax, absmax

        for j, f in enumerate(fields):
            ax = axes[i, j]
            data = np.ma.masked_where(layout == 0, f)
            cf = ax.contourf(X, Y, data, levels=50, cmap=cmap,
                             norm=Normalize(vmin=vmin, vmax=vmax))
            plt.colorbar(cf, ax=ax, label=label)
            if j == 1:
                ax.contourf(X, Y, cfd_mask.astype(float),
                            levels=[0.5, 1.5], colors=['black'], alpha=0.25)
                ax.contour(X, Y, cfd_mask.astype(float), levels=[0.5],
                           colors='lime', linewidths=1.5)
            ax.set_aspect('equal')
            if i == 0:
                ax.set_title(col_titles[j])
            if j == 0:
                ax.set_ylabel(label)
            if i == len(rows) - 1:
                ax.set_xlabel('x')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=1200, bbox_inches='tight')
        print(f"  Saved solution comparison to {save_path}")
    plt.close(fig)


def compute_coverage_curve(pinn_pred, cfd_truth, router_output, layout,
                            n_points=100, X=None, Y=None):
    """
    Compute MSE and R² of PINN points as a function of CFD coverage.
    Uses gauge-invariant pressure-gradient error when X, Y provided.
    """
    u_pinn, v_pinn, p_pinn = pinn_pred
    u_cfd, v_cfd, p_cfd = cfd_truth

    fluid_mask = layout > 0
    n_fluid = np.sum(fluid_mask)

    # Error field (gauge-invariant in p)
    error_u = (u_pinn - u_cfd) ** 2
    error_v = (v_pinn - v_cfd) ** 2
    if X is not None and Y is not None:
        dx = float(X[0, 1] - X[0, 0])
        dy = float(Y[1, 0] - Y[0, 0])
    else:
        dx = dy = 1.0
    pxp, pyp = np.gradient(p_pinn, dy, dx)
    pxc, pyc = np.gradient(p_cfd, dy, dx)
    gradp_scale = float(np.max(pxc[fluid_mask] ** 2 + pyc[fluid_mask] ** 2)) + 1e-10
    error_p = ((pxp - pxc) ** 2 + (pyp - pyc) ** 2) / gradp_scale
    p_range = gradp_scale  # keep name for downstream R² normalization

    error_field = error_u + error_v + error_p
    
    # Get error and confidence at fluid points
    errors = error_field[fluid_mask]
    confidences = router_output[fluid_mask]
    
    # Sort by confidence descending (highest confidence first)
    sorted_idx = np.argsort(confidences)[::-1]
    sorted_errors = errors[sorted_idx]
    
    # Compute metrics at different coverage levels
    coverage = np.linspace(0, 1, n_points)
    mse_scores = np.zeros(n_points)
    r2_scores = np.zeros(n_points)
    
    # Total variance for R² computation
    total_variance = np.var(np.concatenate([
        u_cfd[fluid_mask], v_cfd[fluid_mask], p_cfd[fluid_mask] / p_range
    ]))
    
    for i, cov in enumerate(coverage):
        n_cfd = int(cov * n_fluid)
        
        if n_cfd == 0:
            # All PINN
            mse_scores[i] = np.mean(errors)
            ss_res = np.sum(errors)
        elif n_cfd >= n_fluid:
            # All CFD
            mse_scores[i] = 0.0
            ss_res = 0.0
        else:
            # Hybrid
            pinn_errors = sorted_errors[n_cfd:]
            mse_scores[i] = np.mean(pinn_errors)
            ss_res = np.sum(sorted_errors[n_cfd:])
        
        # R²
        ss_tot = total_variance * n_fluid * 3
        r2_scores[i] = 1 - ss_res / (ss_tot + 1e-10)
    
    return coverage, mse_scores, r2_scores


def compute_expected_losses(residual_field, router_output, layout, beta):
    """
    Compute expected losses for different strategies.

    Router output is logits in R: positive = CFD, negative = PINN.
    Default threshold is 0.
    """
    fluid_mask = layout > 0
    residuals = residual_field[fluid_mask]
    logits = router_output[fluid_mask]

    n_fluid = len(residuals)

    # PINN only: all residuals
    loss_pinn_only = np.mean(residuals)

    # CFD only: just beta (cost)
    loss_cfd_only = beta

    # Hybrid with default threshold 0
    cfd_mask = logits >= 0.0
    n_cfd = np.sum(cfd_mask)
    coverage_hybrid = n_cfd / n_fluid
    pinn_residuals = residuals[~cfd_mask]
    if len(pinn_residuals) > 0:
        loss_hybrid = beta * coverage_hybrid + np.mean(pinn_residuals) * (1 - coverage_hybrid)
    else:
        loss_hybrid = beta

    # Find optimal threshold by sweeping over actual logit range
    logit_min = float(np.min(logits))
    logit_max = float(np.max(logits))
    margin = max(0.1, (logit_max - logit_min) * 0.05)
    n_thresholds = 500
    thresholds = np.linspace(logit_min - margin, logit_max + margin, n_thresholds)
    losses = np.zeros(n_thresholds)

    for i, t in enumerate(thresholds):
        n_cfd = np.sum(logits >= t)
        cov = n_cfd / n_fluid
        if n_cfd == n_fluid:
            losses[i] = beta
        elif n_cfd == 0:
            losses[i] = np.mean(residuals)
        else:
            pinn_res = residuals[logits < t]
            losses[i] = beta * cov + np.mean(pinn_res) * (1 - cov)

    optimal_idx = np.argmin(losses)
    optimal_threshold = thresholds[optimal_idx]
    optimal_loss = losses[optimal_idx]
    optimal_coverage = np.sum(logits >= optimal_threshold) / n_fluid

    return {
        'loss_pinn_only': loss_pinn_only,
        'loss_cfd_only': loss_cfd_only,
        'loss_hybrid': loss_hybrid,
        'default_threshold': 0.0,
        'coverage_hybrid': coverage_hybrid,
        'optimal_threshold': optimal_threshold,
        'optimal_loss': optimal_loss,
        'optimal_coverage': optimal_coverage
    }


def compute_loss_vs_coverage(residual_field, router_output, layout, beta, n_points=200,
                             lambda_tv=0.01, lambda_entropy=0.1):
    """
    Compute the router training loss as a function of coverage.

    Uses logistic loss: 1/N * sum(beta * softplus(s) + R * softplus(-s)) + TV
    For the coverage sweep, uses the asymptotic binary limit.
    """
    fluid_mask = layout > 0
    residuals = residual_field[fluid_mask]
    logits = router_output[fluid_mask]

    n_fluid = len(residuals)

    # Sort by logit descending
    sorted_idx = np.argsort(logits)[::-1]
    sorted_residuals = residuals[sorted_idx]
    sorted_logits = logits[sorted_idx]

    coverage = np.linspace(0, 1, n_points)
    loss = np.zeros(n_points)

    for i, cov in enumerate(coverage):
        n_cfd = int(cov * n_fluid)
        n_pinn = n_fluid - n_cfd

        # Binary limit: beta*c + (1-c)*mean(R_pinn) + TV
        cfd_cost = beta * cov
        if n_pinn > 0:
            residual_loss = (1 - cov) * np.mean(sorted_residuals[n_cfd:])
        else:
            residual_loss = 0.0
        loss[i] = cfd_cost + residual_loss

    # Find optimal
    optimal_idx = np.argmin(loss)
    optimal_coverage = coverage[optimal_idx]
    optimal_loss = loss[optimal_idx]

    # Find corresponding threshold
    n_cfd_optimal = int(optimal_coverage * n_fluid)
    if n_cfd_optimal > 0 and n_cfd_optimal < n_fluid:
        optimal_threshold = sorted_logits[n_cfd_optimal - 1]
    elif n_cfd_optimal == 0:
        optimal_threshold = sorted_logits[0] + 0.001 if n_fluid > 0 else 1.0
    else:
        optimal_threshold = sorted_logits[-1] - 0.001 if n_fluid > 0 else 0.0

    # Compute actual logistic loss
    s = logits
    actual_logistic_loss = np.mean(
        beta * np.log1p(np.exp(np.clip(s, -50, 50))) +
        residuals * np.log1p(np.exp(np.clip(-s, -50, 50)))
    )

    s_2d = router_output * layout
    tv_h = np.sum(np.abs(s_2d[:, 1:] - s_2d[:, :-1]))
    tv_v = np.sum(np.abs(s_2d[1:, :] - s_2d[:-1, :]))
    actual_tv_loss = lambda_tv * (tv_h + tv_v) / n_fluid

    actual_total_loss = actual_logistic_loss + actual_tv_loss
    actual_coverage = np.mean(logits > 0)

    actual_loss_info = {
        'logistic_loss': actual_logistic_loss,
        'tv_loss': actual_tv_loss,
        'actual_total_loss': actual_total_loss,
        'actual_coverage': actual_coverage
    }

    optimal_info = {
        'optimal_threshold': optimal_threshold,
        'optimal_coverage': optimal_coverage,
        'optimal_loss': optimal_loss
    }

    return coverage, loss, actual_loss_info, optimal_info


def compute_residual_field(pinn_model, X, Y, layout, nu=0.01, rho=1.0,
                           bc_error=None):
    """
    Compute physics residual field for the PINN solution.
    Sum PDE residual + BC error, then median-normalize (matches training).
    """
    from train_router_cavity import CavityPINNResidualComputer

    residual_computer = CavityPINNResidualComputer(
        pinn_model=pinn_model,
        nu=nu,
        rho=rho,
        x_domain=(X.min(), X.max()),
        y_domain=(Y.min(), Y.max())
    )

    X_tf = tf.constant(X, dtype=tf.float32)
    Y_tf = tf.constant(Y, dtype=tf.float32)

    continuity, momentum = residual_computer.compute_residuals(X_tf, Y_tf)
    residual_field = (np.array(continuity) + np.array(momentum)) * layout

    # Add BC error before normalizing (sum then normalize, matches training)
    if bc_error is not None:
        residual_field = residual_field + bc_error

    # Median normalization (robust to heavy-tailed residuals)
    fluid_vals = residual_field[layout > 0]
    median = np.median(fluid_vals)
    residual_field = residual_field / (median + 1e-10) * layout

    return residual_field


def plot_combined_metrics(coverage, rmse_scores, results, beta,
                          residual_field, router_output, layout,
                          lambda_tv=0.01, lambda_entropy=0.1, save_path=None):
    """
    Plot combined coverage and loss metrics.
    """
    # Compute loss vs coverage
    loss_coverage, loss_values, actual_loss_info, full_loss_optimal = compute_loss_vs_coverage(
        residual_field, router_output, layout, beta,
        lambda_tv=lambda_tv
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Loss vs Coverage
    ax1.plot(loss_coverage * 100, loss_values, 'b-', linewidth=2.5, label='Loss Curve')

    # Mark endpoints
    ax1.plot(0, loss_values[0], 'o', color='purple', markersize=12, zorder=5)
    ax1.plot(100, loss_values[-1], 'o', color='teal', markersize=12, zorder=5)

    # Mark optimal point
    opt_cov = full_loss_optimal['optimal_coverage']
    opt_loss = full_loss_optimal['optimal_loss']
    ax1.plot(opt_cov * 100, opt_loss, '*', color='green', markersize=18, zorder=6,
             markeredgecolor='black', markeredgewidth=1)

    # Reference lines
    ax1.axhline(y=loss_values[0], color='purple', linestyle='--', linewidth=1.5, alpha=0.5, label=f'All PINN: {loss_values[0]:.4f}')
    ax1.axhline(y=loss_values[-1], color='teal', linestyle='--', linewidth=1.5, alpha=0.5, label=f'All CFD: {loss_values[-1]:.4f}')

    ax1.set_xlabel('Coverage (% solved by CFD)', fontsize=14)
    ax1.set_ylabel('Training Loss', fontsize=14)
    ax1.set_xlim(-5, 105)
    y_min = np.min(loss_values) - 0.1
    y_max = max(loss_values[0], loss_values[-1]) + 0.15
    ax1.set_ylim(y_min, y_max)
    ax1.set_title(f'Loss vs Coverage (β = {beta})', fontsize=16, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.3)

    # Annotations
    ax1.annotate(f'All PINN\n{loss_values[0]:.4f}', xy=(0, loss_values[0]),
                xytext=(8, loss_values[0] + 0.03),
                fontsize=9, color='purple', ha='left', va='bottom', fontweight='bold')
    ax1.annotate(f'All CFD\n{loss_values[-1]:.4f}', xy=(100, loss_values[-1]),
                xytext=(92, loss_values[-1] + 0.03),
                fontsize=9, color='teal', ha='right', va='bottom', fontweight='bold')
    ax1.annotate(f'Opt: {opt_cov*100:.0f}%\n{opt_loss:.4f}',
                xy=(opt_cov * 100, opt_loss),
                xytext=(opt_cov * 100 + 8, opt_loss - 0.08),
                fontsize=9, color='green', ha='left', va='top', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    ax1.legend(loc='upper right', fontsize=8)

    # Right plot: Optimal point loss breakdown
    fluid_mask = layout > 0
    residuals = residual_field[fluid_mask]
    sorted_idx = np.argsort(router_output[fluid_mask])[::-1]
    sorted_residuals = residuals[sorted_idx]
    n_fluid = len(residuals)
    n_cfd_opt = int(opt_cov * n_fluid)
    n_pinn_opt = n_fluid - n_cfd_opt

    opt_cfd_cost = beta * opt_cov
    opt_residual_cost = (1 - opt_cov) * np.mean(sorted_residuals[n_cfd_opt:]) if n_pinn_opt > 0 else 0.0
    components = ['CFD Cost\n(β·cov)', 'Residual\nCost', 'TOTAL']
    values = [opt_cfd_cost, opt_residual_cost, opt_loss]
    colors = ['steelblue', 'coral', 'green']

    bars = ax2.bar(components, values, color=colors, edgecolor='black', linewidth=1.5)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        y_pos = height + 0.01 if height >= 0 else height - 0.03
        va = 'bottom' if height >= 0 else 'top'
        ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{val:.4f}', ha='center', va=va, fontsize=10, fontweight='bold')

    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylabel('Loss Value', fontsize=14)
    ax2.set_title(f'Loss Breakdown at Optimal ({opt_cov*100:.0f}% CFD)', fontsize=16, fontweight='bold')

    info = f"Optimal Coverage: {opt_cov*100:.1f}%\n"
    info += f"---\n"
    info += f"β={beta}, λ_TV={lambda_tv}"
    ax2.text(0.98, 0.98, info, transform=ax2.transAxes, fontsize=9,
             va='top', ha='right', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=1200, bbox_inches='tight')
        print(f"  Saved combined metrics to {save_path}")
    
    plt.close(fig)
    
    return fig


def plot_coverage_curve(coverage, rmse_scores, results, beta, save_path=None):
    """
    Plot the coverage curve (RMSE vs coverage).
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(coverage * 100, rmse_scores, 'k-', linewidth=2.5, label='RMSE (PINN vs CFD)')

    ax.plot(0, rmse_scores[0], 'o', color='purple', markersize=12, zorder=5,
            label=f'All PINN: RMSE={rmse_scores[0]:.4f}')
    ax.plot(100, rmse_scores[-1], 'o', color='teal', markersize=12, zorder=5,
            label=f'All CFD: RMSE={rmse_scores[-1]:.4f}')

    ax.axhline(y=0.0, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='Perfect (RMSE=0)')

    ax.set_xlabel('Coverage (% solved by CFD)', fontsize=14)
    ax.set_ylabel('RMSE', fontsize=14)

    ax.set_xlim(-5, 105)
    ax.set_ylim(bottom=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_title('Coverage vs RMSE (PINN Prediction Quality) - Cavity Flow', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=1200, bbox_inches='tight')
        print(f"  Saved coverage plot to {save_path}")

    plt.close(fig)

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Plot coverage and expected loss metrics for hybrid PINN-CFD router (Cavity Flow)'
    )
    
    # Solution paths
    parser.add_argument('--pinn-path', type=str, default='./models/pinn_cavity_flow.h5',
                        help='Path to pre-trained PINN model weights')
    parser.add_argument('--cfd-path', type=str, default=None,
                        help='Path to CFD solution (.npz file)')
    parser.add_argument('--router-weights', type=str, default=None,
                        help='Path to trained router weights (.h5 file)')
    
    # Whether to compute CFD
    parser.add_argument('--compute-cfd', action='store_true',
                        help='Compute CFD solution')
    parser.add_argument('--save-cfd', type=str, default='./cfd_cavity_solution.npz',
                        help='Path to save computed CFD solution')
    
    # Router parameters
    parser.add_argument('--base-filters', type=int, default=32,
                        help='Base filters in router CNN')
    
    # Cost coefficient
    parser.add_argument('--beta', type=float, default=1,
                        help='Cost coefficient β')
    
    # Regularization weights
    parser.add_argument('--lambda-tv', type=float, default=0.01)
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Manual threshold for router decision (default: 0.0)')
    # Domain parameters (cavity is square)
    parser.add_argument('--N', type=int, default=100,
                        help='Grid size')
    parser.add_argument('--x-min', type=float, default=0.0)
    parser.add_argument('--x-max', type=float, default=1.0)
    parser.add_argument('--y-min', type=float, default=0.0)
    parser.add_argument('--y-max', type=float, default=1.0)
    parser.add_argument('--lid-velocity', type=float, default=1.0)
    
    # CFD solver parameters
    parser.add_argument('--Re', type=float, default=100)
    parser.add_argument('--max-iter', type=int, default=200000)
    parser.add_argument('--tol', type=float, default=1e-6)
    
    # Physical parameters
    parser.add_argument('--nu', type=float, default=0.01)
    parser.add_argument('--rho', type=float, default=1.0)
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./metrics_output_cavity',
                        help='Directory to save output plots')
    parser.add_argument('--morph-kernel', type=int, default=5,
                        help='Kernel size for morphological opening of mask')

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("COVERAGE AND EXPECTED LOSS METRICS (CAVITY FLOW)")
    print("=" * 60)
    
    # =========================================================================
    # Step 1: Setup domain
    # =========================================================================
    print("\n[Step 1] Setting up domain...")
    
    X, Y, layout, bc_mask, bc_u, bc_v, bc_p = create_cavity_setup(
        N=args.N,
        x_domain=(args.x_min, args.x_max),
        y_domain=(args.y_min, args.y_max),
        lid_velocity=args.lid_velocity
    )
    print(f"  Grid: {args.N} × {args.N}")
    print(f"  Fluid points: {np.sum(layout):.0f}")
    
    # =========================================================================
    # Step 2: Load PINN model and compute predictions
    # =========================================================================
    print("\n[Step 2] Loading PINN model...")
    
    network = CavityNetwork()
    pinn_model = network.build(
        num_inputs=2,
        layers=[32, 16, 16, 32],
        activation='swish',
        num_outputs=2
    )
    pinn_model.load_weights(args.pinn_path)
    print(f"  ✓ Loaded PINN from {args.pinn_path}")
    
    pinn_start_time = time.time()
    u_pinn, v_pinn, p_pinn = load_pinn_solution(pinn_model, X, Y, layout)
    pinn_inference_time = time.time() - pinn_start_time
    print(f"  ✓ PINN predictions computed in {pinn_inference_time:.2f}s")
    print(f"  PINN u range: [{u_pinn[layout > 0].min():.4f}, {u_pinn[layout > 0].max():.4f}]")
    print(f"  PINN v range: [{v_pinn[layout > 0].min():.4f}, {v_pinn[layout > 0].max():.4f}]")
    
    # =========================================================================
    # Step 3: Get or compute CFD solution
    # =========================================================================
    cfd_time = None
    
    if args.cfd_path and os.path.exists(args.cfd_path):
        print(f"\n[Step 3] Loading CFD solution from {args.cfd_path}...")
        data = np.load(args.cfd_path)
        u_cfd = data['u']
        v_cfd = data['v']
        p_cfd = data['p']
        print(f"  ✓ Loaded CFD solution")
    elif args.compute_cfd:
        print("\n[Step 3] Computing CFD solution...")
        u_cfd, v_cfd, p_cfd, X_cfd, Y_cfd, cfd_time = compute_cfd_solution(args)
        
        # Save CFD solution
        np.savez(args.save_cfd, u=u_cfd, v=v_cfd, p=p_cfd, X=X, Y=Y)
        print(f"  ✓ Saved CFD solution to {args.save_cfd}")
        
        # Plot CFD solution
        plot_cfd_solution(u_cfd, v_cfd, p_cfd, X, Y, layout,
                         save_path=os.path.join(args.output_dir, 'cfd_solution.pdf'))
    else:
        print("\n[Step 3] No CFD solution provided. Use --compute-cfd or --cfd-path")
        return
    
    print(f"  CFD u range: [{u_cfd[layout > 0].min():.4f}, {u_cfd[layout > 0].max():.4f}]")
    print(f"  CFD v range: [{v_cfd[layout > 0].min():.4f}, {v_cfd[layout > 0].max():.4f}]")
    
    # =========================================================================
    # Step 4: Load router
    # =========================================================================
    print("\n[Step 4] Loading router...")

    # Compute error transport field (used for router input and residual field)
    bc_error_local = compute_bc_error_field(
        bc_mask, bc_u, bc_v, u_pinn, v_pinn, layout
    )
    error_transport = solve_error_transport(
        u_pinn, v_pinn, bc_error_local, layout, nu=args.nu,
        x_domain=(args.x_min, args.x_max),
        y_domain=(args.y_min, args.y_max),
    )

    if args.router_weights is None:
        print("  No router weights provided. Using PINN predictions as proxy for confidence.")
        # Use error field as proxy
        error_field = compute_l2_error_field(u_pinn, v_pinn, p_pinn,
                                            u_cfd, v_cfd, p_cfd, layout,
                                            X=X, Y=Y)
        router_output = error_field / (np.max(error_field) + 1e-10)
    else:
        print(f"  Loading router from {args.router_weights}")
        
        # Create router inputs
        xy_flat = np.stack([X.flatten(), Y.flatten()], axis=-1).astype(np.float32)
        u_flat, v_flat = compute_uv_from_psi(pinn_model, xy_flat)
        psi_p = pinn_model.predict(xy_flat, batch_size=len(xy_flat), verbose=0)
        
        pinn_u = u_flat.reshape(X.shape).astype(np.float32) * layout
        pinn_v = v_flat.reshape(X.shape).astype(np.float32) * layout
        pinn_p = psi_p[:, 1].reshape(X.shape).astype(np.float32) * layout
        
        pde_residual_np = compute_pinn_residual_field(
            pinn_model, X, Y, layout, bc_mask, bc_u, bc_v, nu=args.nu
        )
        inputs = create_router_input(layout, bc_mask, bc_u, bc_v, bc_p,
                                     pinn_u, pinn_v, pinn_p, error_transport,
                                     pde_residual_np)

        router = RouterCNN(base_filters=args.base_filters)
        _ = router(inputs)
        router.load_weights(args.router_weights)
        
        router_output = router(inputs, training=False).numpy().squeeze()
        print(f"  ✓ Router loaded successfully")
    
    print(f"  Router output range: [{router_output[layout > 0].min():.4f}, {router_output[layout > 0].max():.4f}]")
    
    # =========================================================================
    # Step 5: Compute error field
    # =========================================================================
    print("\n[Step 5] Computing error field...")
    
    error_field = compute_l2_error_field(u_pinn, v_pinn, p_pinn,
                                        u_cfd, v_cfd, p_cfd, layout,
                                        X=X, Y=Y)
    print(f"  Error field range: [{error_field[layout > 0].min():.4f}, {error_field[layout > 0].max():.4f}]")
    
    # =========================================================================
    # Step 6: Compute coverage curve
    # =========================================================================
    print("\n[Step 6] Computing coverage curve...")
    
    coverage, mse_scores, r2_scores = compute_coverage_curve(
        (u_pinn, v_pinn, p_pinn),
        (u_cfd, v_cfd, p_cfd),
        router_output, layout, X=X, Y=Y
    )
    rmse_scores = np.sqrt(mse_scores)

    print(f"  RMSE at 0% coverage (all PINN): {rmse_scores[0]:.6f}")
    print(f"  RMSE at 100% coverage (all CFD): {rmse_scores[-1]:.6f}")
    
    # =========================================================================
    # Step 7: Compute physics residual and expected losses
    # =========================================================================
    print("\n[Step 7] Computing physics residuals and expected losses...")
    
    residual_field = compute_residual_field(pinn_model, X, Y, layout,
                                           nu=args.nu, rho=args.rho,
                                           bc_error=error_transport)
    
    results = compute_expected_losses(residual_field, router_output, layout, args.beta)
    
    print(f"\n  Expected Loss Comparison (β = {args.beta}):")
    print(f"  ----------------------------------------")
    print(f"  PINN Only:     {results['loss_pinn_only']:.6f}")
    print(f"  CFD Only:      {results['loss_cfd_only']:.6f}")
    print(f"  Hybrid:        {results['loss_hybrid']:.6f}")
    print(f"  ----------------------------------------")
    print(f"  OPTIMAL THRESHOLD: {results['optimal_threshold']:.4f}")
    print(f"  OPTIMAL LOSS:      {results['optimal_loss']:.6f}")
    print(f"  OPTIMAL COVERAGE:  {results['optimal_coverage']*100:.1f}%")
    
    # =========================================================================
    # Step 7b: Compute optimal threshold from full training loss
    # =========================================================================
    print("\n[Step 7b] Computing optimal threshold from full training loss...")
    
    _, _, _, full_loss_optimal = compute_loss_vs_coverage(
        residual_field, router_output, layout, args.beta,
        lambda_tv=args.lambda_tv
    )

    print(f"  Full Loss Optimal Threshold: {full_loss_optimal['optimal_threshold']:.6f}")
    print(f"  Full Loss Optimal Coverage:  {full_loss_optimal['optimal_coverage']*100:.2f}%")
    
    # =========================================================================
    # Step 7c: Compute hybrid solution
    # =========================================================================
    optimal_threshold = full_loss_optimal['optimal_threshold']
    print(f"\n[Step 7c] Computing hybrid solution with optimal threshold={optimal_threshold:.6f}...")
    
    u_hybrid, v_hybrid, p_hybrid, cfd_mask, hybrid_solve_time = compute_hybrid_solution(
        pinn_model, router_output, layout, optimal_threshold, args
    )
    
    actual_coverage = np.mean(cfd_mask[layout > 0])
    print(f"  Optimal threshold: {optimal_threshold:.6f}")
    print(f"  CFD coverage: {actual_coverage*100:.2f}%")
    print(f"  Hybrid solve time: {hybrid_solve_time:.2f} seconds")

    # Compute RMSE of hybrid solution vs CFD (velocity magnitude)
    hybrid_vel_mag = np.sqrt(u_hybrid**2 + v_hybrid**2)
    cfd_vel_mag = np.sqrt(u_cfd**2 + v_cfd**2)
    fluid_mask = layout > 0
    rmse_hybrid = np.sqrt(np.mean((hybrid_vel_mag[fluid_mask] - cfd_vel_mag[fluid_mask])**2))
    print(f"  Hybrid RMSE (vs CFD): {rmse_hybrid:.6f}")

    # Plot hybrid solution
    plot_hybrid_solution(
        u_hybrid, v_hybrid, p_hybrid, X, Y, layout, cfd_mask,
        threshold=optimal_threshold,
        coverage=actual_coverage,
        save_path=os.path.join(args.output_dir, 'hybrid_solution.pdf')
    )

    # Side-by-side comparison: PINN vs Hybrid vs CFD
    plot_solution_comparison(
        u_pinn, v_pinn, p_pinn,
        u_cfd, v_cfd, p_cfd,
        u_hybrid, v_hybrid, p_hybrid,
        X, Y, layout, cfd_mask,
        save_path=os.path.join(args.output_dir, 'solution_comparison.pdf')
    )

    # =========================================================================
    # Step 7d: Baseline hybrid using simple residual-threshold separator
    #   cfd_mask = (residual_field >= tau), tau chosen so CFD area matches
    #   the optimal hybrid coverage (star in coverage_metrics.pdf).
    # =========================================================================
    print("\n[Step 7d] Computing baseline hybrid with residual-threshold separator (tau = beta)...")
    fluid_mask_full = layout > 0
    residual_threshold = float(args.beta)
    print(f"  Residual threshold (R(x) >= tau => CFD): {residual_threshold:.6f}")

    u_hybrid_r, v_hybrid_r, p_hybrid_r, cfd_mask_r, hybrid_solve_time_r = compute_hybrid_solution(
        pinn_model, residual_field, layout, residual_threshold, args
    )
    actual_coverage_r = np.mean(cfd_mask_r[fluid_mask_full])
    hybrid_vel_mag_r = np.sqrt(u_hybrid_r**2 + v_hybrid_r**2)
    rmse_hybrid_r = np.sqrt(np.mean((hybrid_vel_mag_r[fluid_mask_full] - cfd_vel_mag[fluid_mask_full])**2))
    print(f"  Residual-threshold hybrid CFD coverage: {actual_coverage_r*100:.2f}%")
    print(f"  Residual-threshold hybrid RMSE (vs CFD): {rmse_hybrid_r:.6f}")
    print(f"  Residual-threshold hybrid solve time: {hybrid_solve_time_r:.2f}s")

    plot_hybrid_solution(
        u_hybrid_r, v_hybrid_r, p_hybrid_r, X, Y, layout, cfd_mask_r,
        threshold=residual_threshold,
        coverage=actual_coverage_r,
        save_path=os.path.join(args.output_dir, 'hybrid_solution_residual_threshold.pdf'),
        title=f'Naive Threshold Based Rule (Threshold = {residual_threshold:.2f})',
        show_info=False,
    )

    # =========================================================================
    # Step 8: Generate plots
    # =========================================================================
    print("\n[Step 8] Generating plots...")
    
    # Combined metrics plot
    plot_combined_metrics(
        coverage, rmse_scores, results, args.beta,
        residual_field=residual_field, router_output=router_output, layout=layout,
        lambda_tv=args.lambda_tv,
        save_path=os.path.join(args.output_dir, 'coverage_metrics.pdf')
    )

    # Coverage curve
    plot_coverage_curve(
        coverage, rmse_scores, results, args.beta,
        save_path=os.path.join(args.output_dir, 'coverage_curve.pdf')
    )
    
    # Save numerical results
    results_path = os.path.join(args.output_dir, 'metrics_results.npz')
    np.savez(results_path,
             coverage=coverage,
             rmse_scores=rmse_scores,
             router_output=router_output,
             error_field=error_field,
             residual_field=residual_field,
             cfd_time=cfd_time if cfd_time is not None else -1,
             u_hybrid=u_hybrid,
             v_hybrid=v_hybrid,
             p_hybrid=p_hybrid,
             cfd_mask=cfd_mask,
             full_loss_optimal_threshold=full_loss_optimal['optimal_threshold'],
             full_loss_optimal_coverage=full_loss_optimal['optimal_coverage'],
             u_hybrid_residual=u_hybrid_r,
             v_hybrid_residual=v_hybrid_r,
             p_hybrid_residual=p_hybrid_r,
             cfd_mask_residual=cfd_mask_r,
             residual_threshold=residual_threshold,
             residual_threshold_coverage=actual_coverage_r,
             residual_threshold_rmse=rmse_hybrid_r,
             **results)
    print(f"  ✓ Saved numerical results to {results_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("           RESULTS SUMMARY (Cavity Flow)")
    print("=" * 60)
    print(f"  β (CFD cost):          {args.beta}")
    print(f"  ----------------------------------------")
    print(f"  PINN Only Loss:        {results['loss_pinn_only']:.6f}")
    print(f"  CFD Only Loss:         {results['loss_cfd_only']:.6f} (= β)")
    print(f"  ----------------------------------------")
    print(f"  OPTIMAL THRESHOLD:     {full_loss_optimal['optimal_threshold']:.6f}")
    print(f"  OPTIMAL LOSS:          {full_loss_optimal['optimal_loss']:.6f}")
    print(f"  OPTIMAL COVERAGE:      {full_loss_optimal['optimal_coverage']*100:.2f}%")
    print(f"  ----------------------------------------")
    print(f"  PINN RMSE (all PINN):  {rmse_scores[0]:.6f}")
    print(f"  Hybrid RMSE (vs CFD): {rmse_hybrid:.6f}")
    print(f"  Hybrid CFD coverage:   {actual_coverage*100:.2f}%")
    print(f"  Hybrid solve time:     {hybrid_solve_time:.2f}s")
    print(f"  ----------------------------------------")
    print(f"  Residual-threshold baseline:")
    print(f"    tau (R(x) >= tau => CFD): {residual_threshold:.6f}")
    print(f"    CFD coverage:             {actual_coverage_r*100:.2f}%")
    print(f"    RMSE (vs CFD):            {rmse_hybrid_r:.6f}")
    print(f"    Solve time:               {hybrid_solve_time_r:.2f}s")
    print("=" * 60)
    print(f"\nResults saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
