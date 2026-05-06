"""
Plot Coverage and Expected Loss Metrics for Hybrid PINN-CFD Router.

This script generates two key plots:
1. Coverage Plot: Accuracy (L2 loss vs CFD ground truth) as a function of 
   rejection percentage (fraction sent to CFD solution)
   
2. Expected Loss Comparison: Compare expected true loss for:
   - ONLY PINN
   - ONLY iterative solution (CFD) with cost β per node
   - Hybrid system using the router

Usage:
    python plot_coverage_metrics.py --pinn-path <path> --cfd-path <path> --router-path <path>
    python plot_coverage_metrics.py --compute-cfd  # Compute CFD solution if not available

The graph shows the optimal hybrid system operating point.
"""

import argparse
import os
import time
import numpy as np
import cv2
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
    PINNResidualComputer,
    create_router_input,
    compute_bc_error_field,
    solve_error_transport,
    create_cylinder_setup,
)
from lib.cylinder_flow import CylinderFlowSimulation, CylinderFlowHybridSimulation
from cylinder_network import Network as CylinderNetwork


def compute_uv_direct(network, xy):
    """Extract (u, v) directly from network output (u, v, p)."""
    uvp = network.predict(xy, batch_size=len(xy), verbose=0)
    u = uvp[..., 0]
    v = uvp[..., 1]
    return u, v


def load_pinn_solution(pinn_model, X, Y, layout):
    """
    Compute PINN solution on the grid.
    
    Parameters:
    -----------
    pinn_model : tf.keras.Model
        Pre-trained PINN model
    X, Y : ndarray
        Coordinate grids (Ny, Nx)
    layout : ndarray
        Fluid domain mask (1=fluid, 0=obstacle)
        
    Returns:
    --------
    u_pinn, v_pinn, p_pinn : ndarray
        PINN predictions on the grid
    """
    xy_flat = np.stack([X.flatten(), Y.flatten()], axis=-1).astype(np.float32)
    pinn_uvp = pinn_model.predict(xy_flat, batch_size=len(xy_flat), verbose=0)
    
    u_pinn = pinn_uvp[:, 0].reshape(X.shape).astype(np.float32)
    v_pinn = pinn_uvp[:, 1].reshape(X.shape).astype(np.float32)
    p_pinn = pinn_uvp[:, 2].reshape(X.shape).astype(np.float32)

    # Mask out obstacle regions
    u_pinn = u_pinn * layout
    v_pinn = v_pinn * layout
    p_pinn = p_pinn * layout
    
    return u_pinn, v_pinn, p_pinn


def compute_cfd_solution(args):
    """
    Compute CFD solution using iterative solver.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments with domain parameters
        
    Returns:
    --------
    u_cfd, v_cfd, p_cfd : ndarray
        CFD solution on the grid
    X, Y : ndarray
        Coordinate grids
    cfd_time : float
        Time taken to compute CFD solution in seconds
    """
    print("\n[Computing CFD Solution]")
    print("  This may take a while...")
    
    sim = CylinderFlowSimulation(
        Re=args.Re,
        N=args.ny,
        max_iter=args.max_iter,
        tol=args.tol,
        x_domain=(args.x_min, args.x_max),
        y_domain=(args.y_min, args.y_max),
        cylinder_center=(args.cylinder_x, args.cylinder_y),
        cylinder_radius=args.cylinder_radius,
        inlet_velocity=args.inlet_velocity
    )
    
    start_time = time.time()
    u_cfd, v_cfd, p_cfd = sim.solve()
    cfd_time = time.time() - start_time
    
    print(f"  ✓ CFD solution computed in {cfd_time:.2f} seconds")
    
    return u_cfd, v_cfd, p_cfd, sim.X, sim.Y, cfd_time


def compute_l2_error_field(u_pred, v_pred, p_pred, u_true, v_true, p_true,
                            layout, X=None, Y=None, interface_mask=None):
    """Per-point gauge-invariant L2 error.

    Uses |∇p - ∇p_true|² (gauge-free) instead of (p - p_true)², normalized by
    the CFD pressure-gradient scale. Excludes a 1-cell ring around the fluid
    boundary and, if `interface_mask` is provided (e.g. hybrid PINN/CFD
    split), a 1-cell ring around that interface too.
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

    # Mask out boundary-gradient artifacts.
    valid = binary_erosion(fluid, iterations=1)
    if interface_mask is not None:
        m = interface_mask.astype(bool) & fluid
        ring = binary_dilation(m, iterations=1) & ~binary_erosion(m, iterations=1)
        valid = valid & ~ring
    error_field = np.where(valid, error_field, 0.0)
    return error_field


def plot_cfd_solution(u_cfd, v_cfd, p_cfd, X, Y, layout, cylinder_center, cylinder_radius, save_path=None):
    """
    Plot the CFD solution fields.
    
    Parameters:
    -----------
    u_cfd, v_cfd, p_cfd : ndarray
        CFD solution components
    X, Y : ndarray
        Coordinate grids
    layout : ndarray
        Fluid domain mask
    cylinder_center : tuple
        (cx, cy) cylinder center
    cylinder_radius : float
        Cylinder radius
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    cx, cy = cylinder_center
    
    # Velocity magnitude
    vel_mag = np.sqrt(u_cfd**2 + v_cfd**2)
    
    # Mask obstacle regions for plotting
    u_plot = np.ma.masked_where(layout == 0, u_cfd)
    v_plot = np.ma.masked_where(layout == 0, v_cfd)
    p_plot = np.ma.masked_where(layout == 0, p_cfd)
    vel_plot = np.ma.masked_where(layout == 0, vel_mag)
    
    # Plot velocity magnitude
    ax = axes[0, 0]
    cf = ax.contourf(X, Y, vel_plot, levels=50, cmap='rainbow')
    plt.colorbar(cf, ax=ax)
    circle = plt.Circle((cx, cy), cylinder_radius, color='gray', fill=True)
    ax.add_patch(circle)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('CFD: Velocity Magnitude |u|')
    ax.set_aspect('equal')
    
    # Plot u-velocity
    ax = axes[0, 1]
    cf = ax.contourf(X, Y, u_plot, levels=50, cmap='rainbow')
    plt.colorbar(cf, ax=ax)
    circle = plt.Circle((cx, cy), cylinder_radius, color='gray', fill=True)
    ax.add_patch(circle)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('CFD: u-velocity')
    ax.set_aspect('equal')
    
    # Plot v-velocity
    ax = axes[1, 0]
    cf = ax.contourf(X, Y, v_plot, levels=50, cmap='rainbow')
    plt.colorbar(cf, ax=ax)
    circle = plt.Circle((cx, cy), cylinder_radius, color='gray', fill=True)
    ax.add_patch(circle)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('CFD: v-velocity')
    ax.set_aspect('equal')
    
    # Plot pressure
    ax = axes[1, 1]
    cf = ax.contourf(X, Y, p_plot, levels=50, cmap='rainbow')
    plt.colorbar(cf, ax=ax)
    circle = plt.Circle((cx, cy), cylinder_radius, color='gray', fill=True)
    ax.add_patch(circle)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('CFD: Pressure')
    ax.set_aspect('equal')
    
    plt.suptitle('CFD Ground Truth Solution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=1200, bbox_inches='tight')
        print(f"  Saved CFD solution plot to {save_path}")
    
    plt.close(fig)
    
    return fig


def create_hybrid_solution(u_pinn, v_pinn, p_pinn, u_cfd, v_cfd, p_cfd, 
                           router_output, layout, threshold):
    """
    Create hybrid solution by blending PINN and CFD based on router output and threshold.
    
    NOTE: This is a simple blending approach. For actual hybrid simulation that
    re-computes CFD with PINN boundary conditions, use compute_hybrid_solution().
    
    Points where router_output > threshold use CFD, otherwise use PINN.
    
    Parameters:
    -----------
    u_pinn, v_pinn, p_pinn : ndarray
        PINN solution components
    u_cfd, v_cfd, p_cfd : ndarray
        CFD solution components
    router_output : ndarray
        Router confidence (higher = more likely to use CFD)
    layout : ndarray
        Fluid domain mask
    threshold : float
        Threshold for router decision (router > threshold => CFD)
        
    Returns:
    --------
    u_hybrid, v_hybrid, p_hybrid : ndarray
        Hybrid solution components
    cfd_mask : ndarray
        Boolean mask indicating which points use CFD
    """
    # Determine which points use CFD vs PINN
    cfd_mask = router_output > threshold
    
    # Create hybrid solution
    u_hybrid = np.where(cfd_mask, u_cfd, u_pinn) * layout
    v_hybrid = np.where(cfd_mask, v_cfd, v_pinn) * layout
    p_hybrid = np.where(cfd_mask, p_cfd, p_pinn) * layout
    
    return u_hybrid, v_hybrid, p_hybrid, cfd_mask


def compute_hybrid_solution(pinn_model, router_output, layout, threshold, args):
    """
    Compute actual hybrid solution by running CFD in CFD regions with PINN boundary conditions.
    
    This runs the full hybrid simulation: CFD solver in regions where router > threshold,
    using PINN values as boundary conditions at the interface.
    
    Parameters:
    -----------
    pinn_model : tf.keras.Model
        Pre-trained PINN model
    router_output : ndarray
        Router confidence output
    layout : ndarray
        Fluid domain mask
    threshold : float
        Threshold for CFD vs PINN decision
    args : argparse.Namespace
        Command line arguments with domain parameters
        
    Returns:
    --------
    u_hybrid, v_hybrid, p_hybrid : ndarray
        Hybrid solution components
    cfd_mask : ndarray
        Binary mask indicating CFD regions
    solve_time : float
        Time taken to solve the hybrid system
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
    sim = CylinderFlowHybridSimulation(
        network=pinn_model,
        uv_func=compute_uv_direct,
        mask=cfd_mask,
        Re=args.Re,
        N=args.ny,
        max_iter=args.max_iter,
        tol=args.tol,
        x_domain=(args.x_min, args.x_max),
        y_domain=(args.y_min, args.y_max),
        cylinder_center=(args.cylinder_x, args.cylinder_y),
        cylinder_radius=args.cylinder_radius,
        inlet_velocity=args.inlet_velocity
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
                         cylinder_center, cylinder_radius, threshold, coverage,
                         save_path=None, title=None, show_info=True):
    """
    Plot the hybrid solution fields with CFD/PINN region overlay.
    
    Parameters:
    -----------
    u_hybrid, v_hybrid, p_hybrid : ndarray
        Hybrid solution components
    X, Y : ndarray
        Coordinate grids
    layout : ndarray
        Fluid domain mask
    cfd_mask : ndarray
        Boolean mask indicating CFD regions
    cylinder_center : tuple
        (cx, cy) cylinder center
    cylinder_radius : float
        Cylinder radius
    threshold : float
        Threshold used for hybrid decision
    coverage : float
        Fraction of points using CFD
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    cx, cy = cylinder_center
    
    # Velocity magnitude
    vel_mag = np.sqrt(u_hybrid**2 + v_hybrid**2)
    
    # Mask obstacle regions for plotting
    u_plot = np.ma.masked_where(layout == 0, u_hybrid)
    v_plot = np.ma.masked_where(layout == 0, v_hybrid)
    p_plot = np.ma.masked_where(layout == 0, p_hybrid)
    vel_plot = np.ma.masked_where(layout == 0, vel_mag)
    
    # Plot velocity magnitude
    ax = axes[0, 0]
    cf = ax.contourf(X, Y, vel_plot, levels=50, cmap='rainbow')
    plt.colorbar(cf, ax=ax)
    circle = plt.Circle((cx, cy), cylinder_radius, color='gray', fill=True)
    ax.add_patch(circle)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Hybrid: Velocity Magnitude |u|')
    ax.set_aspect('equal')
    
    # Plot u-velocity
    ax = axes[0, 1]
    cf = ax.contourf(X, Y, u_plot, levels=50, cmap='rainbow')
    plt.colorbar(cf, ax=ax)
    circle = plt.Circle((cx, cy), cylinder_radius, color='gray', fill=True)
    ax.add_patch(circle)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Hybrid: u-velocity')
    ax.set_aspect('equal')
    
    # Plot v-velocity
    ax = axes[0, 2]
    cf = ax.contourf(X, Y, v_plot, levels=50, cmap='rainbow')
    plt.colorbar(cf, ax=ax)
    circle = plt.Circle((cx, cy), cylinder_radius, color='gray', fill=True)
    ax.add_patch(circle)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Hybrid: v-velocity')
    ax.set_aspect('equal')
    
    # Plot pressure
    ax = axes[1, 0]
    cf = ax.contourf(X, Y, p_plot, levels=50, cmap='rainbow')
    plt.colorbar(cf, ax=ax)
    circle = plt.Circle((cx, cy), cylinder_radius, color='gray', fill=True)
    ax.add_patch(circle)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Hybrid: Pressure')
    ax.set_aspect('equal')
    
    # Plot CFD/PINN region map
    ax = axes[1, 1]
    # Create a visualization: CFD=1, PINN=0, obstacle=0.5
    region_map = np.where(layout == 0, 0.5, np.where(cfd_mask, 1.0, 0.0))
    im = ax.imshow(region_map, extent=[X.min(), X.max(), Y.min(), Y.max()],
                   origin='lower', cmap='RdYlBu', vmin=0, vmax=1, aspect='auto')
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 0.5, 1])
    cbar.ax.set_yticklabels(['PINN', 'Obstacle', 'CFD'])
    circle = plt.Circle((cx, cy), cylinder_radius, color='black', fill=False, linewidth=2)
    ax.add_patch(circle)
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
    • Gray circle: Cylinder obstacle

    The hybrid solution uses CFD for regions
    where router_output > threshold, and
    PINN elsewhere.
    """
        ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    suptitle = title if title is not None else 'Hybrid PINN-CFD Solution (Optimal Threshold)'
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
                             cylinder_center, cylinder_radius,
                             save_path=None, title=None):
    """5x3 panel: rows = p, u, v, |velocity|, error; cols = PINN, Hybrid, CFD.

    Each row shares a colour scale across the three columns. Pressure is
    gauge-aligned via fluid-median subtraction. The hybrid column is annotated
    with a closed contour outlining the PINN sub-domain (no shading), padded
    so the contour closes against the domain boundary.
    """
    from matplotlib.colors import Normalize

    cx, cy = cylinder_center
    fluid = layout > 0

    # Align pressures by fluid median (gauge).
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
            absmax = float(np.max(np.abs(stacked))) + 1e-30
            vmin, vmax = -absmax, absmax

        # Each column has its own axes + colorbar, but the colour scale
        # (vmin, vmax) is shared across the row so PINN | Hybrid | CFD use
        # identical mapping.  Matches the Poisson `solution_exp3_*` style.
        for j, f in enumerate(fields):
            ax = axes[i, j]
            data = np.where(fluid, f, np.nan)
            im = ax.pcolormesh(X, Y, data, cmap=cmap,
                               vmin=vmin, vmax=vmax, shading='auto')
            plt.colorbar(im, ax=ax, fraction=0.046, label=label)
            circle = plt.Circle((cx, cy), cylinder_radius,
                                color='gray', fill=True, zorder=5)
            ax.add_patch(circle)
            if j == 1:
                # Closed contour around the PINN sub-domain, padded so the
                # contour closes against the domain boundary where PINN
                # extends to the edge.
                pinn_mask = ((cfd_mask == 0) & fluid).astype(float)
                pad_mask = np.pad(pinn_mask, 1, mode='constant',
                                  constant_values=0.0)
                dx_pad = X[0, 1] - X[0, 0] if X.shape[1] > 1 else 1.0
                dy_pad = Y[1, 0] - Y[0, 0] if Y.shape[0] > 1 else 1.0
                xs_pad = np.concatenate([
                    [X[0, 0] - dx_pad], X[0, :], [X[0, -1] + dx_pad]
                ])
                ys_pad = np.concatenate([
                    [Y[0, 0] - dy_pad], Y[:, 0], [Y[-1, 0] + dy_pad]
                ])
                Xp, Yp = np.meshgrid(xs_pad, ys_pad)
                ax.contour(Xp, Yp, pad_mask, levels=[0.5],
                           colors='black', linewidths=1.2)
            ax.set_aspect('equal')
            if i == 0:
                ax.set_title(col_titles[j])
            if j == 0:
                ax.set_ylabel(label)
            if i == len(rows) - 1:
                ax.set_xlabel('x')

    if title is not None:
        fig.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985] if title else None)
    if save_path:
        fig.savefig(save_path, dpi=1200, bbox_inches='tight')
        print(f"  Saved solution comparison to {save_path}")
    plt.close(fig)


def compute_coverage_curve(pinn_pred, cfd_truth, router_output, layout, n_points=100):
    """
    Compute MSE and R² of PINN points as a function of CFD coverage.
    
    MSE is guaranteed to be monotonically decreasing (if router is good).
    R² uses global normalization for comparability.
    
    Logic:
    - X-axis: coverage = fraction of points solved by CFD (0 to 1)
    - Sort points by router confidence (DESCENDING: highest first)
    - At coverage X%: the top X% points (highest confidence) go to CFD
    - Y-axis: Metric of the remaining (1-X)% PINN points
    
    Parameters:
    -----------
    pinn_pred : ndarray
        PINN predictions
    cfd_truth : ndarray  
        CFD ground truth
    router_output : ndarray
        Router confidence (higher = more likely to use CFD)
    layout : ndarray
        Fluid domain mask (1=fluid, 0=obstacle)
    n_points : int
        Number of points on the curve
        
    Returns:
    --------
    coverage : ndarray
        Fraction solved by CFD (0 to 1)
    mse_scores : ndarray
        Mean Squared Error of remaining PINN points (should decrease)
    r2_scores : ndarray
        R² using global normalization
    """
    # Get fluid points only
    fluid_mask = layout > 0
    pinn_vals = pinn_pred[fluid_mask]
    cfd_vals = cfd_truth[fluid_mask]
    confidences = router_output[fluid_mask]
    
    n_fluid = len(pinn_vals)
    
    # Compute GLOBAL statistics (used as fixed reference for R²)
    global_mean = np.mean(cfd_vals)
    global_var = np.var(cfd_vals)
    
    # Sort by router confidence DESCENDING (highest confidence first → go to CFD first)
    sorted_idx = np.argsort(confidences)[::-1]  # Descending order
    sorted_pinn = pinn_vals[sorted_idx]
    sorted_cfd = cfd_vals[sorted_idx]
    
    # Pre-compute squared errors for efficiency
    sorted_sq_errors = (sorted_cfd - sorted_pinn) ** 2
    
    coverage = np.linspace(0, 1, n_points)
    mse_scores = np.zeros(n_points)
    r2_scores = np.zeros(n_points)
    
    for i, cov in enumerate(coverage):
        # Number of points sent to CFD (the top cov% with highest confidence)
        n_cfd = int(cov * n_fluid)
        # Number of points kept as PINN (the remaining ones with lower confidence)
        n_pinn = n_fluid - n_cfd
        
        if n_pinn > 0:
            # MSE of remaining PINN points
            mse_scores[i] = np.mean(sorted_sq_errors[n_cfd:])
            
            # R² = 1 - MSE / global_variance (normalized by global variance)
            if global_var > 1e-10:
                r2_scores[i] = 1 - mse_scores[i] / global_var
            else:
                r2_scores[i] = 1.0
        else:
            # All points sent to CFD - perfect score
            mse_scores[i] = 0.0
            r2_scores[i] = 1.0
    
    return coverage, mse_scores, r2_scores


def plot_coverage_progression(u_pinn, v_pinn, u_cfd, v_cfd,
                              router_output, layout, X, Y,
                              cylinder_center, cylinder_radius,
                              coverages=(0.1, 0.2, 0.3, 0.4, 0.5,
                                         0.6, 0.7, 0.8, 0.9, 1.0),
                              save_path=None):
    """
    Plot idealized hybrid velocity magnitude at increasing CFD coverage levels.

    At each target coverage c, the top-c fraction of fluid points (ranked by
    router_output) is taken from the CFD field, the rest from the PINN field.
    Uses per-pixel blending (no extra hybrid solve) so this is cheap.
    """
    fluid_mask = layout > 0
    confidences = router_output[fluid_mask]
    n_fluid = confidences.size

    cfd_vel = np.sqrt(u_cfd**2 + v_cfd**2)
    pinn_vel = np.sqrt(u_pinn**2 + v_pinn**2)
    vmin = float(min(np.nanmin(cfd_vel[fluid_mask]), np.nanmin(pinn_vel[fluid_mask])))
    vmax = float(max(np.nanmax(cfd_vel[fluid_mask]), np.nanmax(pinn_vel[fluid_mask])))

    n = len(coverages)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows),
                             squeeze=False)

    for idx, cov in enumerate(coverages):
        ax = axes[idx // ncols][idx % ncols]
        if cov <= 0:
            thresh = np.inf
            cfd_mask = np.zeros_like(layout, dtype=bool)
        elif cov >= 1:
            thresh = -np.inf
            cfd_mask = fluid_mask.copy()
        else:
            # threshold at the (1-cov) quantile -> top cov fraction routed to CFD
            thresh = float(np.quantile(confidences, 1.0 - cov))
            cfd_mask = (router_output >= thresh) & fluid_mask

        u_blend = np.where(cfd_mask, u_cfd, u_pinn) * layout
        v_blend = np.where(cfd_mask, v_cfd, v_pinn) * layout
        vel = np.sqrt(u_blend**2 + v_blend**2)
        vel_plot = np.ma.masked_where(~fluid_mask, vel)

        levels = np.linspace(vmin, vmax, 50)
        im = ax.contourf(X, Y, vel_plot, levels=levels, cmap='RdBu_r',
                         vmin=vmin, vmax=vmax, extend='both')
        ax.contour(X, Y, cfd_mask.astype(float), levels=[0.5],
                   colors='black', linewidths=0.6)
        cx, cy = cylinder_center
        circ = plt.Circle((cx, cy), cylinder_radius, color='gray', fill=True)
        ax.add_patch(circ)

        actual_cov = float(np.mean(cfd_mask[fluid_mask]))
        ax.set_title(f'coverage={actual_cov*100:.0f}%  τ={thresh:.3g}',
                     fontsize=10)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])

    # Hide unused axes
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')

    fig.suptitle('Router Coverage Progression (idealized blend: top-c% to CFD)',
                 fontsize=12)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, shrink=0.9)
    cbar.set_label('|U|')

    if save_path:
        plt.savefig(save_path, dpi=1200, bbox_inches='tight')
        print(f"  Saved coverage progression to {save_path}")
    plt.close(fig)


def compute_expected_losses(residual_field, router_output, layout, beta):
    """
    Compute expected losses for PINN-only, CFD-only, and hybrid systems.

    Uses the binary-limit loss:
    L = β · coverage + (1 - coverage) · E[residual | PINN]

    Router output is logits in R: positive = CFD, negative = PINN.
    Default threshold is 0.

    Parameters:
    -----------
    residual_field : ndarray
        Per-point normalized physics residual
    router_output : ndarray
        Router logits (negative=PINN, positive=CFD)
    layout : ndarray
        Fluid domain mask
    beta : float
        Cost coefficient for CFD

    Returns:
    --------
    dict with loss components and optimal operating point
    """
    fluid_mask = layout > 0
    residuals = residual_field[fluid_mask]
    logits = router_output[fluid_mask]

    n_fluid = len(residuals)

    # PINN only: coverage = 0, all residuals contribute
    loss_pinn_only = np.mean(residuals)

    # CFD only: coverage = 1, cost is β per node
    loss_cfd_only = beta

    # Hybrid: use router threshold=0 to determine coverage
    default_threshold = 0.0
    cfd_mask = logits > default_threshold
    coverage_hybrid = np.mean(cfd_mask)

    pinn_points = ~cfd_mask
    if np.sum(pinn_points) > 0:
        pinn_residual = np.mean(residuals[pinn_points])
    else:
        pinn_residual = 0.0

    loss_hybrid = beta * coverage_hybrid + (1 - coverage_hybrid) * pinn_residual

    # Find optimal operating point by sweeping thresholds over actual logit range
    logit_min = float(np.min(logits))
    logit_max = float(np.max(logits))
    margin = max(0.1, (logit_max - logit_min) * 0.05)
    thresholds = np.linspace(logit_min - margin, logit_max + margin, 500)

    best_loss = float('inf')
    best_coverage = 0.0
    best_threshold = 0.0

    coverages_list = []
    losses = []

    for thresh in thresholds:
        cfd_points = logits > thresh
        cov = np.mean(cfd_points)

        pinn_pts = ~cfd_points
        if np.sum(pinn_pts) > 0:
            pinn_res = np.mean(residuals[pinn_pts])
        else:
            pinn_res = 0.0

        loss = beta * cov + (1 - cov) * pinn_res
        coverages_list.append(cov)
        losses.append(loss)

        if loss < best_loss:
            best_loss = loss
            best_coverage = cov
            best_threshold = thresh

    return {
        'loss_pinn_only': loss_pinn_only,
        'loss_cfd_only': loss_cfd_only,
        'loss_hybrid': loss_hybrid,
        'coverage_hybrid': coverage_hybrid,
        'default_threshold': default_threshold,
        'optimal_coverage': best_coverage,
        'optimal_loss': best_loss,
        'optimal_threshold': best_threshold,
        'all_thresholds': thresholds,
        'all_coverages': np.array(coverages_list),
        'all_losses': np.array(losses)
    }


def plot_coverage_curve(coverage, rmse_scores, results, beta, save_path=None):
    """
    Plot RMSE vs coverage curve.

    Parameters:
    -----------
    coverage : ndarray
        Fraction sent to CFD (x-axis)
    rmse_scores : ndarray
        RMSE at each coverage level (y-axis)
    results : dict
        Results from compute_expected_losses
    beta : float
        Cost coefficient for CFD
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Main coverage curve (RMSE vs Coverage)
    ax.plot(coverage * 100, rmse_scores, 'k-', linewidth=2.5, label='RMSE (PINN vs CFD)')

    # Mark key points
    ax.plot(0, rmse_scores[0], 'o', color='purple', markersize=12, zorder=5, label=f'All PINN: RMSE={rmse_scores[0]:.4f}')
    ax.plot(100, rmse_scores[-1], 'o', color='teal', markersize=12, zorder=5, label=f'All CFD: RMSE={rmse_scores[-1]:.4f}')

    # Horizontal reference line at RMSE=0 (perfect)
    ax.axhline(y=0.0, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='Perfect (RMSE=0)')

    # Labels
    ax.set_xlabel('Coverage (% solved by CFD)', fontsize=14)
    ax.set_ylabel('RMSE', fontsize=14)

    # Set axis limits
    ax.set_xlim(-5, 105)
    ax.set_ylim(bottom=0)

    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_title('Coverage vs RMSE (PINN Prediction Quality)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=1200, bbox_inches='tight')
        print(f"  Saved coverage plot to {save_path}")

    plt.close(fig)

    return fig


def plot_expected_loss_comparison(results, beta, save_path=None):
    """
    Plot expected loss comparison for PINN-only, CFD-only, and hybrid.
    
    Parameters:
    -----------
    results : dict
        Results from compute_expected_losses
    beta : float
        Cost coefficient for CFD
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    methods = ['PINN Only', f'CFD Only\n(cost β={beta})', 'Hybrid\n(Router)']
    losses = [
        results['loss_pinn_only'],
        results['loss_cfd_only'],
        results['loss_hybrid']
    ]
    colors = ['purple', 'gray', 'teal']
    
    bars = ax.bar(methods, losses, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, loss in zip(bars, losses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add optimal line
    ax.axhline(y=results['optimal_loss'], color='green', linestyle='--', 
               linewidth=2, label=f"Optimal: {results['optimal_loss']:.4f}")
    
    ax.set_ylabel('Expected True Loss', fontsize=12)
    ax.set_title(f'Expected Loss Comparison (β = {beta})', fontsize=14)
    ax.legend(loc='upper right')
    
    # Add coverage info
    info_text = f"Hybrid Coverage: {results['coverage_hybrid']*100:.1f}%\n"
    info_text += f"Optimal Coverage: {results['optimal_coverage']*100:.1f}%"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=1200, bbox_inches='tight')
        print(f"  Saved loss comparison to {save_path}")
    
    plt.close(fig)  # Close to free memory
    
    return fig


def compute_loss_vs_coverage(residual_field, router_output, layout, beta, n_points=200,
                             lambda_tv=0.01, lambda_entropy=0.1):
    """
    Compute the router training loss as a function of coverage.

    Uses the logistic loss formulation:
    L = 1/N * sum(beta * softplus(s) + R * softplus(-s)) + lambda_tv * TV(s)

    For the coverage sweep, we use the asymptotic binary limit:
    L(c) = beta*c + (1-c)*mean(R_pinn) + TV_approx

    Parameters:
    -----------
    residual_field : ndarray
        Per-point normalized physics residual (continuity + momentum)
    router_output : ndarray
        Router logits (negative=PINN, positive=CFD)
    layout : ndarray
        Fluid domain mask (1=fluid, 0=obstacle)
    beta : float
        Cost coefficient for CFD
    n_points : int
        Number of coverage points to compute
    lambda_tv : float
        Total variation regularization weight
    lambda_entropy : float
        Unused, kept for API compatibility.

    Returns:
    --------
    coverage : ndarray
        Fraction sent to CFD (0 to 1)
    loss : ndarray
        Training loss at each coverage level
    actual_loss_info : dict
        Breakdown of actual router loss components
    optimal_info : dict
        Information about the optimal operating point (minimum of loss curve)
    """
    fluid_mask = layout > 0
    residuals = residual_field[fluid_mask]
    logits = router_output[fluid_mask]

    n_fluid = len(residuals)

    # Sort by router logit DESCENDING (highest logit first → go to CFD first)
    sorted_idx = np.argsort(logits)[::-1]
    sorted_residuals = residuals[sorted_idx]
    sorted_logits = logits[sorted_idx]

    # For TV computation, we need the 2D structure
    s_2d = router_output * layout

    coverage = np.linspace(0, 1, n_points)
    loss = np.zeros(n_points)

    for i, cov in enumerate(coverage):
        # Number of points sent to CFD (the top cov% with highest logit)
        n_cfd = int(cov * n_fluid)
        n_pinn = n_fluid - n_cfd

        # Binary limit of logistic loss:
        # CFD cost: β * coverage
        cfd_cost = beta * cov

        # Residual cost: (1-cov) * mean(R on PINN points)
        if n_pinn > 0:
            residual_loss = (1 - cov) * np.mean(sorted_residuals[n_cfd:])
        else:
            residual_loss = 0.0

        loss[i] = cfd_cost + residual_loss

    # ===== Compute ACTUAL router logistic loss =====
    s = logits
    R = residuals

    # Logistic loss: 1/N * sum(beta * softplus(s) + R * softplus(-s))
    actual_logistic_loss = np.mean(
        beta * np.log1p(np.exp(np.clip(s, -50, 50))) +
        R * np.log1p(np.exp(np.clip(-s, -50, 50)))
    )

    # Total variation (spatial)
    tv_h = np.mean(np.abs(s_2d[:, 1:] - s_2d[:, :-1]))
    tv_v = np.mean(np.abs(s_2d[1:, :] - s_2d[:-1, :]))
    actual_tv_loss = lambda_tv * (tv_h + tv_v)

    # Total actual loss
    actual_router_loss = actual_logistic_loss + actual_tv_loss

    # ===== Find optimal point on the loss curve =====
    min_idx = np.argmin(loss)
    opt_coverage = coverage[min_idx]
    opt_loss = loss[min_idx]

    # Reverse-engineer the threshold that achieves this coverage
    n_cfd_opt = int(opt_coverage * n_fluid)
    if n_cfd_opt == 0:
        opt_threshold = sorted_logits[0] + 0.001 if n_fluid > 0 else 1.0
    elif n_cfd_opt >= n_fluid:
        opt_threshold = sorted_logits[-1] - 0.001 if n_fluid > 0 else 0.0
    else:
        opt_threshold = (sorted_logits[n_cfd_opt - 1] + sorted_logits[n_cfd_opt]) / 2

    return coverage, loss, {
        'actual_total_loss': actual_router_loss,
        'logistic_loss': actual_logistic_loss,
        'tv_loss': actual_tv_loss,
    }, {
        'optimal_coverage': opt_coverage,
        'optimal_loss': opt_loss,
        'optimal_threshold': opt_threshold,
    }


def plot_loss_vs_coverage(residual_field, router_output, layout, beta,
                          lambda_tv=0.01, save_path=None):
    """Standalone plot of training-loss vs coverage (left panel of
    plot_combined_metrics, saved to its own file).
    """
    fig, ax1 = plt.subplots(figsize=(7, 6))
    cov_for_loss, loss_curve, _, _ = compute_loss_vs_coverage(
        residual_field, router_output, layout, beta, lambda_tv=lambda_tv,
    )
    min_idx = int(np.argmin(loss_curve))
    opt_cov = cov_for_loss[min_idx]
    opt_loss = loss_curve[min_idx]

    ax1.plot(cov_for_loss * 100, loss_curve, 'b-', linewidth=2.5,
             label='Abstention True Loss')
    ax1.plot(0, loss_curve[0], 'o', color='purple', markersize=10, zorder=5)
    ax1.plot(100, loss_curve[-1], 'o', color='teal', markersize=10, zorder=5)
    ax1.plot(opt_cov * 100, opt_loss, '*', color='green', markersize=18,
             zorder=6, markeredgecolor='black', markeredgewidth=1)
    ax1.axhline(y=loss_curve[0], color='purple', linestyle='--',
                linewidth=1.2, alpha=0.5,
                label=f'All PINN: {loss_curve[0]:.4f}')
    ax1.axhline(y=loss_curve[-1], color='teal', linestyle='--',
                linewidth=1.2, alpha=0.5,
                label=f'All CFD: {loss_curve[-1]:.4f}')
    ax1.annotate(f'Opt: {opt_cov*100:.0f}%\n{opt_loss:.4f}',
                 xy=(opt_cov * 100, opt_loss),
                 xytext=(opt_cov * 100 + 8, opt_loss - 0.08),
                 fontsize=9, color='green', ha='left', va='top',
                 fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

    ax1.set_xlabel('Coverage (% solved by CFD)', fontsize=13)
    ax1.set_ylabel(r'Abstention True Loss  $\beta\,c + (1-c)\,\mathbb{E}[R\mid \mathrm{PINN}]$', fontsize=13)
    ax1.set_title(f'Abstention True Loss vs Coverage  (β = {beta})',
                  fontsize=14, fontweight='bold')
    ax1.set_xlim(-5, 105)
    y_min = np.min(loss_curve) - 0.3
    y_max = max(loss_curve[0], loss_curve[-1]) + 0.15
    ax1.set_ylim(y_min, y_max)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=1200, bbox_inches='tight')
        print(f"  Saved loss-vs-coverage plot to {save_path}")
    plt.close(fig)
    return opt_cov, opt_loss


def plot_combined_metrics(coverage, rmse_scores, results, beta, residual_field, router_output, layout,
                          lambda_tv=0.01, lambda_entropy=0.1, save_path=None):
    """
    Create a combined figure: Router Loss vs Coverage + Loss breakdown.

    Parameters:
    -----------
    coverage : ndarray
        Fraction sent to CFD
    rmse_scores : ndarray
        RMSE at each coverage level (unused in this plot, kept for compatibility)
    results : dict
        Results from compute_expected_losses
    beta : float
        Cost coefficient
    residual_field : ndarray
        Per-point normalized physics residual for loss computation
    router_output : ndarray
        Router logits (negative=PINN, positive=CFD)
    layout : ndarray
        Fluid domain mask
    lambda_tv : float
        Total variation regularization weight
    lambda_entropy : float
        Unused, kept for API compatibility.
    save_path : str, optional
        Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ===== Left plot: Training Loss vs Coverage =====
    cov_for_loss, loss_curve, actual_loss_info, optimal_info = compute_loss_vs_coverage(
        residual_field, router_output, layout, beta,
        lambda_tv=lambda_tv
    )

    ax1.plot(cov_for_loss * 100, loss_curve, 'b-', linewidth=2.5, label='Loss Curve')

    # Mark key points
    ax1.plot(0, loss_curve[0], 'o', color='purple', markersize=12, zorder=5)
    ax1.plot(100, loss_curve[-1], 'o', color='teal', markersize=12, zorder=5)

    # Find and mark optimal point (minimum loss)
    min_idx = np.argmin(loss_curve)
    opt_coverage = cov_for_loss[min_idx]
    opt_loss = loss_curve[min_idx]
    ax1.plot(opt_coverage * 100, opt_loss, '*', color='green', markersize=18, zorder=6,
             markeredgecolor='black', markeredgewidth=1)

    # Reference lines
    ax1.axhline(y=loss_curve[0], color='purple', linestyle='--', linewidth=1.5, alpha=0.5, label=f'All PINN: {loss_curve[0]:.4f}')
    ax1.axhline(y=loss_curve[-1], color='teal', linestyle='--', linewidth=1.5, alpha=0.5, label=f'All CFD: {loss_curve[-1]:.4f}')

    # Labels
    ax1.set_xlabel('Coverage (% solved by CFD)', fontsize=14)
    ax1.set_ylabel('Training Loss', fontsize=14)
    ax1.set_title(f'Training Loss vs Coverage (β = {beta})', fontsize=16, fontweight='bold')

    # Annotations
    ax1.annotate(f'All PINN\n{loss_curve[0]:.4f}', xy=(0, loss_curve[0]),
                xytext=(8, loss_curve[0] + 0.03),
                fontsize=9, color='purple', ha='left', va='bottom', fontweight='bold')
    ax1.annotate(f'All CFD\n{loss_curve[-1]:.4f}', xy=(100, loss_curve[-1]),
                xytext=(92, loss_curve[-1] + 0.03),
                fontsize=9, color='teal', ha='right', va='bottom', fontweight='bold')
    ax1.annotate(f'Opt: {opt_coverage*100:.0f}%\n{opt_loss:.4f}',
                xy=(opt_coverage * 100, opt_loss),
                xytext=(opt_coverage * 100 + 8, opt_loss - 0.08),
                fontsize=9, color='green', ha='left', va='top', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

    y_min = np.min(loss_curve) - 0.1
    y_max = max(loss_curve[0], loss_curve[-1]) + 0.15
    ax1.set_xlim(-5, 105)
    ax1.set_ylim(y_min, y_max)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8)

    # ===== Right plot: Optimal point loss breakdown =====
    # Decompose the loss at the optimal coverage into its components
    fluid_mask = layout > 0
    residuals = residual_field[fluid_mask]
    sorted_idx = np.argsort(router_output[fluid_mask])[::-1]
    sorted_residuals = residuals[sorted_idx]
    n_fluid = len(residuals)
    n_cfd_opt = int(opt_coverage * n_fluid)
    n_pinn_opt = n_fluid - n_cfd_opt

    opt_cfd_cost = beta * opt_coverage
    opt_residual_cost = (1 - opt_coverage) * np.mean(sorted_residuals[n_cfd_opt:]) if n_pinn_opt > 0 else 0.0
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
    ax2.set_title(f'Loss Breakdown at Optimal ({opt_coverage*100:.0f}% CFD)', fontsize=16, fontweight='bold')

    # Info box
    info = f"Optimal Coverage: {opt_coverage*100:.1f}%\n"
    info += f"---\n"
    info += f"β={beta}, λ_TV={lambda_tv}"
    ax2.text(0.98, 0.98, info, transform=ax2.transAxes, fontsize=9,
             va='top', ha='right', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=1200, bbox_inches='tight')
        print(f"  Saved combined metrics to {save_path}")
    
    plt.close(fig)  # Close to free memory
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Plot coverage and expected loss metrics for hybrid PINN-CFD router'
    )
    
    # Solution paths
    parser.add_argument('--pinn-path', type=str, default='./models/pinn_cylinder_100.0.h5',
                        help='Path to pre-trained PINN model weights')
    parser.add_argument('--cfd-path', type=str, default=None,
                        help='Path to CFD solution (.npz file with u, v, p)')
    parser.add_argument('--router-weights', type=str, default=None,
                        help='Path to trained router weights (.h5 file)')
    
    # Whether to compute CFD if not provided
    parser.add_argument('--compute-cfd', action='store_true',
                        help='Compute CFD solution (will be saved to --save-cfd path)')
    parser.add_argument('--save-cfd', type=str, default='./cfd_solution.npz',
                        help='Path to save computed CFD solution')
    
    # Router parameters
    parser.add_argument('--base-filters', type=int, default=32,
                        help='Base filters in router CNN (must match training)')
    
    # Cost coefficient
    parser.add_argument('--beta', type=float, default=1,
                        help='Cost coefficient β for CFD computation per node')
    
    # Regularization weights (must match training)
    parser.add_argument('--lambda-tv', type=float, default=0.01,
                        help='Total variation regularization weight')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Manual threshold for router decision (default: 0.0)')
    # Domain parameters (must match PINN training)
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
    
    # CFD solver parameters
    parser.add_argument('--Re', type=float, default=100)
    parser.add_argument('--max-iter', type=int, default=200000)
    parser.add_argument('--tol', type=float, default=1e-6)
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./metrics_output',
                        help='Directory to save output plots')
    parser.add_argument('--morph-kernel', type=int, default=5,
                        help='Kernel size for morphological opening of mask')

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("COVERAGE AND EXPECTED LOSS METRICS")
    print("=" * 60)
    
    # =========================================================================
    # Step 1: Setup domain
    # =========================================================================
    print("\n[Step 1] Setting up domain...")
    
    X, Y, layout, bc_mask, bc_u, bc_v, bc_p = create_cylinder_setup(
        Nx=args.nx,
        Ny=args.ny,
        x_domain=(args.x_min, args.x_max),
        y_domain=(args.y_min, args.y_max),
        cylinder_center=(args.cylinder_x, args.cylinder_y),
        cylinder_radius=args.cylinder_radius,
        inlet_velocity=args.inlet_velocity
    )
    print(f"  Grid: {args.nx} × {args.ny}")
    print(f"  Fluid points: {np.sum(layout):.0f}")
    
    # =========================================================================
    # Step 2: Load PINN model and compute predictions
    # =========================================================================
    print("\n[Step 2] Loading PINN model...")
    
    network = CylinderNetwork()
    pinn_model = network.build(
        num_inputs=2,
        layers=[48, 48, 48, 48],
        activation='tanh',
        num_outputs=3
    )
    pinn_model.load_weights(args.pinn_path)
    print(f"  ✓ Loaded PINN from {args.pinn_path}")
    
    pinn_start_time = time.time()
    u_pinn, v_pinn, p_pinn = load_pinn_solution(pinn_model, X, Y, layout)
    pinn_inference_time = time.time() - pinn_start_time
    print(f"  ✓ PINN inference completed in {pinn_inference_time:.2f} seconds")
    print(f"  PINN u range: [{u_pinn.min():.4f}, {u_pinn.max():.4f}]")
    print(f"  PINN v range: [{v_pinn.min():.4f}, {v_pinn.max():.4f}]")
    
    # =========================================================================
    # Step 3: Load or compute CFD solution (ground truth)
    # =========================================================================
    print("\n[Step 3] Getting CFD solution (ground truth)...")
    
    cfd_time = None  # Track CFD computation time
    
    if args.cfd_path and os.path.exists(args.cfd_path):
        print(f"  Loading from {args.cfd_path}")
        cfd_data = np.load(args.cfd_path)
        u_cfd = cfd_data['u']
        v_cfd = cfd_data['v']
        p_cfd = cfd_data['p']
        # Load cfd_time if it was saved
        if 'cfd_time' in cfd_data:
            cfd_time = float(cfd_data['cfd_time'])
            if cfd_time > 0:
                print(f"  ✓ Loaded CFD solution (originally computed in {cfd_time:.2f}s)")
            else:
                print(f"  ✓ Loaded CFD solution")
        else:
            print(f"  ✓ Loaded CFD solution")
    elif args.compute_cfd:
        u_cfd, v_cfd, p_cfd, X_cfd, Y_cfd, cfd_time = compute_cfd_solution(args)
        # Save for future use
        np.savez(args.save_cfd, u=u_cfd, v=v_cfd, p=p_cfd, X=X_cfd, Y=Y_cfd, cfd_time=cfd_time)
        print(f"  ✓ Saved CFD solution to {args.save_cfd}")
    else:
        print("  ERROR: No CFD solution provided. Use --cfd-path or --compute-cfd")
        return
    
    print(f"  CFD u range: [{u_cfd.min():.4f}, {u_cfd.max():.4f}]")
    print(f"  CFD v range: [{v_cfd.min():.4f}, {v_cfd.max():.4f}]")
    
    # Plot CFD solution
    plot_cfd_solution(
        u_cfd, v_cfd, p_cfd, X, Y, layout,
        cylinder_center=(args.cylinder_x, args.cylinder_y),
        cylinder_radius=args.cylinder_radius,
        save_path=os.path.join(args.output_dir, 'cfd_solution.pdf')
    )
    
    # =========================================================================
    # Step 4: Load router and perform inference
    # =========================================================================
    print("\n[Step 4] Getting router predictions...")

    # Compute error transport field (used for router input and residual field)
    bc_error_local = compute_bc_error_field(
        bc_mask, bc_u, bc_v, u_pinn, v_pinn, layout
    )
    # Dimensional Reynolds number: Re = u_inlet * L_ref / nu, with
    # L_ref = channel height. Solve for nu accordingly.
    L_ref = args.y_max - args.y_min
    nu_eval = args.inlet_velocity * L_ref / args.Re
    error_transport = solve_error_transport(
        u_pinn, v_pinn, bc_error_local, layout, nu=nu_eval,
        x_domain=(args.x_min, args.x_max),
        y_domain=(args.y_min, args.y_max),
    )

    # Pre-compute the PINN PDE residual once; it feeds both the router
    # input (channel 9) and the median-normalised residual_field used as
    # the training target / threshold sweep below.
    residual_computer = PINNResidualComputer(pinn_model, nu=nu_eval, rho=1.0)
    X_tf = tf.constant(X, dtype=tf.float32)
    Y_tf = tf.constant(Y, dtype=tf.float32)
    bc_mask_tf = tf.constant(bc_mask, dtype=tf.float32)
    bc_u_tf = tf.constant(bc_u, dtype=tf.float32)
    bc_v_tf = tf.constant(bc_v, dtype=tf.float32)
    residual_weights = {'continuity': 1.0, 'momentum': 1.0}
    pde_residual_np = residual_computer.compute_total_residual_with_bc(
        X_tf, Y_tf, bc_mask_tf, bc_u_tf, bc_v_tf, residual_weights
    ).numpy().astype(np.float32) * layout

    if args.router_weights and os.path.exists(args.router_weights):
        print(f"  Loading router weights from {args.router_weights}")

        # Create router input tensor (10 channels, all dynamic channels normalised)
        inputs = create_router_input(layout, bc_mask, bc_u, bc_v, bc_p,
                                      u_pinn, v_pinn, p_pinn, error_transport,
                                      pde_residual_np)
        print(f"  Router input shape: {inputs.shape}")
        
        # Initialize router CNN
        router = RouterCNN(base_filters=args.base_filters)
        
        # Build model by running forward pass
        _ = router(inputs)
        
        # Load weights
        router.load_weights(args.router_weights)
        print(f"  ✓ Loaded router weights")
        print(f"  Router parameters: {router.count_params():,}")
        
        # Perform inference
        router_output_4d = router(inputs, training=False)
        router_output = router_output_4d[0, :, :, 0].numpy()  # Remove batch and channel dims
        print(f"  ✓ Router inference complete")
    else:
        # Use error-based proxy for router (higher error -> higher CFD confidence)
        print("  No router weights provided, using error-based proxy...")
        error_field_proxy = compute_l2_error_field(
            u_pinn, v_pinn, p_pinn,
            u_cfd, v_cfd, p_cfd,
            layout, X=X, Y=Y
        )
        # Normalize to [0, 1]
        max_err = np.max(error_field_proxy[layout > 0])
        router_output = error_field_proxy / (max_err + 1e-10)
        print(f"  Using error-based router proxy (for demonstration)")
    
    print(f"  Router output range: [{router_output.min():.4f}, {router_output.max():.4f}]")
    print(f"  Router mean: {np.mean(router_output[layout > 0]):.4f}")
    
    # =========================================================================
    # Step 5: Build the median-normalised residual_field used for the
    # threshold sweep / loss diagnostics. Residual itself was already
    # computed above; here we just add ETE and median-normalise.
    # =========================================================================
    print("\n[Step 5] Building median-normalised residual_field for sweeps...")
    residual_field = pde_residual_np + error_transport
    fluid_residuals = residual_field[layout > 0]
    median_residual = float(np.median(fluid_residuals))
    if median_residual > 1e-10:
        residual_field = residual_field / median_residual

    print(f"  Median residual (raw PDE+BC): {median_residual:.6f}")
    print(f"  Mean residual (normalized): {np.mean(residual_field[layout > 0]):.6f}")
    print(f"  Max residual (normalized): {np.max(residual_field):.6f}")
    
    # Also compute L2 error field for reference (coverage curve uses this)
    print("\n[Step 5b] Computing L2 error field (for R² curve)...")
    
    error_field = compute_l2_error_field(
        u_pinn, v_pinn, p_pinn,
        u_cfd, v_cfd, p_cfd,
        layout, X=X, Y=Y
    )
    print(f"  Mean L2 error (PINN vs CFD): {np.mean(error_field[layout > 0]):.6f}")
    print(f"  Max L2 error: {np.max(error_field):.6f}")
    
    # =========================================================================
    # Step 6: Compute coverage curve (R² as function of CFD coverage)
    # =========================================================================
    print("\n[Step 6] Computing coverage curve...")
    
    # Combine velocity components for calculation
    # Use velocity magnitude: sqrt(u² + v²)
    pinn_vel_mag = np.sqrt(u_pinn**2 + v_pinn**2)
    cfd_vel_mag = np.sqrt(u_cfd**2 + v_cfd**2)
    
    coverage, mse_scores, r2_scores = compute_coverage_curve(pinn_vel_mag, cfd_vel_mag, router_output, layout)
    rmse_scores = np.sqrt(mse_scores)

    print(f"  Coverage range: [{coverage[0]:.4f}, {coverage[-1]:.4f}]")
    print(f"  RMSE at 0% coverage (all PINN): {rmse_scores[0]:.6f}")
    print(f"  RMSE at 100% coverage (all CFD): {rmse_scores[-1]:.6f}")
    
    # =========================================================================
    # Step 7: Compute expected losses (using physics residuals)
    # =========================================================================
    print("\n[Step 7] Computing expected losses (using physics residuals)...")
    
    results = compute_expected_losses(residual_field, router_output, layout, args.beta)
    
    print(f"\n  Expected Loss Comparison (β = {args.beta}):")
    print(f"  ----------------------------------------")
    print(f"  PINN Only:     {results['loss_pinn_only']:.6f} (mean residual)")
    print(f"  CFD Only:      {results['loss_cfd_only']:.6f} (= β)")
    print(f"  Hybrid:        {results['loss_hybrid']:.6f} (threshold: {results['default_threshold']:.2f}, coverage: {results['coverage_hybrid']*100:.1f}%)")
    print(f"  ----------------------------------------")
    print(f"  OPTIMAL THRESHOLD (simple): {results['optimal_threshold']:.4f}")
    print(f"  OPTIMAL LOSS (simple):      {results['optimal_loss']:.6f}")
    print(f"  OPTIMAL COVERAGE (simple):  {results['optimal_coverage']*100:.1f}%")
    print(f"  ----------------------------------------")
    
    # =========================================================================
    # Step 7b: Compute optimal threshold from FULL training loss curve
    # =========================================================================
    print("\n[Step 7b] Computing optimal threshold from full training loss curve...")
    
    # Compute full loss curve to find the threshold corresponding to the star in coverage_metrics.pdf
    _, _, _, full_loss_optimal = compute_loss_vs_coverage(
        residual_field, router_output, layout, args.beta,
        lambda_tv=args.lambda_tv
    )

    print(f"  Full Loss Curve Optimal:")
    print(f"    Threshold: {full_loss_optimal['optimal_threshold']:.6f}")
    print(f"    Coverage:  {full_loss_optimal['optimal_coverage']*100:.2f}%")
    print(f"    Loss:      {full_loss_optimal['optimal_loss']:.6f}")
    
    # =========================================================================
    # Step 7c: Compute actual hybrid solution using threshold
    # =========================================================================
    optimal_threshold = full_loss_optimal['optimal_threshold']
    print(f"\n[Step 7c] Computing hybrid solution with optimal threshold={optimal_threshold:.6f}...")
    print("  Running hybrid PINN-CFD simulation (CFD in high-confidence regions, PINN elsewhere)...")
    
    u_hybrid, v_hybrid, p_hybrid, cfd_mask, hybrid_solve_time = compute_hybrid_solution(
        pinn_model, router_output, layout, optimal_threshold, args
    )
    
    # Compute actual coverage with this threshold
    actual_coverage = np.mean(cfd_mask[layout > 0])
    print(f"  Optimal threshold: {optimal_threshold:.6f}")
    print(f"  CFD coverage: {actual_coverage*100:.2f}%")
    print(f"  PINN coverage: {(1-actual_coverage)*100:.2f}%")
    print(f"  Hybrid solve time: {hybrid_solve_time:.2f} seconds")
    print(f"  Hybrid u range: [{u_hybrid[layout > 0].min():.4f}, {u_hybrid[layout > 0].max():.4f}]")
    print(f"  Hybrid v range: [{v_hybrid[layout > 0].min():.4f}, {v_hybrid[layout > 0].max():.4f}]")

    # Compute RMSE of hybrid solution vs CFD (velocity magnitude)
    hybrid_vel_mag = np.sqrt(u_hybrid**2 + v_hybrid**2)
    fluid_mask = layout > 0
    rmse_hybrid = np.sqrt(np.mean((hybrid_vel_mag[fluid_mask] - cfd_vel_mag[fluid_mask])**2))
    print(f"  Hybrid RMSE (vs CFD): {rmse_hybrid:.6f}")

    # Plot hybrid solution
    plot_hybrid_solution(
        u_hybrid, v_hybrid, p_hybrid, X, Y, layout, cfd_mask,
        cylinder_center=(args.cylinder_x, args.cylinder_y),
        cylinder_radius=args.cylinder_radius,
        threshold=optimal_threshold,
        coverage=actual_coverage,
        save_path=os.path.join(args.output_dir, 'hybrid_solution.pdf')
    )

    # Side-by-side comparison: PINN vs Hybrid vs CFD
    sol_title = (f'PINN vs Hybrid vs CFD  '
                 f'(cyl=({args.cylinder_x:g}, {args.cylinder_y:g}, '
                 f'r={args.cylinder_radius:g}), '
                 f'inlet $u_0$={args.inlet_velocity:g}, Re={args.Re:g})')
    plot_solution_comparison(
        u_pinn, v_pinn, p_pinn,
        u_cfd, v_cfd, p_cfd,
        u_hybrid, v_hybrid, p_hybrid,
        X, Y, layout, cfd_mask,
        cylinder_center=(args.cylinder_x, args.cylinder_y),
        cylinder_radius=args.cylinder_radius,
        save_path=os.path.join(args.output_dir, 'solution_comparison.pdf'),
        title=sol_title,
    )

    # Coverage progression: idealized blend at 10%..100% CFD coverage
    plot_coverage_progression(
        u_pinn, v_pinn, u_cfd, v_cfd,
        router_output, layout, X, Y,
        cylinder_center=(args.cylinder_x, args.cylinder_y),
        cylinder_radius=args.cylinder_radius,
        save_path=os.path.join(args.output_dir, 'coverage_progression.pdf')
    )

    # =========================================================================
    # Step 7d: Baseline hybrid using simple residual-threshold separator
    #   cfd_mask = (residual_field >= beta). Points whose normalized residual
    #   exceeds the CFD cost coefficient are routed to CFD.
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
        cylinder_center=(args.cylinder_x, args.cylinder_y),
        cylinder_radius=args.cylinder_radius,
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
    
    # Combined metrics plot (left: loss vs coverage, right: loss breakdown)
    plot_combined_metrics(
        coverage, rmse_scores, results, args.beta,
        residual_field=residual_field, router_output=router_output, layout=layout,
        lambda_tv=args.lambda_tv,
        save_path=os.path.join(args.output_dir, 'coverage_metrics.pdf')
    )

    # Individual plots
    plot_coverage_curve(
        coverage, rmse_scores, results, args.beta,
        save_path=os.path.join(args.output_dir, 'coverage_curve.pdf')
    )

    # Standalone training-loss vs coverage (left panel of coverage_metrics.pdf).
    plot_loss_vs_coverage(
        residual_field, router_output, layout, args.beta,
        lambda_tv=args.lambda_tv,
        save_path=os.path.join(args.output_dir, 'loss_vs_coverage.pdf'),
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
             pinn_inference_time=pinn_inference_time,
             hybrid_solve_time=hybrid_solve_time,
             hybrid_rmse=rmse_hybrid,
             pinn_rmse=rmse_scores[0],
             u_hybrid=u_hybrid,
             v_hybrid=v_hybrid,
             p_hybrid=p_hybrid,
             cfd_mask=cfd_mask,
             # Run config (for aggregation across configs)
             cfg_cylinder_x=args.cylinder_x,
             cfg_cylinder_y=args.cylinder_y,
             cfg_cylinder_radius=args.cylinder_radius,
             cfg_inlet_velocity=args.inlet_velocity,
             cfg_Re=args.Re,
             cfg_beta=args.beta,
             # Full loss curve optimal (used for hybrid solution)
             full_loss_optimal_threshold=full_loss_optimal['optimal_threshold'],
             full_loss_optimal_coverage=full_loss_optimal['optimal_coverage'],
             full_loss_optimal_loss=full_loss_optimal['optimal_loss'],
             # Residual-threshold baseline hybrid
             u_hybrid_residual=u_hybrid_r,
             v_hybrid_residual=v_hybrid_r,
             p_hybrid_residual=p_hybrid_r,
             cfd_mask_residual=cfd_mask_r,
             residual_threshold=residual_threshold,
             residual_threshold_coverage=actual_coverage_r,
             residual_threshold_rmse=rmse_hybrid_r,
             **results)
    print(f"  ✓ Saved numerical results to {results_path}")
    
    # Print summary box
    print("\n" + "=" * 60)
    print("           RESULTS SUMMARY")
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
    
    # Print timing information
    print("\n" + "=" * 60)
    print("                  TIMING INFORMATION")
    print("=" * 60)
    print(f"  PINN Inference Time:     {pinn_inference_time:.2f} seconds")
    if cfd_time is not None and cfd_time > 0:
        print(f"  CFD Solution Time:       {cfd_time:.2f} seconds")
    else:
        print(f"  CFD Solution Time:       (loaded from file)")
    print(f"  Hybrid Solution Time:    {hybrid_solve_time:.2f} seconds")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("METRICS COMPUTATION COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"  - cfd_solution.pdf: CFD ground truth visualization")
    print(f"  - hybrid_solution.pdf: Hybrid solution with optimal threshold")
    print(f"  - coverage_metrics.pdf: Combined metrics plot")
    print(f"  - coverage_curve.pdf: R² vs coverage curve")
    print(f"  - metrics_results.npz: All numerical results")


if __name__ == "__main__":
    main()
