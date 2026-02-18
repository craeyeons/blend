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
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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
    create_router_input,
    create_cylinder_setup,
)
from lib.cylinder_flow import CylinderFlowSimulation
from cylinder_network import Network as CylinderNetwork


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
    
    u_cfd, v_cfd, p_cfd = sim.solve()
    
    return u_cfd, v_cfd, p_cfd, sim.X, sim.Y


def compute_l2_error_field(u_pred, v_pred, p_pred, u_true, v_true, p_true, layout):
    """
    Compute per-point L2 error between prediction and ground truth.
    
    Parameters:
    -----------
    u_pred, v_pred, p_pred : ndarray
        Predicted solution
    u_true, v_true, p_true : ndarray
        Ground truth (CFD) solution
    layout : ndarray
        Fluid domain mask
        
    Returns:
    --------
    error_field : ndarray
        L2 error at each point
    """
    # Velocity error (main focus)
    error_u = (u_pred - u_true) ** 2
    error_v = (v_pred - v_true) ** 2
    
    # Normalize pressure by its range for fair comparison
    p_range = np.max(np.abs(p_true[layout > 0])) + 1e-10
    error_p = ((p_pred - p_true) / p_range) ** 2
    
    # Combined L2 error (can weight differently if needed)
    error_field = np.sqrt(error_u + error_v + error_p) * layout
    
    return error_field


def compute_coverage_curve(pinn_pred, cfd_truth, router_output, layout, beta, n_points=100):
    """
    Compute the router training loss as a function of CFD coverage.
    
    Loss = β × coverage + (1 - coverage) × mean_PINN_error
    
    This matches the loss function used to train the router:
    L = β · Σ r(x_i) + Σ (1 - r(x_i)) · L_residual(PINN, x_i)
    
    Logic:
    - X-axis: coverage = fraction of points solved by CFD (0 to 1)
    - Sort points by router confidence (DESCENDING: highest first)
    - At coverage X%: the top X% points (highest confidence) go to CFD
    - Y-axis: Total loss = β × coverage + (1 - coverage) × PINN_error
    
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
    beta : float
        Cost coefficient for CFD (same as training)
    n_points : int
        Number of points on the curve
        
    Returns:
    --------
    coverage : ndarray
        Fraction solved by CFD (0 to 1)
    total_loss : ndarray
        Router training loss at each coverage level
    pinn_error : ndarray
        Mean L2 error of remaining PINN points
    """
    # Get fluid points only
    fluid_mask = layout > 0
    pinn_vals = pinn_pred[fluid_mask]
    cfd_vals = cfd_truth[fluid_mask]
    confidences = router_output[fluid_mask]
    
    n_fluid = len(pinn_vals)
    
    # Sort by router confidence DESCENDING (highest confidence first → go to CFD first)
    sorted_idx = np.argsort(confidences)[::-1]  # Descending order
    sorted_pinn = pinn_vals[sorted_idx]
    sorted_cfd = cfd_vals[sorted_idx]
    
    # Pre-compute L2 errors (not squared, to match typical loss)
    sorted_l2_errors = np.abs(sorted_cfd - sorted_pinn)
    
    coverage = np.linspace(0, 1, n_points)
    pinn_error = np.zeros(n_points)
    total_loss = np.zeros(n_points)
    
    for i, cov in enumerate(coverage):
        # Number of points sent to CFD (the top cov% with highest confidence)
        n_cfd = int(cov * n_fluid)
        # Number of points kept as PINN (the remaining ones with lower confidence)
        n_pinn = n_fluid - n_cfd
        
        if n_pinn > 0:
            # Mean L2 error of remaining PINN points
            pinn_error[i] = np.mean(sorted_l2_errors[n_cfd:])
        else:
            # All points sent to CFD
            pinn_error[i] = 0.0
        
        # Total loss = β × coverage + (1 - coverage) × PINN_error
        total_loss[i] = beta * cov + (1 - cov) * pinn_error[i]
    
    return coverage, total_loss, pinn_error

def compute_expected_losses(error_field, router_output, layout, beta):
    """
    Compute expected losses for PINN-only, CFD-only, and hybrid systems.
    
    The loss function is:
    L = β · coverage + (1 - coverage) · E[L2_error | PINN]
    
    Where:
    - β is the cost per node of computing CFD
    - coverage is the fraction sent to CFD
    - E[L2_error | PINN] is the expected error of PINN predictions
    
    Parameters:
    -----------
    error_field : ndarray
        Per-point L2 error of PINN vs CFD
    router_output : ndarray
        Router confidence (0=PINN, 1=CFD)
    layout : ndarray
        Fluid domain mask
    beta : float
        Cost coefficient for CFD
        
    Returns:
    --------
    dict with:
        - loss_pinn_only: Expected loss using only PINN
        - loss_cfd_only: Expected loss using only CFD (= β)
        - loss_hybrid: Expected loss using router
        - optimal_coverage: Coverage that minimizes loss
        - optimal_loss: Minimum achievable loss
        - optimal_threshold: Threshold that achieves optimal loss
    """
    fluid_mask = layout > 0
    errors = error_field[fluid_mask]
    confidences = router_output[fluid_mask]
    
    n_fluid = len(errors)
    
    # PINN only: coverage = 0, all errors are PINN errors
    loss_pinn_only = np.mean(errors)
    
    # CFD only: coverage = 1, cost is β per node
    loss_cfd_only = beta
    
    # Hybrid: use router threshold to determine coverage
    # Router output > 0.5 => CFD
    default_threshold = 0.5
    cfd_mask = confidences > default_threshold
    coverage_hybrid = np.mean(cfd_mask)
    
    # Error on PINN points
    pinn_points = ~cfd_mask
    if np.sum(pinn_points) > 0:
        pinn_error = np.mean(errors[pinn_points])
    else:
        pinn_error = 0.0
    
    loss_hybrid = beta * coverage_hybrid + (1 - coverage_hybrid) * pinn_error
    
    # Find optimal operating point by sweeping thresholds on router output
    # Sort by router confidence (ascending)
    sorted_idx = np.argsort(confidences)
    sorted_errors = errors[sorted_idx]
    sorted_confidences = confidences[sorted_idx]
    
    best_loss = float('inf')
    best_coverage = 0.0
    best_threshold = 0.5
    
    # Sweep through possible thresholds
    thresholds = np.linspace(0, 1, 500)
    coverages_list = []
    losses = []
    
    for thresh in thresholds:
        # Points with confidence > thresh go to CFD
        cfd_points = confidences > thresh
        cov = np.mean(cfd_points)
        
        # PINN points are those with confidence <= thresh
        pinn_pts = ~cfd_points
        if np.sum(pinn_pts) > 0:
            pinn_err = np.mean(errors[pinn_pts])
        else:
            pinn_err = 0.0
        
        loss = beta * cov + (1 - cov) * pinn_err
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


def plot_coverage_curve(coverage, total_loss, pinn_error, results, beta, save_path=None):
    """
    Plot router training loss vs coverage curve.
    
    Loss = β × coverage + (1 - coverage) × PINN_error
    
    Parameters:
    -----------
    coverage : ndarray
        Fraction sent to CFD (x-axis)
    total_loss : ndarray
        Router training loss at each coverage level (y-axis)
    pinn_error : ndarray
        Mean PINN error at each coverage level
    results : dict
        Results from compute_expected_losses
    beta : float
        Cost coefficient for CFD
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Main loss curve
    ax.plot(coverage * 100, total_loss, 'b-', linewidth=2.5, label='Total Loss: β×cov + (1-cov)×PINN_err')
    
    # Show components
    ax.plot(coverage * 100, beta * coverage, 'r--', linewidth=1.5, alpha=0.7, label=f'CFD cost term: β×coverage (β={beta})')
    ax.plot(coverage * 100, (1 - coverage) * pinn_error, 'g--', linewidth=1.5, alpha=0.7, label='PINN error term: (1-cov)×err')
    
    # Mark key points
    ax.plot(0, total_loss[0], 'o', color='purple', markersize=12, zorder=5, label=f'All PINN: L={total_loss[0]:.4f}')
    ax.plot(100, total_loss[-1], 'o', color='teal', markersize=12, zorder=5, label=f'All CFD: L={beta:.4f}')
    
    # Mark optimal point
    opt_cov = results['optimal_coverage'] * 100
    opt_loss = results['optimal_loss']
    ax.plot(opt_cov, opt_loss, '*', color='gold', markersize=20, markeredgecolor='black', 
            markeredgewidth=1.5, zorder=6, label=f'Optimal: cov={opt_cov:.1f}%, L={opt_loss:.4f}')
    ax.axvline(x=opt_cov, color='gold', linestyle=':', linewidth=1.5, alpha=0.5)
    
    # Labels
    ax.set_xlabel('Coverage (% solved by CFD)', fontsize=14)
    ax.set_ylabel('Router Training Loss', fontsize=14)
    
    # Set axis limits
    ax.set_xlim(-5, 105)
    y_max = max(total_loss[0], beta) * 1.2
    ax.set_ylim(0, y_max)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_title(f'Router Loss vs Coverage (β={beta})', fontsize=14)
    ax.legend(loc='upper center', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
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
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved loss comparison to {save_path}")
    
    plt.close(fig)  # Close to free memory
    
    return fig


def plot_combined_metrics(coverage, total_loss, pinn_error, results, beta, save_path=None):
    """
    Create a combined figure: Router Loss curve + Expected Loss comparison.
    
    Parameters:
    -----------
    coverage : ndarray
        Fraction sent to CFD
    total_loss : ndarray  
        Router training loss at each coverage level
    pinn_error : ndarray
        Mean PINN error at each coverage level
    results : dict
        Results from compute_expected_losses
    beta : float
        Cost coefficient
    save_path : str, optional
        Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ===== Left plot: Router Loss vs Coverage curve =====
    ax1.plot(coverage * 100, total_loss, 'b-', linewidth=2.5, label='Total Loss')
    
    # Show components
    ax1.plot(coverage * 100, beta * coverage, 'r--', linewidth=1.5, alpha=0.7, label=f'CFD cost: β×cov')
    ax1.plot(coverage * 100, (1 - coverage) * pinn_error, 'g--', linewidth=1.5, alpha=0.7, label='PINN error: (1-cov)×err')
    
    # Mark key points
    ax1.plot(0, total_loss[0], 'o', color='purple', markersize=12, zorder=5)
    ax1.plot(100, total_loss[-1], 'o', color='teal', markersize=12, zorder=5)
    
    # Mark optimal point
    opt_cov = results['optimal_coverage'] * 100
    opt_loss = results['optimal_loss']
    ax1.plot(opt_cov, opt_loss, '*', color='gold', markersize=18, markeredgecolor='black', 
             markeredgewidth=1.5, zorder=6)
    ax1.axvline(x=opt_cov, color='gold', linestyle=':', linewidth=1.5, alpha=0.5)
    
    # Labels
    ax1.set_xlabel('Coverage (% solved by CFD)', fontsize=14)
    ax1.set_ylabel('Router Training Loss', fontsize=14)
    ax1.set_title(f'Coverage vs Router Loss (β={beta})', fontsize=16, fontweight='bold')
    
    # Annotations
    ax1.annotate(f'All PINN\nL={total_loss[0]:.4f}', xy=(0, total_loss[0]), xytext=(10, total_loss[0]),
                fontsize=10, color='purple', ha='left', va='bottom', fontweight='bold')
    ax1.annotate(f'All CFD\nL={beta:.4f}', xy=(100, total_loss[-1]), xytext=(90, total_loss[-1]),
                fontsize=10, color='teal', ha='right', va='bottom', fontweight='bold')
    ax1.annotate(f'Optimal\ncov={opt_cov:.1f}%\nL={opt_loss:.4f}', 
                xy=(opt_cov, opt_loss), xytext=(opt_cov+5, opt_loss-0.02),
                fontsize=10, color='goldenrod', ha='left', va='top', fontweight='bold')
    
    ax1.set_xlim(-5, 105)
    y_max = max(total_loss[0], beta) * 1.2
    ax1.set_ylim(0, y_max)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper center', fontsize=9)
    
    # ===== Right plot: Loss comparison bar chart =====
    loss_pinn = results['loss_pinn_only']
    loss_cfd = results['loss_cfd_only']
    methods = ['PINN\nOnly', f'CFD Only\n(β={beta})', 'Hybrid\n(Router)', 'Optimal']
    losses = [
        results['loss_pinn_only'],
        results['loss_cfd_only'],
        results['loss_hybrid'],
        results['optimal_loss']
    ]
    colors = ['purple', 'gray', 'teal', 'green']
    
    bars = ax2.bar(methods, losses, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, loss in zip(bars, losses):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{loss:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('Expected True Loss', fontsize=14)
    ax2.set_title('Expected Loss Comparison', fontsize=16, fontweight='bold')
    
    # Info box
    info = f"Default Threshold: {results['default_threshold']:.2f}\n"
    info += f"Router Coverage: {results['coverage_hybrid']*100:.1f}%\n"
    info += f"---\n"
    info += f"Optimal Threshold: {results['optimal_threshold']:.4f}\n"
    info += f"Optimal Coverage: {results['optimal_coverage']*100:.1f}%"
    ax2.text(0.98, 0.98, info, transform=ax2.transAxes, fontsize=10,
             va='top', ha='right', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
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
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Router temperature (must match training)')
    
    # Cost coefficient
    parser.add_argument('--beta', type=float, default=1,
                        help='Cost coefficient β for CFD computation per node')
    
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
    
    u_pinn, v_pinn, p_pinn = load_pinn_solution(pinn_model, X, Y, layout)
    print(f"  PINN u range: [{u_pinn.min():.4f}, {u_pinn.max():.4f}]")
    print(f"  PINN v range: [{v_pinn.min():.4f}, {v_pinn.max():.4f}]")
    
    # =========================================================================
    # Step 3: Load or compute CFD solution (ground truth)
    # =========================================================================
    print("\n[Step 3] Getting CFD solution (ground truth)...")
    
    if args.cfd_path and os.path.exists(args.cfd_path):
        print(f"  Loading from {args.cfd_path}")
        cfd_data = np.load(args.cfd_path)
        u_cfd = cfd_data['u']
        v_cfd = cfd_data['v']
        p_cfd = cfd_data['p']
        print(f"  ✓ Loaded CFD solution")
    elif args.compute_cfd:
        u_cfd, v_cfd, p_cfd, X_cfd, Y_cfd = compute_cfd_solution(args)
        # Save for future use
        np.savez(args.save_cfd, u=u_cfd, v=v_cfd, p=p_cfd, X=X_cfd, Y=Y_cfd)
        print(f"  ✓ Saved CFD solution to {args.save_cfd}")
    else:
        print("  ERROR: No CFD solution provided. Use --cfd-path or --compute-cfd")
        return
    
    print(f"  CFD u range: [{u_cfd.min():.4f}, {u_cfd.max():.4f}]")
    print(f"  CFD v range: [{v_cfd.min():.4f}, {v_cfd.max():.4f}]")
    
    # =========================================================================
    # Step 4: Load router and perform inference
    # =========================================================================
    print("\n[Step 4] Getting router predictions...")
    
    if args.router_weights and os.path.exists(args.router_weights):
        print(f"  Loading router weights from {args.router_weights}")
        
        # Create router input tensor
        inputs = create_router_input(layout, bc_mask, bc_u, bc_v, bc_p,
                                      u_pinn, v_pinn, p_pinn)
        print(f"  Router input shape: {inputs.shape}")
        
        # Initialize router CNN
        router = RouterCNN(base_filters=args.base_filters, temperature=args.temperature)
        
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
            layout
        )
        # Normalize to [0, 1]
        max_err = np.max(error_field_proxy[layout > 0])
        router_output = error_field_proxy / (max_err + 1e-10)
        print(f"  Using error-based router proxy (for demonstration)")
    
    print(f"  Router output range: [{router_output.min():.4f}, {router_output.max():.4f}]")
    print(f"  Router mean: {np.mean(router_output[layout > 0]):.4f}")
    
    # =========================================================================
    # Step 5: Compute error field
    # =========================================================================
    print("\n[Step 5] Computing error field...")
    
    error_field = compute_l2_error_field(
        u_pinn, v_pinn, p_pinn,
        u_cfd, v_cfd, p_cfd,
        layout
    )
    print(f"  Mean L2 error (PINN vs CFD): {np.mean(error_field[layout > 0]):.6f}")
    print(f"  Max L2 error: {np.max(error_field):.6f}")
    
    # =========================================================================
    # Step 6: Compute coverage curve (Router loss as function of CFD coverage)
    # =========================================================================
    print("\n[Step 6] Computing coverage curve...")
    
    # Combine velocity components for calculation
    # Use velocity magnitude: sqrt(u² + v²)
    pinn_vel_mag = np.sqrt(u_pinn**2 + v_pinn**2)
    cfd_vel_mag = np.sqrt(u_cfd**2 + v_cfd**2)
    
    coverage, total_loss, pinn_error = compute_coverage_curve(
        pinn_vel_mag, cfd_vel_mag, router_output, layout, args.beta
    )
    
    print(f"  Coverage range: [{coverage[0]:.4f}, {coverage[-1]:.4f}]")
    print(f"  Loss at 0% coverage (all PINN): {total_loss[0]:.6f}")
    print(f"  Loss at 100% coverage (all CFD): {total_loss[-1]:.6f} (= β = {args.beta})")
    print(f"  PINN error at 0% cov: {pinn_error[0]:.6f}")
    
    # Find minimum loss point
    min_idx = np.argmin(total_loss)
    min_cov = coverage[min_idx]
    min_loss = total_loss[min_idx]
    print(f"  Minimum loss: {min_loss:.6f} at coverage {min_cov*100:.1f}%")
    
    # =========================================================================
    # Step 7: Compute expected losses
    # =========================================================================
    print("\n[Step 7] Computing expected losses...")
    
    results = compute_expected_losses(error_field, router_output, layout, args.beta)
    
    print(f"\n  Expected Loss Comparison (β = {args.beta}):")
    print(f"  ----------------------------------------")
    print(f"  PINN Only:     {results['loss_pinn_only']:.6f}")
    print(f"  CFD Only:      {results['loss_cfd_only']:.6f} (= β)")
    print(f"  Hybrid:        {results['loss_hybrid']:.6f} (threshold: {results['default_threshold']:.2f}, coverage: {results['coverage_hybrid']*100:.1f}%)")
    print(f"  ----------------------------------------")
    print(f"  OPTIMAL THRESHOLD: {results['optimal_threshold']:.4f}")
    print(f"  OPTIMAL LOSS:      {results['optimal_loss']:.6f}")
    print(f"  OPTIMAL COVERAGE:  {results['optimal_coverage']*100:.1f}%")
    print(f"  ----------------------------------------")
    
    # =========================================================================
    # Step 8: Generate plots
    # =========================================================================
    print("\n[Step 8] Generating plots...")
    
    # Combined metrics plot
    plot_combined_metrics(
        coverage, total_loss, pinn_error, results, args.beta,
        save_path=os.path.join(args.output_dir, 'coverage_metrics.png')
    )
    
    # Individual plots
    plot_coverage_curve(
        coverage, total_loss, pinn_error, results, args.beta,
        save_path=os.path.join(args.output_dir, 'coverage_curve.png')
    )
    
    plot_expected_loss_comparison(
        results, args.beta,
        save_path=os.path.join(args.output_dir, 'loss_comparison.png')
    )
    
    # Save numerical results
    results_path = os.path.join(args.output_dir, 'metrics_results.npz')
    np.savez(results_path,
             coverage=coverage,
             total_loss=total_loss,
             pinn_error=pinn_error,
             router_output=router_output,
             error_field=error_field,
             beta=args.beta,
             **results)
    print(f"  ✓ Saved numerical results to {results_path}")
    
    # Print summary box
    print("\n" + "=" * 60)
    print("                    OPTIMAL RESULTS")
    print("=" * 60)
    print(f"  β (CFD cost):        {args.beta}")
    print(f"  ----------------------------------------")
    print(f"  OPTIMAL THRESHOLD:   {results['optimal_threshold']:.6f}")
    print(f"  OPTIMAL LOSS:        {results['optimal_loss']:.6f}")
    print(f"  OPTIMAL COVERAGE:    {results['optimal_coverage']*100:.2f}%")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("METRICS COMPUTATION COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
