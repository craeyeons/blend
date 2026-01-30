"""
Plot router separation at multiple thresholds.

This script loads a trained router and generates separation plots
at thresholds from 0.01 to 0.11 (10 increments), saving each as an image.

Usage:
    python plot_router_thresholds.py --router-path ./router_output/router.weights.h5
"""

import argparse
import os
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

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from lib.router import (
    RouterCNN,
    create_router_input,
    create_cylinder_setup,
)


def plot_separation_at_threshold(r, X, Y, layout, threshold, 
                                  cylinder_center, cylinder_radius,
                                  save_path):
    """
    Plot the router separation at a specific threshold.
    
    Parameters:
    -----------
    r : np.ndarray
        Continuous router output (Ny, Nx)
    X, Y : np.ndarray
        Coordinate grids
    layout : np.ndarray
        Layout mask (1=fluid, 0=obstacle)
    threshold : float
        Threshold for binary mask
    cylinder_center : tuple
        (cx, cy) of cylinder
    cylinder_radius : float
        Cylinder radius
    save_path : str
        Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    cx, cy = cylinder_center
    
    # 1. Continuous output with threshold line
    ax = axes[0]
    r_masked = np.ma.masked_where(layout == 0, r)
    cf = ax.contourf(X, Y, r_masked, levels=50, cmap='RdBu_r',
                     norm=Normalize(vmin=0, vmax=1))
    # Add contour line at threshold
    ax.contour(X, Y, r_masked, levels=[threshold], colors='green', 
               linewidths=2, linestyles='--')
    plt.colorbar(cf, ax=ax, label='Router Score')
    circle = plt.Circle((cx, cy), cylinder_radius, color='gray', fill=True)
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Continuous Output (threshold={threshold:.2f} shown as green line)')
    
    # 2. Binary mask at threshold
    ax = axes[1]
    mask = (r >= threshold).astype(np.float32)
    mask_masked = np.ma.masked_where(layout == 0, mask)
    
    # Create colored regions
    combined = np.zeros_like(r)
    combined[layout == 0] = 0  # Obstacle
    combined[(layout == 1) & (r < threshold)] = 1  # PINN region
    combined[(layout == 1) & (r >= threshold)] = 2  # CFD region
    
    cf = ax.contourf(X, Y, combined, levels=[-0.5, 0.5, 1.5, 2.5],
                     colors=['gray', 'blue', 'red'], alpha=0.7)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', label='Obstacle'),
        Patch(facecolor='blue', alpha=0.7, label='PINN'),
        Patch(facecolor='red', alpha=0.7, label='CFD')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    circle = plt.Circle((cx, cy), cylinder_radius, color='gray', fill=True)
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Compute CFD percentage
    fluid_mask = layout == 1
    cfd_fraction = np.sum((r >= threshold) & fluid_mask) / np.sum(fluid_mask) * 100
    ax.set_title(f'Binary Mask (threshold={threshold:.2f}, CFD={cfd_fraction:.1f}%)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot router separation at multiple thresholds'
    )
    
    # Model paths
    parser.add_argument('--router-path', type=str,
                        default='./router_output/router.weights.h5',
                        help='Path to trained router weights')
    parser.add_argument('--output-dir', type=str, default='./threshold_plots',
                        help='Directory to save output images')
    
    # Threshold range
    parser.add_argument('--threshold-start', type=float, default=0.01,
                        help='Starting threshold value')
    parser.add_argument('--threshold-end', type=float, default=0.11,
                        help='Ending threshold value')
    parser.add_argument('--num-thresholds', type=int, default=11,
                        help='Number of threshold values')
    
    # Domain parameters
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
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Domain setup
    x_domain = (args.x_min, args.x_max)
    y_domain = (args.y_min, args.y_max)
    cylinder_center = (args.cylinder_x, args.cylinder_y)
    
    print("Creating domain setup...")
    X, Y, layout, bc_mask, bc_u, bc_v, bc_p = create_cylinder_setup(
        Nx=args.nx, Ny=args.ny,
        x_domain=x_domain, y_domain=y_domain,
        cylinder_center=cylinder_center,
        cylinder_radius=args.cylinder_radius,
        inlet_velocity=args.inlet_velocity
    )
    
    # Create router input
    inputs = create_router_input(layout, bc_mask, bc_u, bc_v, bc_p)
    
    # Load router
    print(f"Loading router from {args.router_path}...")
    router = RouterCNN(base_filters=32)
    
    # Build the model with correct input shape
    dummy_input = tf.zeros((1, args.ny, args.nx, 5))
    _ = router(dummy_input)
    
    # Load weights
    router.load_weights(args.router_path)
    print("Router loaded successfully!")
    
    # Get router output (continuous values)
    inputs_tensor = tf.constant(inputs, dtype=tf.float32)
    r = router(inputs_tensor, training=False)
    r = r[0, :, :, 0].numpy()  # Remove batch and channel dims
    
    # Generate thresholds
    thresholds = np.linspace(args.threshold_start, args.threshold_end, 
                             args.num_thresholds)
    
    print(f"\nGenerating plots for {len(thresholds)} thresholds...")
    print(f"Thresholds: {[f'{t:.2f}' for t in thresholds]}")
    
    # Plot at each threshold
    for i, threshold in enumerate(thresholds):
        save_path = os.path.join(args.output_dir, f'threshold_{threshold:.10f}.png')
        plot_separation_at_threshold(
            r, X, Y, layout, threshold,
            cylinder_center, args.cylinder_radius,
            save_path
        )
    
    # Also save a summary grid with all thresholds
    print("\nGenerating summary grid...")
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, threshold in enumerate(thresholds):
        ax = axes[i]
        
        # Binary mask
        combined = np.zeros_like(r)
        combined[layout == 0] = 0
        combined[(layout == 1) & (r < threshold)] = 1
        combined[(layout == 1) & (r >= threshold)] = 2
        
        ax.contourf(X, Y, combined, levels=[-0.5, 0.5, 1.5, 2.5],
                   colors=['gray', 'blue', 'red'], alpha=0.7)
        circle = plt.Circle(cylinder_center, args.cylinder_radius, 
                           color='gray', fill=True)
        ax.add_patch(circle)
        ax.set_aspect('equal')
        
        fluid_mask = layout == 1
        cfd_frac = np.sum((r >= threshold) & fluid_mask) / np.sum(fluid_mask) * 100
        ax.set_title(f't={threshold:.2f}, CFD={cfd_frac:.1f}%')
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused axes
    for j in range(len(thresholds), len(axes)):
        axes[j].axis('off')
    
    # Add legend to last used plot
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', label='Obstacle'),
        Patch(facecolor='blue', alpha=0.7, label='PINN'),
        Patch(facecolor='red', alpha=0.7, label='CFD')
    ]
    fig.legend(handles=legend_elements, loc='lower right', fontsize=12)
    
    plt.suptitle('Router Separation at Different Thresholds', fontsize=14)
    plt.tight_layout()
    
    summary_path = os.path.join(args.output_dir, 'summary_all_thresholds.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved summary: {summary_path}")
    
    print(f"\nDone! All images saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
