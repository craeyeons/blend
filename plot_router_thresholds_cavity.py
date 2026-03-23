"""
Plot router separation at multiple thresholds for CAVITY FLOW.

This script loads a trained router and generates separation plots
at multiple thresholds, saving each as an image.

Usage:
    python plot_router_thresholds_cavity.py --router-path ./router_output_cavity/router.weights.h5
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
import scienceplots
plt.style.use(['science', 'no-latex'])
from matplotlib.colors import Normalize

from lib.router import (
    RouterCNN,
    create_router_input,
    compute_smeared_bc_error,
    create_cavity_setup,
)
from lib.network import Network as CavityNetwork


def compute_uv_from_psi(pinn_model, xy):
    """
    Compute (u, v) from stream function: u = ∂ψ/∂y, v = -∂ψ/∂x
    """
    xy_tf = tf.constant(xy, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(xy_tf)
        psi_p = pinn_model(xy_tf, training=False)
        psi = psi_p[:, 0]
    grad_psi = tape.gradient(psi, xy_tf)
    u = grad_psi[:, 1].numpy()   # ∂ψ/∂y
    v = -grad_psi[:, 0].numpy()  # -∂ψ/∂x
    return u, v


def plot_separation_at_threshold(r, X, Y, layout, threshold, save_path):
    """
    Plot the router separation at a specific threshold for cavity flow.
    
    Parameters:
    -----------
    r : np.ndarray
        Continuous router output (N, N)
    X, Y : np.ndarray
        Coordinate grids
    layout : np.ndarray
        Layout mask (all 1s for cavity)
    threshold : float
        Threshold for binary mask
    save_path : str
        Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Continuous output with threshold line
    ax = axes[0]
    r_masked = np.ma.masked_where(layout == 0, r)
    cf = ax.contourf(X, Y, r_masked, levels=50, cmap='RdBu_r',
                     norm=Normalize(vmin=0, vmax=1))
    # Add contour line at threshold
    ax.contour(X, Y, r_masked, levels=[threshold], colors='green', 
               linewidths=2, linestyles='--')
    plt.colorbar(cf, ax=ax, label='Router Score')
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Continuous Output (threshold={threshold:.2f} shown as green line)')
    
    # 2. Binary mask at threshold
    ax = axes[1]
    
    # Create colored regions
    combined = np.zeros_like(r)
    combined[(layout == 1) & (r < threshold)] = 1  # PINN region
    combined[(layout == 1) & (r >= threshold)] = 2  # CFD region
    
    cf = ax.contourf(X, Y, combined, levels=[0.5, 1.5, 2.5],
                     colors=['blue', 'red'], alpha=0.7)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label='PINN'),
        Patch(facecolor='red', alpha=0.7, label='CFD')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
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
        description='Plot router separation at multiple thresholds for cavity flow'
    )
    
    # Model paths
    parser.add_argument('--router-path', type=str,
                        default='./router_output_cavity/router.weights.h5',
                        help='Path to trained router weights')
    parser.add_argument('--pinn-path', type=str,
                        default='./models/pinn_cavity_flow.h5',
                        help='Path to pre-trained PINN model')
    parser.add_argument('--output-dir', type=str, default='./threshold_plots_cavity',
                        help='Directory to save output images')
    
    # Threshold range
    parser.add_argument('--threshold-start', type=float, default=0.1,
                        help='Starting threshold value')
    parser.add_argument('--threshold-end', type=float, default=0.9,
                        help='Ending threshold value')
    parser.add_argument('--num-thresholds', type=int, default=9,
                        help='Number of threshold values')
    
    # Domain parameters (cavity is square)
    parser.add_argument('--N', type=int, default=100,
                        help='Grid size (N x N)')
    parser.add_argument('--x-min', type=float, default=0.0)
    parser.add_argument('--x-max', type=float, default=1.0)
    parser.add_argument('--y-min', type=float, default=0.0)
    parser.add_argument('--y-max', type=float, default=1.0)
    parser.add_argument('--lid-velocity', type=float, default=1.0)
    
    # Router parameters
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Sigmoid temperature for router (should match training)')
    parser.add_argument('--base-filters', type=int, default=32,
                        help='Base filters in router CNN')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Domain setup
    x_domain = (args.x_min, args.x_max)
    y_domain = (args.y_min, args.y_max)
    
    print("Creating domain setup...")
    X, Y, layout, bc_mask, bc_u, bc_v, bc_p = create_cavity_setup(
        N=args.N,
        x_domain=x_domain, 
        y_domain=y_domain,
        lid_velocity=args.lid_velocity
    )
    
    # Load PINN model for predictions
    print(f"Loading PINN model from {args.pinn_path}...")
    network = CavityNetwork()
    pinn_model = network.build(
        num_inputs=2,
        layers=[32, 16, 16, 32],
        activation='swish',
        num_outputs=2  # (psi, p)
    )
    pinn_model.load_weights(args.pinn_path)
    
    # Compute PINN predictions for router input
    xy_flat = np.stack([X.flatten(), Y.flatten()], axis=-1).astype(np.float32)
    
    # Get velocities from stream function
    u_flat, v_flat = compute_uv_from_psi(pinn_model, xy_flat)
    
    # Get pressure
    psi_p = pinn_model.predict(xy_flat, batch_size=len(xy_flat), verbose=0)
    
    pinn_u = u_flat.reshape(X.shape).astype(np.float32) * layout
    pinn_v = v_flat.reshape(X.shape).astype(np.float32) * layout
    pinn_p = psi_p[:, 1].reshape(X.shape).astype(np.float32) * layout
    
    # Compute smeared BC error
    smeared_bc_err = compute_smeared_bc_error(
        bc_mask, bc_u, bc_v, pinn_u, pinn_v, layout
    )

    # Create router input (9 channels including PINN predictions)
    inputs = create_router_input(layout, bc_mask, bc_u, bc_v, bc_p,
                                  pinn_u, pinn_v, pinn_p, smeared_bc_err)
    
    # Load router
    print(f"Loading router from {args.router_path}...")
    router = RouterCNN(base_filters=args.base_filters, temperature=args.temperature)
    
    # Build the model with correct input shape (9 channels)
    _ = router(inputs)
    
    # Load weights
    router.load_weights(args.router_path)
    print(f"Router loaded successfully! (temperature={args.temperature})")
    
    # Get router output (continuous values)
    inputs_tensor = tf.constant(inputs, dtype=tf.float32)
    r = router(inputs_tensor, training=False)
    r = r[0, :, :, 0].numpy()  # Remove batch and channel dims
    
    # Print router output statistics for debugging
    print(f"\nRouter output statistics:")
    print(f"  min: {r.min():.6f}, max: {r.max():.6f}")
    print(f"  mean: {r.mean():.6f}, std: {r.std():.6f}")
    print(f"  median: {np.median(r):.6f}")
    
    # Generate thresholds
    thresholds = np.linspace(args.threshold_start, args.threshold_end, 
                             args.num_thresholds)
    
    print(f"\nGenerating plots for {len(thresholds)} thresholds...")
    print(f"Thresholds: {[f'{t:.2f}' for t in thresholds]}")
    
    # Plot at each threshold
    for i, threshold in enumerate(thresholds):
        save_path = os.path.join(args.output_dir, f'threshold_{threshold:.4f}.png')
        plot_separation_at_threshold(
            r, X, Y, layout, threshold,
            save_path
        )
    
    # Also save a summary grid with all thresholds
    print("\nGenerating summary grid...")
    n_cols = 3
    n_rows = (len(thresholds) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, threshold in enumerate(thresholds):
        ax = axes[i]
        
        # Binary mask
        combined = np.zeros_like(r)
        combined[(layout == 1) & (r < threshold)] = 1
        combined[(layout == 1) & (r >= threshold)] = 2
        
        ax.contourf(X, Y, combined, levels=[0.5, 1.5, 2.5],
                   colors=['blue', 'red'], alpha=0.7)
        ax.set_aspect('equal')
        
        fluid_mask = layout == 1
        cfd_frac = np.sum((r >= threshold) & fluid_mask) / np.sum(fluid_mask) * 100
        ax.set_title(f't={threshold:.2f}, CFD={cfd_frac:.1f}%')
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused axes
    for j in range(len(thresholds), len(axes)):
        axes[j].axis('off')
    
    # Add legend to figure
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label='PINN'),
        Patch(facecolor='red', alpha=0.7, label='CFD')
    ]
    fig.legend(handles=legend_elements, loc='lower right', fontsize=12)
    
    plt.suptitle('Router Separation at Different Thresholds (Cavity Flow)', fontsize=14)
    plt.tight_layout()
    
    summary_path = os.path.join(args.output_dir, 'summary_all_thresholds.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved summary: {summary_path}")
    
    print(f"\nDone! All images saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
