"""
Train and evaluate the learned router for hybrid PINN-CFD flow simulation.

This script demonstrates the physics-based router training that learns to
decide whether to accept PINN predictions or use CFD at each spatial location.

Key features:
- No ground truth CFD data required - trains from physics residuals only
- Soft-hard decoupling for differentiable training with discrete CFD execution
- Spatial smoothness regularization for connected CFD regions
- NO forced boundary/grid regions - purely learned from physics

Usage:
    python router_train.py --Re 100 --epochs 200
    python router_train.py --Re 100 --epochs 500 --lambda_spatial 0.2
"""

import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from lib.router import (
    RouterNetwork,
    RouterFeatureExtractor,
    PhysicsLoss,
    HybridRouterTrainer,
    create_router_cfd_solver
)
from lib.cylinder_flow import CylinderFlowSimulation
from lib.plotting import plot_solution, plot_streamlines
from cylinder_network import Network as CylinderNetwork


def plot_router_mask(r_soft, r_binary, X, Y, cylinder_center, cylinder_radius,
                     title="Router Learned Mask", save_path=None):
    """
    Visualize the router's learned mask.
    
    Parameters:
    -----------
    r_soft : ndarray of shape (Ny, Nx)
        Soft probability mask [0, 1]
    r_binary : ndarray of shape (Ny, Nx)
        Binary mask (0=PINN, 1=CFD)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot soft probabilities
    ax1 = axes[0]
    im1 = ax1.pcolormesh(X, Y, r_soft, cmap='RdYlBu_r', vmin=0, vmax=1)
    plt.colorbar(im1, ax=ax1, label='CFD Probability')
    
    # Draw cylinder
    circle1 = plt.Circle(cylinder_center, cylinder_radius, 
                        fill=False, color='black', linewidth=2)
    ax1.add_patch(circle1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Router Soft Probabilities r(x,y)')
    ax1.set_aspect('equal')
    
    # Plot binary mask
    ax2 = axes[1]
    cmap_binary = plt.cm.colors.ListedColormap(['#2ecc71', '#e74c3c'])  # Green=PINN, Red=CFD
    im2 = ax2.pcolormesh(X, Y, r_binary, cmap=cmap_binary, vmin=0, vmax=1)
    cbar2 = plt.colorbar(im2, ax=ax2, ticks=[0.25, 0.75])
    cbar2.ax.set_yticklabels(['PINN', 'CFD'])
    
    # Draw cylinder
    circle2 = plt.Circle(cylinder_center, cylinder_radius,
                        fill=True, color='gray')
    ax2.add_patch(circle2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Binary Mask (threshold=0.5)')
    ax2.set_aspect('equal')
    
    # Add statistics
    cfd_fraction = np.mean(r_binary)
    pinn_fraction = 1 - cfd_fraction
    fig.suptitle(f"{title}\nCFD: {100*cfd_fraction:.1f}% | PINN: {100*pinn_fraction:.1f}%",
                fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved mask visualization to {save_path}")
    
    plt.show()


def plot_training_history(history, save_path=None):
    """Plot training history metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    epochs = range(len(history['total_loss']))
    
    # Total loss
    axes[0, 0].semilogy(epochs, history['total_loss'], 'b-', linewidth=1.5)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Physics loss
    axes[0, 1].semilogy(epochs, history['physics_loss'], 'r-', linewidth=1.5)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Physics Loss')
    axes[0, 1].set_title('Navier-Stokes Residual')
    axes[0, 1].grid(True, alpha=0.3)
    
    # BC loss
    axes[0, 2].semilogy(epochs, history['bc_loss'], 'g-', linewidth=1.5)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('BC Loss')
    axes[0, 2].set_title('Boundary Condition Violation')
    axes[0, 2].grid(True, alpha=0.3)
    
    # CFD fraction
    axes[1, 0].plot(epochs, [100*x for x in history['cfd_fraction']], 'purple', linewidth=1.5)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('CFD Fraction (%)')
    axes[1, 0].set_title('CFD Region Fraction')
    axes[1, 0].set_ylim([0, 100])
    axes[1, 0].grid(True, alpha=0.3)
    
    # Inlet rejection rate
    axes[1, 1].plot(epochs, [100*x for x in history['inlet_rejection_rate']], 'orange', linewidth=1.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Inlet Rejection Rate (%)')
    axes[1, 1].set_title('Inlet Region → CFD Rate')
    axes[1, 1].set_ylim([0, 100])
    axes[1, 1].grid(True, alpha=0.3)
    
    # Spatial and cost losses
    axes[1, 2].semilogy(epochs, history['spatial_loss'], 'c-', label='Spatial', linewidth=1.5)
    axes[1, 2].semilogy(epochs, history['cost_loss'], 'm-', label='Cost', linewidth=1.5)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].set_title('Regularization Losses')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history to {save_path}")
    
    plt.show()


def plot_solution_comparison(u_pinn, v_pinn, u_hybrid, v_hybrid, u_cfd, v_cfd,
                             X, Y, cylinder_center, cylinder_radius, save_path=None):
    """Compare PINN, hybrid, and pure CFD solutions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Velocity magnitude for each solution
    vel_pinn = np.sqrt(u_pinn**2 + v_pinn**2)
    vel_hybrid = np.sqrt(u_hybrid**2 + v_hybrid**2)
    vel_cfd = np.sqrt(u_cfd**2 + v_cfd**2)
    
    vmax = max(vel_pinn.max(), vel_hybrid.max(), vel_cfd.max())
    
    # PINN solution
    im1 = axes[0, 0].pcolormesh(X, Y, vel_pinn, cmap='viridis', vmin=0, vmax=vmax)
    plt.colorbar(im1, ax=axes[0, 0])
    circle = plt.Circle(cylinder_center, cylinder_radius, fill=True, color='gray')
    axes[0, 0].add_patch(circle)
    axes[0, 0].set_title('PINN: |u|')
    axes[0, 0].set_aspect('equal')
    
    # Hybrid solution
    im2 = axes[0, 1].pcolormesh(X, Y, vel_hybrid, cmap='viridis', vmin=0, vmax=vmax)
    plt.colorbar(im2, ax=axes[0, 1])
    circle = plt.Circle(cylinder_center, cylinder_radius, fill=True, color='gray')
    axes[0, 1].add_patch(circle)
    axes[0, 1].set_title('Hybrid (Router): |u|')
    axes[0, 1].set_aspect('equal')
    
    # Pure CFD solution
    im3 = axes[0, 2].pcolormesh(X, Y, vel_cfd, cmap='viridis', vmin=0, vmax=vmax)
    plt.colorbar(im3, ax=axes[0, 2])
    circle = plt.Circle(cylinder_center, cylinder_radius, fill=True, color='gray')
    axes[0, 2].add_patch(circle)
    axes[0, 2].set_title('Pure CFD: |u|')
    axes[0, 2].set_aspect('equal')
    
    # Errors vs CFD
    err_pinn = vel_pinn - vel_cfd
    err_hybrid = vel_hybrid - vel_cfd
    
    err_max = max(np.abs(err_pinn).max(), np.abs(err_hybrid).max())
    
    # PINN error
    im4 = axes[1, 0].pcolormesh(X, Y, err_pinn, cmap='RdBu_r', vmin=-err_max, vmax=err_max)
    plt.colorbar(im4, ax=axes[1, 0])
    circle = plt.Circle(cylinder_center, cylinder_radius, fill=True, color='gray')
    axes[1, 0].add_patch(circle)
    axes[1, 0].set_title(f'PINN Error (RMSE={np.sqrt(np.mean(err_pinn**2)):.4f})')
    axes[1, 0].set_aspect('equal')
    
    # Hybrid error
    im5 = axes[1, 1].pcolormesh(X, Y, err_hybrid, cmap='RdBu_r', vmin=-err_max, vmax=err_max)
    plt.colorbar(im5, ax=axes[1, 1])
    circle = plt.Circle(cylinder_center, cylinder_radius, fill=True, color='gray')
    axes[1, 1].add_patch(circle)
    axes[1, 1].set_title(f'Hybrid Error (RMSE={np.sqrt(np.mean(err_hybrid**2)):.4f})')
    axes[1, 1].set_aspect('equal')
    
    # Improvement factor
    improvement = np.abs(err_pinn) - np.abs(err_hybrid)
    im6 = axes[1, 2].pcolormesh(X, Y, improvement, cmap='RdYlGn', vmin=-err_max/2, vmax=err_max/2)
    plt.colorbar(im6, ax=axes[1, 2])
    circle = plt.Circle(cylinder_center, cylinder_radius, fill=True, color='gray')
    axes[1, 2].add_patch(circle)
    axes[1, 2].set_title('Error Improvement (green = hybrid better)')
    axes[1, 2].set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved solution comparison to {save_path}")
    
    plt.show()
    
    # Print summary statistics
    rmse_pinn = np.sqrt(np.mean(err_pinn**2))
    rmse_hybrid = np.sqrt(np.mean(err_hybrid**2))
    improvement_pct = 100 * (rmse_pinn - rmse_hybrid) / rmse_pinn
    
    print(f"\n{'='*50}")
    print("Solution Quality Comparison (vs Pure CFD)")
    print(f"{'='*50}")
    print(f"PINN RMSE:   {rmse_pinn:.6f}")
    print(f"Hybrid RMSE: {rmse_hybrid:.6f}")
    print(f"Improvement: {improvement_pct:.1f}%")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="Train learned router for hybrid PINN-CFD")
    
    # Flow parameters
    parser.add_argument('--Re', type=float, default=100, help='Reynolds number')
    parser.add_argument('--grid_size', type=int, default=100, help='Base grid size N')
    parser.add_argument('--inlet_velocity', type=float, default=1.0, help='Inlet velocity')
    
    # Domain parameters
    parser.add_argument('--x_min', type=float, default=0.0, help='Domain x min')
    parser.add_argument('--x_max', type=float, default=2.0, help='Domain x max')
    parser.add_argument('--y_min', type=float, default=0.0, help='Domain y min')
    parser.add_argument('--y_max', type=float, default=1.0, help='Domain y max')
    
    # Cylinder parameters
    parser.add_argument('--cyl_x', type=float, default=0.5, help='Cylinder center x')
    parser.add_argument('--cyl_y', type=float, default=0.5, help='Cylinder center y')
    parser.add_argument('--cyl_radius', type=float, default=0.1, help='Cylinder radius')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--threshold', type=float, default=0.5, help='Binary threshold')
    
    # Loss weights
    parser.add_argument('--lambda_BC', type=float, default=10.0, help='BC loss weight')
    parser.add_argument('--lambda_spatial', type=float, default=0.1, help='Spatial smoothness weight')
    parser.add_argument('--lambda_cost', type=float, default=0.01, help='Cost penalty weight')
    
    # Router architecture
    parser.add_argument('--num_filters', type=int, default=32, help='Conv filters')
    parser.add_argument('--num_layers', type=int, default=4, help='Conv layers')
    
    # CFD solver parameters
    parser.add_argument('--cfd_max_iter', type=int, default=30000, help='CFD max iterations')
    parser.add_argument('--cfd_tol', type=float, default=1e-5, help='CFD convergence tolerance')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./images/router', help='Output directory')
    parser.add_argument('--model_path', type=str, default=None, help='Path to PINN model')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("LEARNED ROUTER FOR HYBRID PINN-CFD SIMULATION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Reynolds number: {args.Re}")
    print(f"  Grid size: {args.grid_size}")
    print(f"  Training epochs: {args.epochs}")
    print(f"  Loss weights: λ_BC={args.lambda_BC}, λ_spatial={args.lambda_spatial}, λ_cost={args.lambda_cost}")
    print()
    
    # Physical parameters
    nu = 1.0 / args.Re
    rho = 1.0
    x_domain = (args.x_min, args.x_max)
    y_domain = (args.y_min, args.y_max)
    cylinder_center = (args.cyl_x, args.cyl_y)
    cylinder_radius = args.cyl_radius
    
    # Grid setup
    Lx = args.x_max - args.x_min
    Ly = args.y_max - args.y_min
    aspect_ratio = Lx / Ly
    
    Ny = args.grid_size
    Nx = int(args.grid_size * aspect_ratio)
    
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    
    x = np.linspace(args.x_min, args.x_max, Nx)
    y = np.linspace(args.y_min, args.y_max, Ny)
    X, Y = np.meshgrid(x, y)
    xy = np.stack([X.flatten(), Y.flatten()], axis=-1)
    
    # Create cylinder mask
    dist_sq = (X - args.cyl_x)**2 + (Y - args.cyl_y)**2
    cylinder_mask = (dist_sq <= args.cyl_radius**2).astype(np.int32)
    
    print(f"Grid: {Nx}x{Ny} (dx={dx:.4f}, dy={dy:.4f})")
    
    # =========================================================================
    # Step 1: Load pre-trained PINN
    # =========================================================================
    print("\nStep 1: Loading pre-trained PINN model...")
    
    pinn_network = CylinderNetwork().build(
        num_inputs=2, layers=[48, 48, 48, 48], 
        activation='tanh', num_outputs=3
    )
    
    # Auto-select model based on Re
    if args.model_path:
        model_path = args.model_path
    else:
        if args.Re <= 10:
            model_path = './models/pinn_cylinder_1.0.h5'
        else:
            model_path = './models/pinn_cylinder_100.0.h5'
    
    try:
        pinn_network.load_weights(model_path)
        print(f"  ✓ Loaded PINN from {model_path}")
    except Exception as e:
        print(f"  ⚠ Warning: Could not load PINN model: {e}")
        print("  Using untrained network (results will be poor)")
    
    # =========================================================================
    # Step 2: Get pure CFD solution for comparison
    # =========================================================================
    print("\nStep 2: Computing pure CFD solution for comparison...")
    
    cfd_sim = CylinderFlowSimulation(
        Re=args.Re,
        N=args.grid_size,
        max_iter=100000,
        tol=1e-6,
        x_domain=x_domain,
        y_domain=y_domain,
        cylinder_center=cylinder_center,
        cylinder_radius=cylinder_radius,
        inlet_velocity=args.inlet_velocity
    )
    
    start_time = time.time()
    u_cfd, v_cfd, p_cfd = cfd_sim.solve()
    cfd_time = time.time() - start_time
    print(f"  Pure CFD solution time: {cfd_time:.2f}s")
    
    # =========================================================================
    # Step 3: Create router components
    # =========================================================================
    print("\nStep 3: Initializing router components...")
    
    # Router network
    router = RouterNetwork(
        num_input_features=14,  # Number of feature channels
        num_filters=args.num_filters,
        kernel_size=3,
        num_layers=args.num_layers
    )
    
    # Build router with dummy input
    dummy_input = np.zeros((1, Ny, Nx, 14), dtype=np.float32)
    _ = router(dummy_input)
    print(f"  Router parameters: {router.count_params():,}")
    
    # Feature extractor
    feature_extractor = RouterFeatureExtractor(
        x_domain=x_domain,
        y_domain=y_domain,
        cylinder_center=cylinder_center,
        cylinder_radius=cylinder_radius,
        inlet_velocity=args.inlet_velocity,
        dx=dx,
        dy=dy
    )
    
    # Physics loss
    physics_loss = PhysicsLoss(
        nu=nu,
        rho=rho,
        dx=dx,
        dy=dy,
        lambda_BC=args.lambda_BC,
        lambda_spatial=args.lambda_spatial,
        lambda_cost=args.lambda_cost,
        x_domain=x_domain,
        y_domain=y_domain,
        inlet_velocity=args.inlet_velocity
    )
    
    # CFD solver for training
    cfd_solver = create_router_cfd_solver(
        CylinderFlowSimulation,
        Re=args.Re,
        N=args.grid_size,
        max_iter=args.cfd_max_iter,
        tol=args.cfd_tol,
        x_domain=x_domain,
        y_domain=y_domain,
        cylinder_center=cylinder_center,
        cylinder_radius=cylinder_radius,
        inlet_velocity=args.inlet_velocity
    )
    
    # =========================================================================
    # Step 4: Create trainer and train router
    # =========================================================================
    print("\nStep 4: Training router...")
    
    trainer = HybridRouterTrainer(
        router=router,
        pinn_network=pinn_network,
        cfd_solver=cfd_solver,
        feature_extractor=feature_extractor,
        physics_loss=physics_loss,
        learning_rate=args.lr,
        threshold=args.threshold
    )
    
    start_time = time.time()
    history = trainer.train(
        X=X,
        Y=Y,
        xy=xy,
        num_epochs=args.epochs,
        verbose=True,
        cylinder_mask=cylinder_mask
    )
    train_time = time.time() - start_time
    print(f"\nTraining time: {train_time:.2f}s")
    
    # =========================================================================
    # Step 5: Get final mask and hybrid solution
    # =========================================================================
    print("\nStep 5: Computing final hybrid solution...")
    
    # Get PINN predictions
    uvp_pinn = pinn_network.predict(xy, batch_size=len(xy))
    u_pinn = uvp_pinn[..., 0].reshape(Ny, Nx)
    v_pinn = uvp_pinn[..., 1].reshape(Ny, Nx)
    p_pinn = uvp_pinn[..., 2].reshape(Ny, Nx)
    
    # Apply cylinder mask
    u_pinn = np.where(cylinder_mask == 1, 0.0, u_pinn)
    v_pinn = np.where(cylinder_mask == 1, 0.0, v_pinn)
    
    # Get final router mask
    r_binary, r_soft = trainer.get_final_mask(X, Y, u_pinn, v_pinn, p_pinn, cylinder_mask)
    
    # Compute final hybrid solution
    if r_binary.sum() > 0:
        u_hybrid_cfd, v_hybrid_cfd, p_hybrid_cfd = cfd_solver(
            cfd_mask=r_binary.astype(bool),
            u_pinn=u_pinn,
            v_pinn=v_pinn,
            p_pinn=p_pinn,
            cylinder_mask=cylinder_mask
        )
        
        # Blend using soft weights for smooth transition
        u_hybrid = (1 - r_soft) * u_pinn + r_soft * u_hybrid_cfd
        v_hybrid = (1 - r_soft) * v_pinn + r_soft * v_hybrid_cfd
        p_hybrid = (1 - r_soft) * p_pinn + r_soft * p_hybrid_cfd
    else:
        u_hybrid = u_pinn.copy()
        v_hybrid = v_pinn.copy()
        p_hybrid = p_pinn.copy()
    
    # =========================================================================
    # Step 6: Visualize results
    # =========================================================================
    print("\nStep 6: Generating visualizations...")
    
    # Plot router mask
    plot_router_mask(
        r_soft, r_binary, X, Y, cylinder_center, cylinder_radius,
        title=f"Router Learned Mask (Re={args.Re}, {args.epochs} epochs)",
        save_path=os.path.join(args.output_dir, 'router_mask.png')
    )
    
    # Plot training history
    plot_training_history(
        history,
        save_path=os.path.join(args.output_dir, 'training_history.png')
    )
    
    # Plot solution comparison
    plot_solution_comparison(
        u_pinn, v_pinn, u_hybrid, v_hybrid, u_cfd, v_cfd,
        X, Y, cylinder_center, cylinder_radius,
        save_path=os.path.join(args.output_dir, 'solution_comparison.png')
    )
    
    # =========================================================================
    # Step 7: Save results
    # =========================================================================
    print("\nStep 7: Saving results...")
    
    # Save router weights
    router_weights_path = os.path.join(args.output_dir, 'router_weights.h5')
    router.save_weights(router_weights_path)
    print(f"  Saved router weights to {router_weights_path}")
    
    # Save mask
    mask_path = os.path.join(args.output_dir, 'learned_mask.npy')
    np.save(mask_path, {
        'r_soft': r_soft,
        'r_binary': r_binary,
        'X': X,
        'Y': Y
    })
    print(f"  Saved mask data to {mask_path}")
    
    # Save training history
    history_path = os.path.join(args.output_dir, 'training_history.npy')
    np.save(history_path, history)
    print(f"  Saved training history to {history_path}")
    
    # Save configuration
    config_path = os.path.join(args.output_dir, 'config.txt')
    with open(config_path, 'w') as f:
        f.write("Router Training Configuration\n")
        f.write("="*50 + "\n")
        f.write(f"Reynolds number: {args.Re}\n")
        f.write(f"Grid size: {Nx}x{Ny}\n")
        f.write(f"Training epochs: {args.epochs}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"λ_BC: {args.lambda_BC}\n")
        f.write(f"λ_spatial: {args.lambda_spatial}\n")
        f.write(f"λ_cost: {args.lambda_cost}\n")
        f.write(f"Threshold: {args.threshold}\n")
        f.write(f"\nResults:\n")
        f.write(f"Final CFD fraction: {100*np.mean(r_binary):.1f}%\n")
        f.write(f"Training time: {train_time:.2f}s\n")
        
        # Compute final errors
        vel_pinn = np.sqrt(u_pinn**2 + v_pinn**2)
        vel_hybrid = np.sqrt(u_hybrid**2 + v_hybrid**2)
        vel_cfd = np.sqrt(u_cfd**2 + v_cfd**2)
        
        rmse_pinn = np.sqrt(np.mean((vel_pinn - vel_cfd)**2))
        rmse_hybrid = np.sqrt(np.mean((vel_hybrid - vel_cfd)**2))
        
        f.write(f"PINN RMSE: {rmse_pinn:.6f}\n")
        f.write(f"Hybrid RMSE: {rmse_hybrid:.6f}\n")
        f.write(f"Improvement: {100*(rmse_pinn-rmse_hybrid)/rmse_pinn:.1f}%\n")
    
    print(f"  Saved configuration to {config_path}")
    
    print("\n" + "="*70)
    print("ROUTER TRAINING COMPLETE")
    print("="*70)
    print(f"\nFinal Statistics:")
    print(f"  CFD region: {100*np.mean(r_binary):.1f}%")
    print(f"  PINN region: {100*(1-np.mean(r_binary)):.1f}%")
    print(f"  Inlet rejection rate: {100*history['inlet_rejection_rate'][-1]:.1f}%")
    print(f"\nOutput saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
