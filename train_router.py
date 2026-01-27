"""
Train the CNN-based router for hybrid PINN-CFD simulations.

This script trains a router to decide which regions should use PINN vs CFD.
The router learns to minimize CFD usage while maintaining PINN accuracy.

Usage:
    python train_router.py --model-path ./models/pinn_cylinder_100.0.h5 --epochs 200
    python train_router.py --beta 0.05 --lambda-tv 0.02 --epochs 500
"""

import argparse
import os
import numpy as np
import tensorflow as tf
from datetime import datetime

# Import router components
from lib.router import (
    RouterCNN,
    RouterTrainer,
    create_router_input,
    create_cylinder_setup,
    plot_router_output,
    plot_training_history
)
from cylinder_network import Network as CylinderNetwork


def main():
    parser = argparse.ArgumentParser(
        description='Train CNN router for hybrid PINN-CFD simulations'
    )
    
    # Model paths
    parser.add_argument('--model-path', type=str, 
                        default='./models/pinn_cylinder_100.0.h5',
                        help='Path to pre-trained PINN model')
    parser.add_argument('--output-dir', type=str, default='./router_output',
                        help='Directory to save outputs')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='CFD cost coefficient (higher = less CFD)')
    parser.add_argument('--lambda-tv', type=float, default=0.01,
                        help='Total variation regularization weight')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    
    # Residual weights
    parser.add_argument('--weight-continuity', type=float, default=1.0,
                        help='Weight for continuity residual')
    parser.add_argument('--weight-momentum', type=float, default=1.0,
                        help='Weight for momentum residual')
    
    # Domain parameters (matching cylinder flow setup)
    parser.add_argument('--nx', type=int, default=200,
                        help='Grid points in x direction')
    parser.add_argument('--ny', type=int, default=100,
                        help='Grid points in y direction')
    parser.add_argument('--x-min', type=float, default=0.0,
                        help='Domain x minimum')
    parser.add_argument('--x-max', type=float, default=2.0,
                        help='Domain x maximum')
    parser.add_argument('--y-min', type=float, default=0.0,
                        help='Domain y minimum')
    parser.add_argument('--y-max', type=float, default=1.0,
                        help='Domain y maximum')
    parser.add_argument('--cylinder-x', type=float, default=0.5,
                        help='Cylinder center x')
    parser.add_argument('--cylinder-y', type=float, default=0.5,
                        help='Cylinder center y')
    parser.add_argument('--cylinder-radius', type=float, default=0.1,
                        help='Cylinder radius')
    parser.add_argument('--inlet-velocity', type=float, default=1.0,
                        help='Inlet velocity')
    
    # Physical parameters
    parser.add_argument('--nu', type=float, default=0.01,
                        help='Kinematic viscosity')
    parser.add_argument('--rho', type=float, default=1.0,
                        help='Fluid density')
    
    # Router architecture
    parser.add_argument('--base-filters', type=int, default=32,
                        help='Base number of filters in router CNN')
    
    # Inference
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary mask')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("CNN ROUTER TRAINING FOR HYBRID PINN-CFD")
    print("=" * 60)
    
    # =========================================================================
    # Step 1: Load PINN model
    # =========================================================================
    print("\n[Step 1] Loading PINN model...")
    print(f"  Model path: {args.model_path}")
    
    # Build network architecture matching training
    network = CylinderNetwork()
    pinn_model = network.build(
        num_inputs=2, 
        layers=[48, 48, 48, 48], 
        activation='tanh', 
        num_outputs=3
    )
    
    try:
        pinn_model.load_weights(args.model_path)
        print("  ✓ PINN model loaded successfully")
    except Exception as e:
        print(f"  ✗ Failed to load PINN model: {e}")
        return
    
    # Test PINN model
    test_xy = np.array([[0.5, 0.5], [1.0, 0.5]])
    test_output = pinn_model.predict(test_xy, verbose=0)
    print(f"  Test output shape: {test_output.shape}")
    print(f"  Test output: u={test_output[0,0]:.4f}, v={test_output[0,1]:.4f}, p={test_output[0,2]:.4f}")
    
    # =========================================================================
    # Step 2: Create domain setup
    # =========================================================================
    print("\n[Step 2] Creating domain setup...")
    
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
    print(f"  Domain: x=[{args.x_min}, {args.x_max}], y=[{args.y_min}, {args.y_max}]")
    print(f"  Cylinder: center=({args.cylinder_x}, {args.cylinder_y}), r={args.cylinder_radius}")
    print(f"  Fluid points: {np.sum(layout):.0f} / {layout.size} ({100*np.mean(layout):.1f}%)")
    print(f"  BC points: {np.sum(bc_mask):.0f}")
    
    # Create router input tensor
    inputs = create_router_input(layout, bc_mask, bc_u, bc_v, bc_p)
    print(f"  Router input shape: {inputs.shape}")
    
    # =========================================================================
    # Step 3: Initialize router
    # =========================================================================
    print("\n[Step 3] Initializing router CNN...")
    
    router = RouterCNN(base_filters=args.base_filters)
    
    # Build the model by running a forward pass
    _ = router(inputs)
    print(f"  Router parameters: {router.count_params():,}")
    
    # =========================================================================
    # Step 4: Initialize trainer
    # =========================================================================
    print("\n[Step 4] Initializing trainer...")
    
    residual_weights = {
        'continuity': args.weight_continuity,
        'momentum': args.weight_momentum
    }
    
    trainer = RouterTrainer(
        router=router,
        pinn_model=pinn_model,
        beta=args.beta,
        lambda_tv=args.lambda_tv,
        residual_weights=residual_weights,
        nu=args.nu,
        rho=args.rho
    )
    trainer.optimizer.learning_rate.assign(args.lr)
    
    print(f"  β (CFD cost): {args.beta}")
    print(f"  λ (TV reg): {args.lambda_tv}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Residual weights: {residual_weights}")
    
    # =========================================================================
    # Step 5: Train router
    # =========================================================================
    print("\n[Step 5] Training router...")
    print("-" * 50)
    
    start_time = datetime.now()
    
    history = trainer.train(
        inputs=inputs,
        X=X,
        Y=Y,
        layout_mask=layout,
        epochs=args.epochs,
        verbose=True
    )
    
    training_time = (datetime.now() - start_time).total_seconds()
    print("-" * 50)
    print(f"  Training completed in {training_time:.1f} seconds")
    
    # =========================================================================
    # Step 6: Get final predictions
    # =========================================================================
    print("\n[Step 6] Generating predictions...")
    
    r, mask = trainer.predict(inputs, threshold=args.threshold)
    
    cfd_fraction = np.sum(mask * layout) / np.sum(layout) * 100
    pinn_fraction = 100 - cfd_fraction
    
    print(f"  CFD region: {cfd_fraction:.1f}%")
    print(f"  PINN region: {pinn_fraction:.1f}%")
    
    # =========================================================================
    # Step 7: Save results
    # =========================================================================
    print("\n[Step 7] Saving results...")
    
    # Save router weights
    router_path = os.path.join(args.output_dir, 'router_weights.h5')
    router.save_weights(router_path)
    print(f"  ✓ Saved router weights to {router_path}")
    
    # Save training history
    history_path = os.path.join(args.output_dir, 'training_history.npz')
    np.savez(history_path, **history)
    print(f"  ✓ Saved training history to {history_path}")
    
    # Save predictions
    predictions_path = os.path.join(args.output_dir, 'predictions.npz')
    np.savez(predictions_path, 
             router_output=r, 
             mask=mask,
             X=X, Y=Y, 
             layout=layout)
    print(f"  ✓ Saved predictions to {predictions_path}")
    
    # Save configuration
    config_path = os.path.join(args.output_dir, 'config.txt')
    with open(config_path, 'w') as f:
        f.write("# Router Training Configuration\n")
        f.write(f"python train_router.py \\\n")
        f.write(f"    --model-path {args.model_path} \\\n")
        f.write(f"    --epochs {args.epochs} \\\n")
        f.write(f"    --beta {args.beta} \\\n")
        f.write(f"    --lambda-tv {args.lambda_tv} \\\n")
        f.write(f"    --lr {args.lr} \\\n")
        f.write(f"    --weight-continuity {args.weight_continuity} \\\n")
        f.write(f"    --weight-momentum {args.weight_momentum} \\\n")
        f.write(f"    --nx {args.nx} \\\n")
        f.write(f"    --ny {args.ny} \\\n")
        f.write(f"    --threshold {args.threshold}\n")
        f.write(f"\n# Results:\n")
        f.write(f"# CFD region: {cfd_fraction:.1f}%\n")
        f.write(f"# PINN region: {pinn_fraction:.1f}%\n")
        f.write(f"# Training time: {training_time:.1f}s\n")
    print(f"  ✓ Saved config to {config_path}")
    
    # =========================================================================
    # Step 8: Visualize results
    # =========================================================================
    print("\n[Step 8] Generating visualizations...")
    
    # Plot router output
    router_plot_path = os.path.join(args.output_dir, 'router_output.png')
    plot_router_output(
        r, X, Y, layout,
        title=f'Trained Router (β={args.beta}, λ={args.lambda_tv})',
        save_path=router_plot_path,
        show_circle=(args.cylinder_x, args.cylinder_y, args.cylinder_radius)
    )
    
    # Plot training history
    history_plot_path = os.path.join(args.output_dir, 'training_history.png')
    plot_training_history(history, save_path=history_plot_path)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"  - router_weights.h5: Trained router model")
    print(f"  - predictions.npz: Router output and mask")
    print(f"  - training_history.npz: Loss history")
    print(f"  - router_output.png: Visualization")
    print(f"  - training_history.png: Loss curves")
    
    return router, trainer, history


if __name__ == "__main__":
    main()
