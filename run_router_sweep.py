#!/usr/bin/env python3
"""
Run router training and simulation for multiple beta values.

This script automates sweeping over different beta (CFD cost) values,
training routers, and generating visualizations for each configuration.

Usage:
    # Train and run for specific beta values
    python run_router_sweep.py --betas 0.01 0.05 0.1 0.2 0.5
    
    # Train only (no simulation)
    python run_router_sweep.py --betas 0.01 0.1 1.0 --train-only
    
    # Run simulation only (for already trained routers)
    python run_router_sweep.py --betas 0.01 0.1 1.0 --simulate-only
    
    # Specify custom output directories
    python run_router_sweep.py --betas 0.1 --output-base-dir ./experiments/router
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime


def format_beta_str(beta):
    """Format beta value for directory naming."""
    return f"beta_{beta:.4f}".rstrip('0').rstrip('.')


def run_training(beta, args):
    """Run router training for a specific beta value."""
    beta_str = format_beta_str(beta)
    output_dir = os.path.join(args.output_base_dir, beta_str)
    
    cmd = [
        sys.executable, 'train_router.py',
        '--beta', str(beta),
        '--output-dir', output_dir,
        '--epochs', str(args.epochs),
        '--lambda-tv', str(args.lambda_tv),
        '--lambda-entropy', str(args.lambda_entropy),
        '--lambda-variance', str(args.lambda_variance),
        '--lr', str(args.lr),
        '--temperature', str(args.temperature),
        '--model-path', args.model_path,
        '--threshold', str(args.threshold),
    ]
    
    if args.grad_clip > 0:
        cmd.extend(['--grad-clip', str(args.grad_clip)])
    
    print(f"\n{'=' * 60}")
    print(f"TRAINING ROUTER: beta = {beta}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}\n")
    
    result = subprocess.run(cmd, cwd=args.working_dir)
    return result.returncode == 0


def run_simulation(beta, args):
    """Run hybrid simulation for a specific beta value."""
    beta_str = format_beta_str(beta)
    router_path = os.path.join(args.output_base_dir, beta_str, 'router.weights.h5')
    output_dir = os.path.join(args.hybrid_output_base_dir, beta_str)
    
    if not os.path.exists(router_path):
        print(f"WARNING: Router weights not found at {router_path}")
        print(f"  Skipping simulation for beta = {beta}")
        return False
    
    cmd = [
        sys.executable, 'run_router_hybrid.py',
        '--beta', str(beta),
        '--router-path', router_path,
        '--output-dir', output_dir,
        '--pinn-path', args.model_path,
        '--threshold', str(args.threshold),
        '--temperature', str(args.temperature),
        '--Re', str(args.Re),
        '--max-iter', str(args.max_iter),
    ]
    
    print(f"\n{'=' * 60}")
    print(f"RUNNING SIMULATION: beta = {beta}")
    print(f"Router: {router_path}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}\n")
    
    result = subprocess.run(cmd, cwd=args.working_dir)
    return result.returncode == 0


def generate_comparison_plot(betas, args):
    """Generate a comparison plot across all beta values."""
    import numpy as np
    import matplotlib.pyplot as plt
    
    output_path = os.path.join(args.output_base_dir, 'beta_comparison.png')
    
    fig, axes = plt.subplots(2, len(betas), figsize=(4 * len(betas), 8))
    if len(betas) == 1:
        axes = axes.reshape(2, 1)
    
    for i, beta in enumerate(betas):
        beta_str = format_beta_str(beta)
        
        # Load router predictions
        pred_path = os.path.join(args.output_base_dir, beta_str, 'predictions.npz')
        if os.path.exists(pred_path):
            data = np.load(pred_path)
            router_output = data['router_output']
            mask = data['mask']
            X = data['X']
            Y = data['Y']
            layout = data['layout']
            
            # Plot router output
            ax1 = axes[0, i]
            im1 = ax1.imshow(router_output.T, origin='lower', 
                           extent=[X.min(), X.max(), Y.min(), Y.max()],
                           cmap='RdBu_r', vmin=0, vmax=1)
            ax1.set_title(f'β = {beta}')
            ax1.set_xlabel('x')
            if i == 0:
                ax1.set_ylabel('Router Output')
            plt.colorbar(im1, ax=ax1)
            
            # Calculate CFD fraction
            cfd_frac = np.sum(mask * layout) / np.sum(layout) * 100
            
            # Plot mask
            ax2 = axes[1, i]
            im2 = ax2.imshow(mask.T, origin='lower',
                           extent=[X.min(), X.max(), Y.min(), Y.max()],
                           cmap='RdBu_r', vmin=0, vmax=1)
            ax2.set_title(f'CFD: {cfd_frac:.1f}%')
            ax2.set_xlabel('x')
            if i == 0:
                ax2.set_ylabel('Binary Mask')
            plt.colorbar(im2, ax=ax2)
        else:
            print(f"Warning: No predictions found for beta = {beta}")
    
    plt.suptitle('Router Output vs Beta (CFD Cost)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved comparison plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run router training and simulation for multiple beta values'
    )
    
    # Beta values
    parser.add_argument('--betas', type=float, nargs='+', 
                        default=[0.01, 0.05, 0.1, 0.2, 0.5],
                        help='List of beta values to sweep')
    
    # Mode
    parser.add_argument('--train-only', action='store_true',
                        help='Only train routers, skip simulation')
    parser.add_argument('--simulate-only', action='store_true',
                        help='Only run simulations (requires pre-trained routers)')
    parser.add_argument('--skip-comparison', action='store_true',
                        help='Skip generating comparison plot')
    
    # Output directories
    parser.add_argument('--output-base-dir', type=str, default='./router_output',
                        help='Base directory for router training outputs')
    parser.add_argument('--hybrid-output-base-dir', type=str, default='./router_hybrid_output',
                        help='Base directory for simulation outputs')
    parser.add_argument('--working-dir', type=str, default='.',
                        help='Working directory for running scripts')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--lambda-tv', type=float, default=0.01,
                        help='Total variation regularization weight')
    parser.add_argument('--lambda-entropy', type=float, default=0.1,
                        help='Entropy regularization weight')
    parser.add_argument('--lambda-variance', type=float, default=0.05,
                        help='Variance regularization weight')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Sigmoid temperature')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping norm (0 to disable)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary mask')
    
    # Model path
    parser.add_argument('--model-path', type=str, 
                        default='./models/pinn_cylinder_100.0.h5',
                        help='Path to pre-trained PINN model')
    
    # Simulation parameters
    parser.add_argument('--Re', type=float, default=100,
                        help='Reynolds number')
    parser.add_argument('--max-iter', type=int, default=100000,
                        help='Maximum CFD iterations')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.train_only and args.simulate_only:
        print("Error: Cannot specify both --train-only and --simulate-only")
        sys.exit(1)
    
    print("=" * 60)
    print("ROUTER BETA SWEEP")
    print("=" * 60)
    print(f"Beta values: {args.betas}")
    print(f"Training output: {args.output_base_dir}")
    print(f"Simulation output: {args.hybrid_output_base_dir}")
    print(f"Mode: {'Train only' if args.train_only else 'Simulate only' if args.simulate_only else 'Train + Simulate'}")
    
    start_time = datetime.now()
    
    # Run training
    if not args.simulate_only:
        print("\n" + "=" * 60)
        print("PHASE 1: TRAINING ROUTERS")
        print("=" * 60)
        
        for beta in args.betas:
            success = run_training(beta, args)
            if not success:
                print(f"WARNING: Training failed for beta = {beta}")
    
    # Run simulations
    if not args.train_only:
        print("\n" + "=" * 60)
        print("PHASE 2: RUNNING SIMULATIONS")
        print("=" * 60)
        
        for beta in args.betas:
            success = run_simulation(beta, args)
            if not success:
                print(f"WARNING: Simulation failed for beta = {beta}")
    
    # Generate comparison plot
    if not args.skip_comparison and not args.simulate_only:
        print("\n" + "=" * 60)
        print("PHASE 3: GENERATING COMPARISON PLOT")
        print("=" * 60)
        
        try:
            generate_comparison_plot(args.betas, args)
        except Exception as e:
            print(f"Warning: Could not generate comparison plot: {e}")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "=" * 60)
    print("SWEEP COMPLETE")
    print("=" * 60)
    print(f"Total time: {elapsed:.1f} seconds")
    print(f"\nOutputs saved to:")
    for beta in args.betas:
        beta_str = format_beta_str(beta)
        print(f"  {beta_str}/")
        print(f"    Router: {os.path.join(args.output_base_dir, beta_str)}")
        if not args.train_only:
            print(f"    Simulation: {os.path.join(args.hybrid_output_base_dir, beta_str)}")


if __name__ == "__main__":
    main()
