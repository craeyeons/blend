"""
Run hybrid PINN-CFD simulation using trained router.

This script:
1. Loads a trained router model
2. Generates the domain segmentation mask
3. Runs the hybrid simulation combining PINN and CFD solutions
4. Visualizes and saves results

Usage:
    python run_router_hybrid.py --router-path ./router_output/router_weights.h5
"""

import argparse
import os
import numpy as np
import tensorflow as tf

# Configure TensorFlow GPU memory growth to avoid cuDNN issues
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU memory growth setting failed: {e}")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Import components
from lib.router import (
    RouterCNN,
    create_router_input,
    create_cylinder_setup,
    plot_router_output
)
from lib.cylinder_flow import CylinderFlowHybridSimulation, CylinderFlowSimulation
from lib.plotting import plot_solution, plot_hybrid_solution, plot_streamlines
from cylinder_network import Network as CylinderNetwork


def compute_uv_direct(network, xy):
    """Extract (u, v) directly from network output (u, v, p)."""
    uvp = network.predict(xy, batch_size=len(xy), verbose=0)
    u = uvp[..., 0]
    v = uvp[..., 1]
    return u, v


def main():
    parser = argparse.ArgumentParser(
        description='Run hybrid PINN-CFD simulation with trained router'
    )
    
    # Model paths
    parser.add_argument('--router-path', type=str,
                        default='./router_output/router_weights.h5',
                        help='Path to trained router weights')
    parser.add_argument('--pinn-path', type=str,
                        default='./models/pinn_cylinder_100.0.h5',
                        help='Path to pre-trained PINN model')
    parser.add_argument('--output-dir', type=str, default='./router_hybrid_output',
                        help='Directory to save outputs')
    
    # Domain parameters
    parser.add_argument('--nx', type=int, default=200,
                        help='Grid points in x direction')
    parser.add_argument('--ny', type=int, default=100,
                        help='Grid points in y direction')
    parser.add_argument('--x-min', type=float, default=0.0)
    parser.add_argument('--x-max', type=float, default=2.0)
    parser.add_argument('--y-min', type=float, default=0.0)
    parser.add_argument('--y-max', type=float, default=1.0)
    parser.add_argument('--cylinder-x', type=float, default=0.5)
    parser.add_argument('--cylinder-y', type=float, default=0.5)
    parser.add_argument('--cylinder-radius', type=float, default=0.1)
    parser.add_argument('--inlet-velocity', type=float, default=1.0)
    
    # Physical parameters
    parser.add_argument('--Re', type=float, default=100,
                        help='Reynolds number')
    
    # CFD parameters
    parser.add_argument('--max-iter', type=int, default=100000,
                        help='Maximum CFD iterations')
    parser.add_argument('--tol', type=float, default=1e-6,
                        help='Convergence tolerance')
    
    # Router
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary mask')
    parser.add_argument('--base-filters', type=int, default=32,
                        help='Base filters in router CNN')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("HYBRID PINN-CFD SIMULATION WITH TRAINED ROUTER")
    print("=" * 60)
    
    # =========================================================================
    # Step 1: Load models
    # =========================================================================
    print("\n[Step 1] Loading models...")
    
    # Load PINN model
    network = CylinderNetwork()
    pinn_model = network.build(
        num_inputs=2, 
        layers=[48, 48, 48, 48], 
        activation='tanh', 
        num_outputs=3
    )
    pinn_model.load_weights(args.pinn_path)
    print(f"  ✓ Loaded PINN model from {args.pinn_path}")
    
    # Create domain setup
    X, Y, layout, bc_mask, bc_u, bc_v, bc_p = create_cylinder_setup(
        Nx=args.nx,
        Ny=args.ny,
        x_domain=(args.x_min, args.x_max),
        y_domain=(args.y_min, args.y_max),
        cylinder_center=(args.cylinder_x, args.cylinder_y),
        cylinder_radius=args.cylinder_radius,
        inlet_velocity=args.inlet_velocity
    )
    
    # Create router input
    inputs = create_router_input(layout, bc_mask, bc_u, bc_v, bc_p)
    
    # Load router
    router = RouterCNN(base_filters=args.base_filters)
    _ = router(inputs)  # Build model
    router.load_weights(args.router_path)
    print(f"  ✓ Loaded router from {args.router_path}")
    
    # =========================================================================
    # Step 2: Generate mask from router
    # =========================================================================
    print("\n[Step 2] Generating domain mask...")
    
    r = router(tf.constant(inputs, dtype=tf.float32), training=False)
    r = r[0, :, :, 0].numpy()
    
    # Apply threshold to get binary mask
    # mask = 1 means CFD, mask = 0 means PINN
    mask = (r >= args.threshold).astype(np.int32)
    
    # Apply layout mask (obstacle = 0)
    mask = mask * layout.astype(np.int32)
    
    cfd_fraction = np.sum(mask) / np.sum(layout) * 100
    print(f"  Router output range: [{r.min():.4f}, {r.max():.4f}]")
    print(f"  CFD region: {cfd_fraction:.1f}%")
    print(f"  PINN region: {100 - cfd_fraction:.1f}%")
    
    # Visualize router output
    router_plot_path = os.path.join(args.output_dir, 'router_mask.png')
    plot_router_output(
        r, X, Y, layout,
        title='Router-Generated Domain Segmentation',
        save_path=router_plot_path,
        show_circle=(args.cylinder_x, args.cylinder_y, args.cylinder_radius)
    )
    
    # =========================================================================
    # Step 3: Run hybrid simulation
    # =========================================================================
    print("\n[Step 3] Running hybrid simulation...")
    
    # Create hybrid simulation with router-generated mask
    sim = CylinderFlowHybridSimulation(
        network=pinn_model,
        uv_func=compute_uv_direct,
        mask=mask,
        Re=args.Re,
        N=args.ny,  # Base resolution
        max_iter=args.max_iter,
        tol=args.tol,
        x_domain=(args.x_min, args.x_max),
        y_domain=(args.y_min, args.y_max),
        cylinder_center=(args.cylinder_x, args.cylinder_y),
        cylinder_radius=args.cylinder_radius,
        inlet_velocity=args.inlet_velocity
    )
    
    # Solve
    u, v, p = sim.solve()
    
    # Convert to numpy if needed
    u = np.array(u)
    v = np.array(v)
    p = np.array(p)
    
    print(f"\n  Solution statistics:")
    print(f"    u: [{u.min():.4f}, {u.max():.4f}]")
    print(f"    v: [{v.min():.4f}, {v.max():.4f}]")
    print(f"    p: [{p.min():.4f}, {p.max():.4f}]")
    
    # =========================================================================
    # Step 4: Save and visualize results
    # =========================================================================
    print("\n[Step 4] Saving results...")
    
    # Save solution
    solution_path = os.path.join(args.output_dir, 'solution.npz')
    np.savez(solution_path,
             u=u, v=v, p=p,
             X=X, Y=Y,
             mask=mask,
             router_output=r)
    print(f"  ✓ Saved solution to {solution_path}")
    
    # Plot solution
    print("\n[Step 5] Generating visualizations...")
    
    # Full solution
    solution_path = os.path.join(args.output_dir, 'solution.png')
    plot_solution(
        u, v, p, x=X, y=Y,
        title_prefix='Router Hybrid',
        save_path=solution_path,
        show_circle=(args.cylinder_x, args.cylinder_y, args.cylinder_radius),
        simulation_type='cylinder',
        mode='router_hybrid',
        Re=args.Re
    )
    
    # Hybrid solution with mask overlay
    hybrid_path = os.path.join(args.output_dir, 'hybrid_solution.png')
    plot_hybrid_solution(
        u, v, p, mask, x=X, y=Y,
        save_path=hybrid_path,
        show_circle=(args.cylinder_x, args.cylinder_y, args.cylinder_radius),
        simulation_type='cylinder',
        mode='router_hybrid',
        Re=args.Re
    )
    
    # Streamlines
    streamlines_path = os.path.join(args.output_dir, 'streamlines.png')
    plot_streamlines(
        u, v, x=X, y=Y,
        density=2.0,
        save_path=streamlines_path,
        show_circle=(args.cylinder_x, args.cylinder_y, args.cylinder_radius),
        title='Streamlines (Router Hybrid)',
        simulation_type='cylinder',
        mode='router_hybrid',
        Re=args.Re
    )
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {args.output_dir}/")
    
    return u, v, p, mask


if __name__ == "__main__":
    main()
