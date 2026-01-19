"""
Main entry point for fluid dynamics simulations.

Supports multiple simulation types:
- cavity: Lid-driven cavity flow
- cylinder: Flow around a cylinder

Usage:
    python main.py --simulation cavity --mode hybrid
    python main.py --simulation cylinder --mode cfd
"""

import argparse
import time
import numpy as np
import tensorflow as tf

from lib.network import Network
from lib.plotting import (
    plot_solution, 
    plot_hybrid_solution, 
    plot_streamlines,
    plot_comparison,
    generate_filename
)


def compute_uv_from_psi(network, xy):
    """
    Compute flow velocities (u, v) from stream function.
    
    For network with output (psi, p):
    u = ∂ψ/∂y
    v = -∂ψ/∂x
    
    Parameters:
    -----------
    network : tf.keras.Model
        Network that outputs (psi, p)
    xy : ndarray
        Input coordinates
    
    Returns:
    --------
    u, v : ndarray
        Velocity components
    """
    xy_tf = tf.constant(xy, dtype=tf.float32)
    with tf.GradientTape() as g:
        g.watch(xy_tf)
        psi_p = network(xy_tf)
    psi_p_j = g.batch_jacobian(psi_p, xy_tf)
    u = psi_p_j[..., 0, 1]   # ∂ψ/∂y
    v = -psi_p_j[..., 0, 0]  # -∂ψ/∂x
    return u.numpy(), v.numpy()


def compute_uv_direct(network, xy):
    """
    Extract flow velocities (u, v) directly from network output.
    
    For network with output (u, v, p):
    u and v are directly available.
    
    Parameters:
    -----------
    network : tf.keras.Model
        Network that outputs (u, v, p)
    xy : ndarray
        Input coordinates
    
    Returns:
    --------
    u, v : ndarray
        Velocity components
    """
    uvp = network.predict(xy, batch_size=len(xy))
    u = uvp[..., 0]
    v = uvp[..., 1]
    return u, v


def run_cavity_simulation(args):
    """Run lid-driven cavity flow simulation."""
    from lib.cavity_flow import (
        CavityFlowSimulation,
        CavityFlowHybridSimulation,
        CavityFlowDynamicHybridSimulation,
        create_center_pinn_mask,
        create_boundary_pinn_mask
    )
    
    print("=" * 60)
    print("LID-DRIVEN CAVITY FLOW SIMULATION")
    print("=" * 60)
    
    # Physical parameters
    u0 = 1.0
    L = 1.0
    nu = 0.01
    rho = 1.0
    Re = rho * u0 * L / nu
    
    print(f"\nPhysical parameters:")
    print(f"  Reynolds number: {Re}")
    print(f"  Grid size: {args.grid_size}x{args.grid_size}")
    print(f"  Mode: {args.mode}")
    print()
    
    if args.mode == 'cfd':
        # Pure CFD simulation
        sim = CavityFlowSimulation(
            Re=Re,
            N=args.grid_size,
            max_iter=args.max_iter,
            tol=args.tolerance
        )
        
        start_time = time.time()
        u, v, p = sim.solve()
        elapsed = time.time() - start_time
        
        print(f"\nSimulation time: {elapsed:.2f} seconds")
        
        # Plot results
        plot_solution(u, v, p, title_prefix='CFD: ', 
                     save_path='cavity_flow_cfd.png')
        plot_streamlines(u, v, save_path='cavity_flow_streamlines.png',
                        title='Cavity Flow Streamlines')
        
    elif args.mode == 'hybrid':
        # Hybrid PINN-CFD simulation
        network = Network().build()
        
        if args.model_path:
            print(f"Loading PINN model from {args.model_path}")
            network.load_weights(args.model_path)
        else:
            print("Warning: No model path provided. Using untrained network.")
        
        # Create mask
        border_width = int(args.grid_size * args.border_fraction)
        mask = create_center_pinn_mask(args.grid_size, border_width)
        
        sim = CavityFlowHybridSimulation(
            network=network,
            uv_func=compute_uv_from_psi,
            mask=mask,
            Re=Re,
            N=args.grid_size,
            max_iter=args.max_iter,
            tol=args.tolerance
        )
        
        start_time = time.time()
        u, v, p = sim.solve()
        elapsed = time.time() - start_time
        
        print(f"\nSimulation time: {elapsed:.2f} seconds")
        
        # Plot results
        plot_solution(u, v, p, title_prefix='Hybrid: ',
                     save_path='cavity_flow_hybrid.png')
        plot_hybrid_solution(u, v, p, mask, 
                            save_path='cavity_flow_hybrid_mask.png')
        plot_streamlines(u, v, save_path='cavity_flow_hybrid_streamlines.png',
                        title='Hybrid PINN-CFD Streamlines')
        
    elif args.mode == 'pinn':
        # PINN-only solution (using pre-trained model)
        network = Network().build()
        
        if args.model_path:
            print(f"Loading PINN model from {args.model_path}")
            network.load_weights(args.model_path)
        else:
            raise ValueError("Model path required for PINN mode")
        
        # Create grid
        N = args.grid_size
        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, N)
        X, Y = np.meshgrid(x, y)
        xy = np.stack([X.flatten(), Y.flatten()], axis=-1)
        
        # Predict
        start_time = time.time()
        psi_p = network.predict(xy, batch_size=len(xy))
        u_pinn, v_pinn = compute_uv_from_psi(network, xy)
        elapsed = time.time() - start_time
        
        u = u_pinn.reshape(X.shape)
        v = v_pinn.reshape(X.shape)
        p = psi_p[..., 1].reshape(X.shape)
        
        print(f"\nPrediction time: {elapsed:.2f} seconds")
        
        # Plot results
        plot_solution(u, v, p, x=X, y=Y, title_prefix='PINN: ',
                     save_path='cavity_flow_pinn.png')
    
    elif args.mode == 'dynamic':
        # Dynamic hybrid PINN-CFD simulation with complexity-based segregation
        network = Network().build()
        
        if args.model_path:
            print(f"Loading PINN model from {args.model_path}")
            network.load_weights(args.model_path)
        else:
            print("Warning: No model path provided. Using untrained network.")
        
        # Create initial flow field from PINN
        N = args.grid_size
        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, N)
        X, Y = np.meshgrid(x, y)
        xy = np.stack([X.flatten(), Y.flatten()], axis=-1)
        
        print("Computing initial flow field from PINN...")
        psi_p = network.predict(xy, batch_size=len(xy))
        u_init, v_init = compute_uv_from_psi(network, xy)
        u_init = u_init.reshape(X.shape)
        v_init = v_init.reshape(X.shape)
        p_init = psi_p[..., 1].reshape(X.shape)
        
        # Parse complexity weights
        complexity_weights = {
            'strain': args.weight_strain,
            'vorticity': args.weight_vorticity,
            'momentum': args.weight_momentum,
            'continuity': args.weight_continuity
        }
        
        sim = CavityFlowDynamicHybridSimulation(
            network=network,
            uv_func=compute_uv_from_psi,
            u_init=u_init,
            v_init=v_init,
            p_init=p_init,
            Re=Re,
            N=args.grid_size,
            max_iter=args.max_iter,
            tol=args.tolerance,
            complexity_threshold=args.complexity_threshold,
            complexity_weights=complexity_weights,
            normalization=args.normalization,
            merge_distance=args.merge_distance
        )
        
        start_time = time.time()
        u, v, p = sim.solve()
        elapsed = time.time() - start_time
        
        print(f"\nSimulation time: {elapsed:.2f} seconds")
        
        # Plot results
        plot_solution(u, v, p, title_prefix='Dynamic Hybrid: ',
                     save_path='cavity_flow_dynamic.png')
        plot_hybrid_solution(u, v, p, sim.mask, 
                            save_path='cavity_flow_dynamic_mask.png')
        plot_streamlines(u, v, save_path='cavity_flow_dynamic_streamlines.png',
                        title='Dynamic Hybrid PINN-CFD Streamlines')
    
    return u, v, p


def run_cylinder_simulation(args):
    """Run flow around cylinder simulation."""
    from lib.cylinder_flow import (
        CylinderFlowSimulation,
        CylinderFlowHybridSimulation,
        CylinderFlowDynamicHybridSimulation,
        create_cylinder_boundary_mask,
        create_cylinder_wake_mask
    )
    from cylinder_network import Network as CylinderNetwork
    
    print("=" * 60)
    print("FLOW AROUND CYLINDER SIMULATION")
    print("=" * 60)
    
    # Physical parameters
    u0 = 1.0
    nu = 0.01
    rho = 1.0
    D = 2 * args.cylinder_radius  # Cylinder diameter
    Re = rho * u0 / nu
    
    print(f"\nPhysical parameters:")
    print(f"  Reynolds number: {Re}")
    print(f"  Grid size: {args.grid_size}x{args.grid_size}")
    print(f"  Domain: x=[{args.x_min}, {args.x_max}], y=[{args.y_min}, {args.y_max}]")
    print(f"  Cylinder: center=({args.cylinder_x}, {args.cylinder_y}), radius={args.cylinder_radius}")
    print(f"  Mode: {args.mode}")
    print()
    
    if args.mode == 'cfd':
        sim = CylinderFlowSimulation(
            Re=Re,
            N=args.grid_size,
            max_iter=args.max_iter,
            tol=args.tolerance,
            x_domain=(args.x_min, args.x_max),
            y_domain=(args.y_min, args.y_max),
            cylinder_center=(args.cylinder_x, args.cylinder_y),
            cylinder_radius=args.cylinder_radius,
            inlet_velocity=u0
        )
        
        start_time = time.time()
        u, v, p = sim.solve()
        elapsed = time.time() - start_time
        
        print(f"\nSimulation time: {elapsed:.2f} seconds")
        
        # Use coordinate arrays from the simulation (handles Nx × Ny grid)
        X, Y = sim.X, sim.Y
        
        circle_info = (args.cylinder_x, args.cylinder_y, args.cylinder_radius)
        
        # Plot results
        plot_solution(u, v, p, x=X, y=Y, title_prefix='CFD: ',
                     save_path='cylinder_flow_cfd.png',
                     show_circle=circle_info)
        plot_streamlines(u, v, x=X, y=Y, 
                        save_path='cylinder_flow_streamlines.png',
                        show_circle=circle_info,
                        title='Flow Around Cylinder - Streamlines')
    
    elif args.mode == 'hybrid':
        # Hybrid PINN-CFD simulation for cylinder flow
        # Build cylinder network (outputs u, v, p directly)
        network = CylinderNetwork().build(num_inputs=2, layers=[48,48,48,48], 
                                          activation='tanh', num_outputs=3)
        
        # Determine model path - auto-select based on Re if not provided
        if args.model_path:
            model_path = args.model_path
        else:
            # Auto-select model based on Reynolds number
            if Re <= 10:
                model_path = './models/pinn_cylinder_1.0.h5'
            else:
                model_path = './models/pinn_cylinder_100.0.h5'
            print(f"Auto-selected model: {model_path}")
        
        print(f"Loading PINN model from {model_path}")
        network.load_weights(model_path)
        
        # Create mask - CFD at boundaries and wake, PINN elsewhere
        x_domain = (args.x_min, args.x_max)
        y_domain = (args.y_min, args.y_max)
        cylinder_center = (args.cylinder_x, args.cylinder_y)
        
        border_width = int(args.grid_size * args.border_fraction)
        
        # Use boundary mask with wake region for better accuracy
        if args.mask_type == 'boundary':
            mask = create_cylinder_boundary_mask(
                args.grid_size, x_domain, y_domain,
                cylinder_center, args.cylinder_radius,
                border_width=border_width,
                include_wake=False,
                cfd_radius_factor=args.cfd_radius_factor
            )
        elif args.mask_type == 'wake':
            mask = create_cylinder_wake_mask(
                args.grid_size, x_domain, y_domain,
                cylinder_center, args.cylinder_radius,
                wake_length=args.wake_length
            )
        else:  # 'combined' - default
            mask = create_cylinder_boundary_mask(
                args.grid_size, x_domain, y_domain,
                cylinder_center, args.cylinder_radius,
                border_width=border_width,
                include_wake=True,
                wake_length=args.wake_length,
                cfd_radius_factor=args.cfd_radius_factor
            )
        
        sim = CylinderFlowHybridSimulation(
            network=network,
            uv_func=compute_uv_direct,
            mask=mask,
            Re=Re,
            N=args.grid_size,
            max_iter=args.max_iter,
            tol=args.tolerance,
            x_domain=x_domain,
            y_domain=y_domain,
            cylinder_center=cylinder_center,
            cylinder_radius=args.cylinder_radius,
            inlet_velocity=u0
        )
        
        start_time = time.time()
        u, v, p = sim.solve()
        elapsed = time.time() - start_time
        
        print(f"\nSimulation time: {elapsed:.2f} seconds")
        
        # Use coordinate arrays from the simulation
        X, Y = sim.X, sim.Y
        
        circle_info = (args.cylinder_x, args.cylinder_y, args.cylinder_radius)
        
        # Plot results
        plot_solution(u, v, p, x=X, y=Y, title_prefix='Hybrid: ',
                     save_path='cylinder_flow_hybrid.png',
                     show_circle=circle_info)
        plot_hybrid_solution(u, v, p, mask, x=X, y=Y,
                            save_path='cylinder_flow_hybrid_mask.png',
                            show_circle=circle_info)
        plot_streamlines(u, v, x=X, y=Y,
                        save_path='cylinder_flow_hybrid_streamlines.png',
                        show_circle=circle_info,
                        title='Hybrid PINN-CFD - Streamlines')
    
    elif args.mode == 'pinn':
        # PINN-only solution (using pre-trained model)
        # Build cylinder network (outputs u, v, p directly)
        network = CylinderNetwork().build(num_inputs=2, layers=[48,48,48,48],
                                          activation='tanh', num_outputs=3)
        
        # Determine model path - auto-select based on Re if not provided
        if args.model_path:
            model_path = args.model_path
        else:
            # Auto-select model based on Reynolds number
            if Re <= 10:
                model_path = './models/pinn_cylinder_1.0.h5'
            else:
                model_path = './models/pinn_cylinder_100.0.h5'
            print(f"Auto-selected model: {model_path}")
        
        print(f"Loading PINN model from {model_path}")
        network.load_weights(model_path)
        
        # Create grid matching cylinder domain (Nx × Ny with aspect ratio)
        x_domain = (args.x_min, args.x_max)
        y_domain = (args.y_min, args.y_max)
        Lx = args.x_max - args.x_min
        Ly = args.y_max - args.y_min
        aspect_ratio = Lx / Ly
        
        Ny = args.grid_size
        Nx = int(args.grid_size * aspect_ratio)
        
        x = np.linspace(args.x_min, args.x_max, Nx)
        y = np.linspace(args.y_min, args.y_max, Ny)
        X, Y = np.meshgrid(x, y)
        xy = np.stack([X.flatten(), Y.flatten()], axis=-1)
        
        # Predict using PINN
        start_time = time.time()
        uvp = network.predict(xy, batch_size=len(xy))
        u_pinn, v_pinn = compute_uv_direct(network, xy)
        elapsed = time.time() - start_time
        
        u = u_pinn.reshape(X.shape)
        v = v_pinn.reshape(X.shape)
        p = uvp[..., 2].reshape(X.shape)
        
        # Apply cylinder mask (zero velocity inside cylinder)
        Cx, Cy = args.cylinder_x, args.cylinder_y
        cylinder_mask = ((X - Cx)**2 + (Y - Cy)**2) <= args.cylinder_radius**2
        u = np.where(cylinder_mask, 0.0, u)
        v = np.where(cylinder_mask, 0.0, v)
        
        print(f"\nPrediction time: {elapsed:.2f} seconds")
        
        circle_info = (args.cylinder_x, args.cylinder_y, args.cylinder_radius)
        
        # Plot results
        plot_solution(u, v, p, x=X, y=Y, title_prefix='PINN: ',
                     save_path='cylinder_flow_pinn.png',
                     show_circle=circle_info)
        plot_streamlines(u, v, x=X, y=Y,
                        save_path='cylinder_flow_pinn_streamlines.png',
                        show_circle=circle_info,
                        title='PINN - Flow Around Cylinder - Streamlines')
    
    elif args.mode == 'dynamic':
        # Dynamic hybrid PINN-CFD simulation with complexity-based segregation
        # Build cylinder network (outputs u, v, p directly)
        network = CylinderNetwork().build(num_inputs=2, layers=[48,48,48,48],
                                          activation='tanh', num_outputs=3)
        
        # Determine model path - auto-select based on Re if not provided
        if args.model_path:
            model_path = args.model_path
        else:
            if Re <= 10:
                model_path = './models/pinn_cylinder_1.0.h5'
            else:
                model_path = './models/pinn_cylinder_100.0.h5'
            print(f"Auto-selected model: {model_path}")
        
        print(f"Loading PINN model from {model_path}")
        network.load_weights(model_path)
        
        # Create grid matching cylinder domain
        x_domain = (args.x_min, args.x_max)
        y_domain = (args.y_min, args.y_max)
        cylinder_center = (args.cylinder_x, args.cylinder_y)
        Lx = args.x_max - args.x_min
        Ly = args.y_max - args.y_min
        aspect_ratio = Lx / Ly
        
        Ny = args.grid_size
        Nx = int(args.grid_size * aspect_ratio)
        
        x = np.linspace(args.x_min, args.x_max, Nx)
        y = np.linspace(args.y_min, args.y_max, Ny)
        X, Y = np.meshgrid(x, y)
        xy = np.stack([X.flatten(), Y.flatten()], axis=-1)
        
        # Compute initial flow field from PINN
        print("Computing initial flow field from PINN...")
        uvp = network.predict(xy, batch_size=len(xy))
        u_init = uvp[..., 0].reshape(X.shape)
        v_init = uvp[..., 1].reshape(X.shape)
        p_init = uvp[..., 2].reshape(X.shape)
        
        # Apply cylinder mask (zero velocity inside cylinder)
        Cx, Cy = args.cylinder_x, args.cylinder_y
        cylinder_mask = ((X - Cx)**2 + (Y - Cy)**2) <= args.cylinder_radius**2
        u_init = np.where(cylinder_mask, 0.0, u_init)
        v_init = np.where(cylinder_mask, 0.0, v_init)
        
        # Parse complexity weights
        complexity_weights = {
            'strain': args.weight_strain,
            'vorticity': args.weight_vorticity,
            'momentum': args.weight_momentum,
            'continuity': args.weight_continuity
        }
        
        sim = CylinderFlowDynamicHybridSimulation(
            network=network,
            uv_func=compute_uv_direct,
            u_init=u_init,
            v_init=v_init,
            p_init=p_init,
            Re=Re,
            N=args.grid_size,
            max_iter=args.max_iter,
            tol=args.tolerance,
            complexity_threshold=args.complexity_threshold,
            complexity_weights=complexity_weights,
            normalization=args.normalization,
            merge_distance=args.merge_distance,
            x_domain=x_domain,
            y_domain=y_domain,
            cylinder_center=cylinder_center,
            cylinder_radius=args.cylinder_radius,
            inlet_velocity=u_inlet
        )
        
        start_time = time.time()
        u, v, p = sim.solve()
        elapsed = time.time() - start_time
        
        print(f"\nSimulation time: {elapsed:.2f} seconds")
        
        # Use coordinate arrays from the simulation
        X, Y = sim.X, sim.Y
        
        circle_info = (args.cylinder_x, args.cylinder_y, args.cylinder_radius)
        
        # Plot results
        plot_solution(u, v, p, x=X, y=Y, title_prefix='Dynamic Hybrid: ',
                     save_path='cylinder_flow_dynamic.png',
                     show_circle=circle_info)
        plot_hybrid_solution(u, v, p, sim.mask, x=X, y=Y,
                            save_path='cylinder_flow_dynamic_mask.png',
                            show_circle=circle_info)
        plot_streamlines(u, v, x=X, y=Y,
                        save_path='cylinder_flow_dynamic_streamlines.png',
                        show_circle=circle_info,
                        title='Dynamic Hybrid PINN-CFD - Streamlines')
    
    return u, v, p


def main():
    parser = argparse.ArgumentParser(
        description='Fluid dynamics simulation with PINN-CFD hybrid methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run cavity flow with pure CFD
  python main.py --simulation cavity --mode cfd --grid-size 100
  
  # Run cavity flow with hybrid PINN-CFD
  python main.py --simulation cavity --mode hybrid --model-path ./models/pinn_cavity_flow.h5
  
  # Run cylinder flow with pure CFD
  python main.py --simulation cylinder --mode cfd --grid-size 200
  
  # Run cylinder flow with hybrid PINN-CFD (combined mask - default)
  python main.py --simulation cylinder --mode hybrid --grid-size 100
  
  # Run cylinder flow with hybrid using only boundary CFD
  python main.py --simulation cylinder --mode hybrid --mask-type boundary
  
  # Run cylinder flow with hybrid using only wake CFD
  python main.py --simulation cylinder --mode hybrid --mask-type wake --wake-length 0.5
  
  # Run cylinder flow with PINN only
  python main.py --simulation cylinder --mode pinn --grid-size 100
  
  # Run cylinder flow with specific model
  python main.py --simulation cylinder --mode hybrid --model-path ./models/pinn_cylinder_100.0.h5
  
  # Run cavity flow with dynamic segregation
  python main.py --simulation cavity --mode dynamic --complexity-threshold 1.0
  
  # Run cylinder flow with dynamic segregation and custom weights
  python main.py --simulation cylinder --mode dynamic --complexity-threshold 1.5 --weight-momentum 3.0
  
  # Run with percentile-based normalization
  python main.py --simulation cavity --mode dynamic --normalization percentile
        """
    )
    
    # General arguments
    parser.add_argument('--simulation', '-s', type=str, default='cavity',
                       choices=['cavity', 'cylinder'],
                       help='Simulation type (default: cavity)')
    parser.add_argument('--mode', '-m', type=str, default='cfd',
                       choices=['cfd', 'hybrid', 'pinn', 'dynamic'],
                       help='Solver mode: cfd, hybrid, pinn, or dynamic (default: cfd)')
    parser.add_argument('--grid-size', '-N', type=int, default=100,
                       help='Grid size (default: 100)')
    parser.add_argument('--max-iter', type=int, default=200000,
                       help='Maximum iterations (default: 200000)')
    parser.add_argument('--tolerance', type=float, default=1e-6,
                       help='Convergence tolerance (default: 1e-6)')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to PINN model weights')
    
    # Cavity-specific arguments
    parser.add_argument('--border-fraction', type=float, default=0.13,
                       help='Fraction of grid for CFD border in hybrid mode (default: 0.13)')
    
    # Cylinder-specific arguments
    parser.add_argument('--x-min', type=float, default=0.0,
                       help='Domain x minimum (default: 0.0)')
    parser.add_argument('--x-max', type=float, default=2.0,
                       help='Domain x maximum (default: 2.0)')
    parser.add_argument('--y-min', type=float, default=0.0,
                       help='Domain y minimum (default: 0.0)')
    parser.add_argument('--y-max', type=float, default=1.0,
                       help='Domain y maximum (default: 1.0)')
    parser.add_argument('--cylinder-x', type=float, default=0.5,
                       help='Cylinder center x (default: 0.5)')
    parser.add_argument('--cylinder-y', type=float, default=0.5,
                       help='Cylinder center y (default: 0.5)')
    parser.add_argument('--cylinder-radius', type=float, default=0.1,
                       help='Cylinder radius (default: 0.1)')
    parser.add_argument('--mask-type', type=str, default='combined',
                       choices=['boundary', 'wake', 'combined'],
                       help='Mask type for hybrid mode: boundary (CFD at edges), '
                            'wake (CFD in wake only), combined (both) (default: combined)')
    parser.add_argument('--wake-length', type=float, default=1.0,
                       help='Length of wake region for CFD in hybrid mode (default: 1.0)')
    parser.add_argument('--cfd-radius-factor', type=float, default=3.0,
                       help='CFD region extends this factor times cylinder_radius from cylinder center (default: 3.0)')
    
    # Dynamic segregation arguments
    parser.add_argument('--complexity-threshold', type=float, default=1.0,
                       help='Complexity score threshold for CFD/PINN segregation (default: 1.0)')
    parser.add_argument('--normalization', type=str, default='mean',
                       choices=['mean', 'max', 'percentile'],
                       help='Normalization method for complexity scoring (default: mean)')
    parser.add_argument('--weight-strain', type=float, default=1.0,
                       help='Weight for strain-rate in complexity score (default: 1.0)')
    parser.add_argument('--weight-vorticity', type=float, default=1.0,
                       help='Weight for vorticity in complexity score (default: 1.0)')
    parser.add_argument('--weight-momentum', type=float, default=2.0,
                       help='Weight for momentum residual in complexity score (default: 2.0)')
    parser.add_argument('--weight-continuity', type=float, default=2.0,
                       help='Weight for continuity residual in complexity score (default: 2.0)')
    parser.add_argument('--merge-distance', type=int, default=0,
                       help='Grid cells for merging nearby CFD regions (default: 0, disabled)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("PINN-CFD HYBRID FLUID DYNAMICS SOLVER")
    print("=" * 60 + "\n")
    
    if args.simulation == 'cavity':
        u, v, p = run_cavity_simulation(args)
    elif args.simulation == 'cylinder':
        u, v, p = run_cylinder_simulation(args)
    
    print("\nSimulation complete!")
    return u, v, p


if __name__ == "__main__":
    main()
