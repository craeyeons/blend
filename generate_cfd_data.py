"""
Generate CFD Training Data for Learned Router

This script computes CFD solutions for all training cases:
- 5 Reynolds numbers: [1.0, 10.0, 40.0, 100.0, 200.0]
- 5 cylinder positions: [(0.5, 0.3), (0.5, 0.4), (0.5, 0.5), (0.5, 0.6), (0.5, 0.7)]
- Total: 25 cases

Each CFD solution is saved as a .npz file containing:
- u, v, p: velocity and pressure fields
- X, Y: mesh grids
- cylinder_mask: boolean mask for cylinder region

Usage:
    python generate_cfd_data.py              # Run all 25 cases
    python generate_cfd_data.py --Re 100     # Run only Re=100 cases
    python generate_cfd_data.py --parallel   # Run with multiprocessing
"""

import numpy as np
import os
import json
import time
import argparse
from typing import Tuple, List, Optional

from lib.cylinder_flow import CylinderFlowSimulation


# =============================================================================
# Configuration
# =============================================================================

# Reynolds numbers for training
# RE_VALUES = [1.0, 10.0, 40.0, 100.0, 200.0]
RE_VALUES = [100.0]

# Cylinder positions (Cx, Cy) for training
# CYLINDER_POSITIONS = [
#     (0.5, 0.3),
#     (0.5, 0.4),
#     (0.5, 0.5),
#     (0.5, 0.6),
#     (0.5, 0.7),
# ]
CYLINDER_POSITIONS = [
    (0.5, 0.5)
]

# Domain settings
X_DOMAIN = (0.0, 2.0)
Y_DOMAIN = (0.0, 1.0)
N = 100
CYLINDER_RADIUS = 0.1
INLET_VELOCITY = 1.0

# CFD solver settings
MAX_ITER = 100000
TOL = 1e-6

# Output directory
DATA_DIR = './data'

# PINN model filename (expected in same directory as CFD solution)
PINN_MODEL_FILENAME = 'pinn_model.h5'


# =============================================================================
# CFD Computation Functions
# =============================================================================

def compute_cfd_solution(Re: float, 
                         cylinder_center: Tuple[float, float],
                         cylinder_radius: float = CYLINDER_RADIUS,
                         inlet_velocity: float = INLET_VELOCITY,
                         N: int = N,
                         x_domain: Tuple[float, float] = X_DOMAIN,
                         y_domain: Tuple[float, float] = Y_DOMAIN,
                         max_iter: int = MAX_ITER,
                         tol: float = TOL,
                         verbose: bool = True) -> dict:
    """
    Compute CFD solution for a single case.
    
    Parameters:
    -----------
    Re : float
        Reynolds number
    cylinder_center : tuple
        (Cx, Cy) position of cylinder center
    cylinder_radius : float
        Radius of cylinder
    inlet_velocity : float
        Inlet velocity
    N : int
        Grid resolution (N x N)
    x_domain, y_domain : tuple
        Domain bounds
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    verbose : bool
        Print progress
    
    Returns:
    --------
    dict with solution data
    """
    if verbose:
        print(f"  Re={Re}, cylinder=({cylinder_center[0]}, {cylinder_center[1]})")
    
    # Create simulation
    sim = CylinderFlowSimulation(
        Re=Re,
        N=N,
        max_iter=max_iter,
        tol=tol,
        x_domain=x_domain,
        y_domain=y_domain,
        cylinder_center=cylinder_center,
        cylinder_radius=cylinder_radius,
        inlet_velocity=inlet_velocity
    )
    
    # Solve
    start_time = time.time()
    u, v, p = sim.solve()
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"    Converged in {elapsed:.1f}s")
    
    return {
        'u': u,
        'v': v,
        'p': p,
        'X': sim.X,
        'Y': sim.Y,
        'cylinder_mask': sim.cylinder_mask,
        'elapsed_time': elapsed,
        'converged': True,  # CylinderFlowSimulation raises if not converged
    }


def save_cfd_solution(solution: dict, 
                      output_path: str,
                      Re: float,
                      cylinder_center: Tuple[float, float],
                      cylinder_radius: float) -> None:
    """
    Save CFD solution to .npz file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.savez(
        output_path,
        u=solution['u'],
        v=solution['v'],
        p=solution['p'],
        X=solution['X'],
        Y=solution['Y'],
        cylinder_mask=solution['cylinder_mask'],
        Re=Re,
        cylinder_center=np.array(cylinder_center),
        cylinder_radius=cylinder_radius,
        elapsed_time=solution['elapsed_time']
    )


def compute_single_case(args: tuple) -> dict:
    """
    Compute a single case (for multiprocessing).
    """
    Re, cylinder_center, output_path = args
    
    try:
        solution = compute_cfd_solution(
            Re=Re,
            cylinder_center=cylinder_center,
            verbose=True
        )
        
        save_cfd_solution(
            solution, output_path,
            Re=Re,
            cylinder_center=cylinder_center,
            cylinder_radius=CYLINDER_RADIUS
        )
        
        return {
            'Re': Re,
            'cylinder_center': cylinder_center,
            'success': True,
            'elapsed_time': solution['elapsed_time'],
            'output_path': output_path
        }
    except Exception as e:
        print(f"  ✗ Failed: Re={Re}, cyl={cylinder_center}: {e}")
        return {
            'Re': Re,
            'cylinder_center': cylinder_center,
            'success': False,
            'error': str(e)
        }


def generate_all_cfd_solutions(re_values: Optional[List[float]] = None,
                                cylinder_positions: Optional[List[Tuple[float, float]]] = None,
                                output_dir: str = DATA_DIR,
                                use_parallel: bool = False,
                                n_workers: int = 4) -> List[dict]:
    """
    Generate CFD solutions for all training cases.
    
    Parameters:
    -----------
    re_values : list or None
        Reynolds numbers to compute (default: all)
    cylinder_positions : list or None
        Cylinder positions to compute (default: all)
    output_dir : str
        Output directory
    use_parallel : bool
        Use multiprocessing
    n_workers : int
        Number of parallel workers
    
    Returns:
    --------
    results : list of dict
        Results for each case
    """
    if re_values is None:
        re_values = RE_VALUES
    if cylinder_positions is None:
        cylinder_positions = CYLINDER_POSITIONS
    
    # Build list of cases
    cases = []
    for Re in re_values:
        for (Cx, Cy) in cylinder_positions:
            output_path = os.path.join(
                output_dir, 
                f"Re_{int(Re)}/cyl_pos_{Cx}_{Cy}/cfd_solution.npz"
            )
            cases.append((Re, (Cx, Cy), output_path))
    
    print(f"\n{'='*70}")
    print(f"Generating CFD Training Data")
    print(f"{'='*70}")
    print(f"Total cases: {len(cases)}")
    print(f"Reynolds numbers: {re_values}")
    print(f"Cylinder positions: {len(cylinder_positions)}")
    print(f"Grid resolution: {N}x{N}")
    print(f"Output directory: {output_dir}")
    print()
    
    results = []
    start_time = time.time()
    
    if use_parallel:
        from multiprocessing import Pool
        print(f"Using {n_workers} parallel workers")
        with Pool(n_workers) as pool:
            results = pool.map(compute_single_case, cases)
    else:
        for i, case in enumerate(cases):
            print(f"\n[{i+1}/{len(cases)}] Computing case...")
            result = compute_single_case(case)
            results.append(result)
    
    total_time = time.time() - start_time
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Successful: {successful}/{len(results)}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average time per case: {total_time/len(results):.1f}s")
    
    return results


def generate_cases_json(output_dir: str = DATA_DIR) -> None:
    """
    Generate cases.json metadata file after CFD solutions are computed.
    
    Expected directory structure:
        data/Re_{Re}/cyl_pos_{Cx}_{Cy}/
            ├── cfd_solution.npz
            └── pinn_model.h5
    """
    cases = []
    
    for Re in RE_VALUES:
        for (Cx, Cy) in CYLINDER_POSITIONS:
            case_dir = f"Re_{int(Re)}/cyl_pos_{Cx}_{Cy}"
            cfd_path = f"{case_dir}/cfd_solution.npz"
            pinn_path = f"{case_dir}/{PINN_MODEL_FILENAME}"
            full_cfd_path = os.path.join(output_dir, cfd_path)
            
            if os.path.exists(full_cfd_path):
                case_id = f"Re{int(Re)}_cyl_{Cx}_{Cy}"
                cases.append({
                    "id": case_id,
                    "Re": Re,
                    "cylinder_center": [Cx, Cy],
                    "cylinder_radius": CYLINDER_RADIUS,
                    "inlet_velocity": INLET_VELOCITY,
                    "cfd_path": cfd_path,
                    "pinn_model_path": pinn_path
                })
    
    metadata = {
        "description": "Training cases for learned router (5 Re x 5 cylinder positions)",
        "note": f"PINN models expected at {{case_dir}}/{PINN_MODEL_FILENAME}",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "domain": {
            "x_domain": list(X_DOMAIN),
            "y_domain": list(Y_DOMAIN),
            "N": N
        },
        "cases": cases
    }
    
    output_path = os.path.join(output_dir, 'cases.json')
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nGenerated {output_path} with {len(cases)} cases")


def check_existing_data(output_dir: str = DATA_DIR) -> dict:
    """
    Check which CFD solutions already exist.
    """
    existing = []
    missing = []
    
    for Re in RE_VALUES:
        for (Cx, Cy) in CYLINDER_POSITIONS:
            cfd_path = os.path.join(
                output_dir,
                f"Re_{int(Re)}/cyl_pos_{Cx}_{Cy}/cfd_solution.npz"
            )
            
            case_id = f"Re{int(Re)}_cyl_{Cx}_{Cy}"
            
            if os.path.exists(cfd_path):
                existing.append(case_id)
            else:
                missing.append((Re, (Cx, Cy)))
    
    return {
        'existing': existing,
        'missing': missing,
        'n_existing': len(existing),
        'n_missing': len(missing),
        'n_total': len(existing) + len(missing)
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate CFD training data for learned router',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_cfd_data.py              # Compute all 25 cases
  python generate_cfd_data.py --Re 100     # Only Re=100 (5 positions)
  python generate_cfd_data.py --check      # Check existing data
  python generate_cfd_data.py --missing    # Compute only missing cases
  python generate_cfd_data.py --parallel   # Use multiprocessing
        """
    )
    
    parser.add_argument('--Re', type=float, nargs='+', 
                        help='Reynolds numbers to compute (default: all)')
    parser.add_argument('--output', type=str, default=DATA_DIR,
                        help='Output directory (default: ./data)')
    parser.add_argument('--parallel', action='store_true',
                        help='Use multiprocessing')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--check', action='store_true',
                        help='Check existing data only')
    parser.add_argument('--missing', action='store_true',
                        help='Compute only missing cases')
    parser.add_argument('--N', type=int, default=100,
                        help='Grid resolution (default: 100)')
    parser.add_argument('--max-iter', type=int, default=100000,
                        help='Max iterations (default: 100000)')
    
    args = parser.parse_args()
    
    # Use command-line args for grid resolution and max iterations
    grid_N = args.N
    max_iterations = args.max_iter
    
    # Check existing data
    if args.check:
        status = check_existing_data(args.output)
        print(f"\n{'='*70}")
        print("DATA STATUS")
        print(f"{'='*70}")
        print(f"Existing: {status['n_existing']}/{status['n_total']}")
        print(f"Missing:  {status['n_missing']}/{status['n_total']}")
        
        if status['existing']:
            print(f"\nExisting cases:")
            for case_id in status['existing']:
                print(f"  ✓ {case_id}")
        
        if status['missing']:
            print(f"\nMissing cases:")
            for Re, pos in status['missing']:
                print(f"  ✗ Re={Re}, cyl={pos}")
        return
    
    # Determine which cases to compute
    re_values = args.Re if args.Re else RE_VALUES
    cylinder_positions = CYLINDER_POSITIONS
    
    if args.missing:
        status = check_existing_data(args.output)
        if not status['missing']:
            print("All cases already computed!")
            return
        
        # Extract Re values and positions from missing cases
        missing_re = set()
        missing_pos = set()
        for Re, pos in status['missing']:
            missing_re.add(Re)
            missing_pos.add(pos)
        
        # Filter to only missing
        re_values = [Re for Re in re_values if Re in missing_re]
        # For simplicity, compute all positions for missing Re values
        # (could be optimized to only compute truly missing combinations)
    
    # Compute CFD solutions
    results = generate_all_cfd_solutions(
        re_values=re_values,
        cylinder_positions=cylinder_positions,
        output_dir=args.output,
        use_parallel=args.parallel,
        n_workers=args.workers
    )
    
    # Generate cases.json
    generate_cases_json(args.output)
    
    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")
    print(f"CFD data saved to: {args.output}/")
    print(f"Metadata saved to: {args.output}/cases.json")
    print(f"\nNext steps:")
    print(f"  1. Ensure PINN models exist at data/Re_{{Re}}/cyl_pos_{{Cx}}_{{Cy}}/{PINN_MODEL_FILENAME}")
    print(f"  2. Run: python example_learned_router.py train")


if __name__ == '__main__':
    main()
