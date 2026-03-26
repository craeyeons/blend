"""
Run the FDM solver for 2D linear elasticity to generate ground-truth solutions.

Usage:
    python run_fdm.py --problem plate_with_hole
    python run_fdm.py --problem l_bracket
"""

import argparse
import os
import numpy as np

from lib.solver import ElasticitySolver
from lib.domains import create_plate_with_hole, create_l_bracket


def main():
    parser = argparse.ArgumentParser(
        description='Run FDM solver for 2D linear elasticity'
    )
    parser.add_argument('--problem', type=str, default='plate_with_hole',
                        choices=['plate_with_hole', 'l_bracket'])
    parser.add_argument('--nx', type=int, default=200)
    parser.add_argument('--ny', type=int, default=200)
    parser.add_argument('--max-iter', type=int, default=200000)
    parser.add_argument('--tol', type=float, default=1e-8)
    parser.add_argument('--omega', type=float, default=0.2,
                        help='Jacobi under-relaxation factor in (0,1], smaller is more stable')
    parser.add_argument('--output-dir', type=str, default='./results')

    # Material
    parser.add_argument('--E', type=float, default=1.0)
    parser.add_argument('--nu', type=float, default=0.3)
    parser.add_argument('--applied-stress', type=float, default=10.0)

    # Plate with hole
    parser.add_argument('--x-min', type=float, default=-2.0)
    parser.add_argument('--x-max', type=float, default=2.0)
    parser.add_argument('--y-min', type=float, default=-2.0)
    parser.add_argument('--y-max', type=float, default=2.0)
    parser.add_argument('--hole-x', type=float, default=0.0)
    parser.add_argument('--hole-y', type=float, default=0.0)
    parser.add_argument('--hole-radius', type=float, default=0.5)

    # L-bracket
    parser.add_argument('--corner-x', type=float, default=1.0)
    parser.add_argument('--corner-y', type=float, default=1.0)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.problem == 'l_bracket' and args.x_min == -2.0:
        args.x_min = 0.0
        args.y_min = 0.0

    print("=" * 60)
    print(f"FDM SOLVER: {args.problem}")
    print("=" * 60)

    # Create domain
    if args.problem == 'plate_with_hole':
        X, Y, layout, disp_bc_mask, trac_bc_mask, bc_ux, bc_uy, bc_tx, bc_ty = \
            create_plate_with_hole(
                Nx=args.nx, Ny=args.ny,
                x_domain=(args.x_min, args.x_max),
                y_domain=(args.y_min, args.y_max),
                hole_center=(args.hole_x, args.hole_y),
                hole_radius=args.hole_radius,
                applied_stress=args.applied_stress,
            )
    else:
        X, Y, layout, disp_bc_mask, trac_bc_mask, bc_ux, bc_uy, bc_tx, bc_ty = \
            create_l_bracket(
                Nx=args.nx, Ny=args.ny,
                x_domain=(args.x_min, args.x_max),
                y_domain=(args.y_min, args.y_max),
                corner_x=args.corner_x,
                corner_y=args.corner_y,
                applied_stress=args.applied_stress,
            )

    # Solve
    solver = ElasticitySolver(
        E=args.E, nu=args.nu,
        x_domain=(args.x_min, args.x_max),
        y_domain=(args.y_min, args.y_max),
        Nx=args.nx, Ny=args.ny,
        max_iter=args.max_iter, tol=args.tol,
        relaxation=args.omega,
    )

    ux, uy, sxx, syy, sxy = solver.solve(
        layout, disp_bc_mask, bc_ux, bc_uy,
        trac_bc_mask, bc_tx, bc_ty,
    )

    vm = solver.compute_von_mises(sxx, syy, sxy)

    # Save
    out_path = os.path.join(args.output_dir, f'fdm_{args.problem}.npz')
    np.savez(out_path,
             X=X, Y=Y, layout=layout,
             ux=ux, uy=uy,
             sxx=sxx, syy=syy, sxy=sxy,
             von_mises=vm)
    print(f"\nSaved to {out_path}")
    print(f"  ux range: [{ux.min():.6f}, {ux.max():.6f}]")
    print(f"  uy range: [{uy.min():.6f}, {uy.max():.6f}]")
    print(f"  VM stress max: {vm.max():.6f}")

    # Visualize
    try:
        import matplotlib.pyplot as plt
        try:
            import scienceplots
            plt.style.use(['science', 'no-latex'])
        except ImportError:
            pass

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        fields = [
            (ux, 'ux (displacement)', 'RdBu_r'),
            (uy, 'uy (displacement)', 'RdBu_r'),
            (vm, 'von Mises stress', 'hot'),
            (sxx, 'sigma_xx', 'RdBu_r'),
            (syy, 'sigma_yy', 'RdBu_r'),
            (sxy, 'sigma_xy', 'RdBu_r'),
        ]

        for ax, (field, title, cmap) in zip(axes.flat, fields):
            f_masked = np.ma.masked_where(layout == 0, field)
            cf = ax.contourf(X, Y, f_masked, levels=50, cmap=cmap)
            plt.colorbar(cf, ax=ax)
            ax.set_aspect('equal')
            ax.set_title(title)

        plt.suptitle(f'FDM Solution: {args.problem}', fontsize=14)
        plt.tight_layout()
        fig_path = os.path.join(args.output_dir, f'fdm_{args.problem}.png')
        plt.savefig(fig_path, dpi=150)
        print(f"Saved plot to {fig_path}")
        plt.show()
    except Exception as e:
        print(f"Plotting skipped: {e}")


if __name__ == "__main__":
    main()
