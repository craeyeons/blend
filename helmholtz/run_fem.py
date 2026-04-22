"""
Solve the manufactured-solution Helmholtz problem with scikit-fem
and save the solution interpolated onto a regular grid.

Usage:
    python run_fem.py --k 12.566 --mesh-n 129 --nx 201 --ny 201 --tag k4pi
"""

import argparse
import os
import time
import numpy as np

from lib.fem_solver import HelmholtzSolver
from lib.domains import manufactured_solution, gaussian_source


def rmse(pred, exact):
    return float(np.sqrt(np.mean((pred - exact) ** 2)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--k', type=float, required=True)
    p.add_argument('--mesh-n', type=int, default=129,
                   help='Nodes per side of the triangular mesh')
    p.add_argument('--nx', type=int, default=201,
                   help='Regular-grid Nx for interpolation')
    p.add_argument('--ny', type=int, default=201)
    p.add_argument('--tag', type=str, default=None)
    p.add_argument('--output-dir', type=str, default='./results')
    p.add_argument('--source', type=str, default='manufactured',
                   choices=['manufactured', 'gaussian'])
    p.add_argument('--x-s', type=float, default=0.5)
    p.add_argument('--y-s', type=float, default=0.5)
    p.add_argument('--sigma', type=float, default=0.05)
    p.add_argument('--amplitude', type=float, default=1.0)
    p.add_argument('--domain', type=str, default='square',
                   choices=['square', 'square_hole'])
    p.add_argument('--hole-center-x', type=float, default=0.5)
    p.add_argument('--hole-center-y', type=float, default=0.5)
    p.add_argument('--hole-radius', type=float, default=0.15)
    args = p.parse_args()

    tag = args.tag or f'k{args.k:.3f}'.replace('.', 'p')
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"FEM SOLVE: Helmholtz  k={args.k:.4f}  mesh_n={args.mesh_n}")
    print("=" * 60)

    dirichlet_predicate = None
    if args.domain == 'square_hole':
        cx, cy, r = args.hole_center_x, args.hole_center_y, args.hole_radius
        def dirichlet_predicate(x, y):
            return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2

    solver = HelmholtzSolver(k=args.k, mesh_n=args.mesh_n,
                             dirichlet_predicate=dirichlet_predicate)

    if args.domain == 'square_hole' and args.source == 'manufactured':
        raise ValueError(
            "square_hole domain does not have an analytic solution; use "
            "--source gaussian.")

    if args.source == 'manufactured':
        def f_callable(x, y):
            return (args.k ** 2) * np.sin(args.k * x) * np.sin(args.k * y)

        def g_callable(x, y):
            return np.sin(args.k * x) * np.sin(args.k * y)
    else:  # gaussian, homogeneous Dirichlet everywhere
        def f_callable(x, y):
            r2 = (x - args.x_s) ** 2 + (y - args.y_s) ** 2
            return args.amplitude * np.exp(-r2 / (2.0 * args.sigma ** 2))

        def g_callable(x, y):
            return np.zeros_like(x)

    t_assemble_start = time.perf_counter()
    u_dof, solve_time_s = solver.solve(f_callable, g_callable)
    assemble_plus_solve = time.perf_counter() - t_assemble_start
    print(f"FEM assemble+solve time: {assemble_plus_solve:.4f}s "
          f"(inner solve: {solve_time_s:.4f}s)")

    # Interpolate to regular grid
    xs = np.linspace(0.0, 1.0, args.nx, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, args.ny, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    u_fem = solver.interp_to_grid(u_dof, X, Y)

    save_kw = dict(
        X=X, Y=Y, u_fem=u_fem,
        k=np.float32(args.k),
        mesh_n=np.int32(args.mesh_n),
        solve_time_s=np.float32(solve_time_s),
        assemble_plus_solve_s=np.float32(assemble_plus_solve),
        source=np.array(args.source),
        domain=np.array(args.domain),
        hole_center_x=np.float32(args.hole_center_x),
        hole_center_y=np.float32(args.hole_center_y),
        hole_radius=np.float32(args.hole_radius),
    )
    if args.source == 'manufactured':
        u_exact, _ = manufactured_solution(X, Y, args.k)
        err = rmse(u_fem, u_exact)
        print(f"FEM RMSE vs u*: {err:.4e}")
        save_kw['u_exact'] = u_exact
        save_kw['fem_rmse'] = np.float32(err)
    else:
        print(f"Gaussian source: no analytic u*; FEM will serve as reference.")
        save_kw['x_s'] = np.float32(args.x_s)
        save_kw['y_s'] = np.float32(args.y_s)
        save_kw['sigma'] = np.float32(args.sigma)
        save_kw['amplitude'] = np.float32(args.amplitude)

    out_path = os.path.join(args.output_dir, f'fem_helmholtz_{tag}.npz')
    np.savez(out_path, **save_kw)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
