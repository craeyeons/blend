"""
Run FEM reference for every config in configs_exp3.json. One .npz per config.

Configs JSON format:
    [
      {"name": "id_k4pi_c", "k": 12.566, "x_s": 0.7, "y_s": 0.5,
       "split": "id"},
      ...
    ]

All configs share domain (square_hole), sigma, amplitude, hole geometry
(set via CLI flags or taken from --parametric-meta JSON).

Usage:
    python run_fem_multi.py --configs configs_exp3.json \
        --tag-prefix exp3 --parametric-meta history/training_meta_exp3_parametric.json
"""

import argparse
import json
import os
import time

import numpy as np

from lib.fem_solver import HelmholtzSolver


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--configs', type=str, required=True)
    p.add_argument('--tag-prefix', type=str, required=True)
    p.add_argument('--parametric-meta', type=str, default=None,
                   help='Read hole + sigma + amplitude defaults from this '
                        'PINN training meta JSON.')
    p.add_argument('--mesh-n', type=int, default=129)
    p.add_argument('--nx', type=int, default=201)
    p.add_argument('--ny', type=int, default=201)
    p.add_argument('--sigma', type=float, default=None)
    p.add_argument('--amplitude', type=float, default=None)
    p.add_argument('--hole-center-x', type=float, default=None)
    p.add_argument('--hole-center-y', type=float, default=None)
    p.add_argument('--hole-radius', type=float, default=None)
    p.add_argument('--output-dir', type=str, default='./results')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    defaults = {'sigma': 0.05, 'amplitude': 1.0,
                'hole_center_x': 0.5, 'hole_center_y': 0.5, 'hole_radius': 0.15}
    if args.parametric_meta and os.path.exists(args.parametric_meta):
        with open(args.parametric_meta) as f:
            meta = json.load(f)
        for key in defaults:
            if key in meta:
                defaults[key] = meta[key]

    sigma = args.sigma if args.sigma is not None else defaults['sigma']
    amplitude = args.amplitude if args.amplitude is not None else defaults['amplitude']
    cx = args.hole_center_x if args.hole_center_x is not None else defaults['hole_center_x']
    cy = args.hole_center_y if args.hole_center_y is not None else defaults['hole_center_y']
    r_h = args.hole_radius if args.hole_radius is not None else defaults['hole_radius']

    with open(args.configs) as f:
        configs = json.load(f)
    print(f"Loaded {len(configs)} configs from {args.configs}")

    # One shared grid + solver (cached stiffness/mass since k changes per config,
    # but build a fresh solver per config since _A = K - k^2 M depends on k).
    xs = np.linspace(0.0, 1.0, args.nx, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, args.ny, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    def dirichlet_predicate(x, y):
        return (x - cx) ** 2 + (y - cy) ** 2 <= r_h ** 2

    for cfg in configs:
        name = cfg['name']
        k = float(cfg['k']); xs_ = float(cfg['x_s']); ys_ = float(cfg['y_s'])
        split = cfg.get('split', 'id')
        tag = f"{args.tag_prefix}_{name}"
        print(f"\n== {name}  k={k:.4f}  x_s={xs_:.3f}  y_s={ys_:.3f}  "
              f"split={split} ==")

        solver = HelmholtzSolver(k=k, mesh_n=args.mesh_n,
                                 dirichlet_predicate=dirichlet_predicate)

        def f_callable(x, y, xs_=xs_, ys_=ys_):
            return amplitude * np.exp(-((x - xs_) ** 2 + (y - ys_) ** 2)
                                      / (2.0 * sigma ** 2))
        def g_callable(x, y):
            return np.zeros_like(x)

        t0 = time.perf_counter()
        u_dof, solve_time_s = solver.solve(f_callable, g_callable)
        wall = time.perf_counter() - t0
        u_fem = solver.interp_to_grid(u_dof, X, Y)
        print(f"  solve={solve_time_s:.4f}s  wall={wall:.4f}s")

        out = os.path.join(args.output_dir, f'fem_helmholtz_{tag}.npz')
        np.savez(out, X=X, Y=Y, u_fem=u_fem,
                 k=np.float32(k), x_s=np.float32(xs_), y_s=np.float32(ys_),
                 sigma=np.float32(sigma), amplitude=np.float32(amplitude),
                 hole_center_x=np.float32(cx),
                 hole_center_y=np.float32(cy),
                 hole_radius=np.float32(r_h),
                 mesh_n=np.int32(args.mesh_n),
                 solve_time_s=np.float32(solve_time_s),
                 split=np.array(split), name=np.array(name))
        print(f"  saved: {out}")


if __name__ == '__main__':
    main()
