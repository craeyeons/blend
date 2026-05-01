"""
Naive-threshold sanity check for the cylinder NSE setup.

No router. Compute R(x) = median_normalize(|residual| + |ete|), set
Omega_C = {R(x) >= beta}, run the existing hybrid PINN+CFD simulation,
and dump a single hybrid RMSE vs full CFD into stats.txt.

Reuses helpers from plot_coverage_metrics.py and lib/router.py.
"""

import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf

# Reuse the analysis pipeline's helpers verbatim (residual, ETE, hybrid).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_coverage_metrics import (
    compute_cfd_solution,
    compute_hybrid_solution,
    compute_uv_direct,
    load_pinn_solution,
    plot_solution_comparison,
)
from lib.cylinder_flow import CylinderFlowHybridSimulation
from lib.router import (
    PINNResidualComputer,
    compute_bc_error_field,
    solve_error_transport,
)


def parse_args():
    p = argparse.ArgumentParser(
        description='Naive-threshold sanity check for cylinder NSE.'
    )
    p.add_argument('--pinn-path', type=str, required=True)
    p.add_argument('--cfd-path', type=str, default=None,
                   help='Optional cached full-CFD .npz; computed if missing.')
    p.add_argument('--save-cfd', type=str, default='./cfd_naive.npz')
    p.add_argument('--output-dir', type=str, default='./naive_threshold_output_cfd')
    p.add_argument('--beta', type=float, default=1.1)
    # Domain / discretisation (match plot_coverage_metrics.py defaults)
    p.add_argument('--nx', type=int, default=200)
    p.add_argument('--ny', type=int, default=100)
    p.add_argument('--x-min', type=float, default=0.0)
    p.add_argument('--x-max', type=float, default=2.0)
    p.add_argument('--y-min', type=float, default=0.0)
    p.add_argument('--y-max', type=float, default=1.0)
    p.add_argument('--cylinder-x', type=float, default=0.5)
    p.add_argument('--cylinder-y', type=float, default=0.5)
    p.add_argument('--cylinder-radius', type=float, default=0.1)
    p.add_argument('--inlet-velocity', type=float, default=1.0)
    p.add_argument('--Re', type=float, default=100)
    p.add_argument('--max-iter', type=int, default=200000)
    p.add_argument('--tol', type=float, default=1e-6)
    p.add_argument('--morph-kernel', type=int, default=5)
    p.add_argument('--no-morph', action='store_true',
                   help='Skip morphological opening of the mask (true naive rule).')
    p.add_argument('--compute-cfd', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Grid + layout (cylinder hole removed from active region).
    x = np.linspace(args.x_min, args.x_max, args.nx)
    y = np.linspace(args.y_min, args.y_max, args.ny)
    X, Y = np.meshgrid(x, y)
    layout = np.ones_like(X, dtype=np.float32)
    layout[(X - args.cylinder_x) ** 2 + (Y - args.cylinder_y) ** 2
           < args.cylinder_radius ** 2] = 0.0

    # 2) Load PINN.
    print(f'[1/5] Loading PINN from {args.pinn_path} ...')
    pinn_model = tf.keras.models.load_model(args.pinn_path, compile=False)

    u_pinn, v_pinn, p_pinn, bc_mask, bc_u, bc_v, bc_p = load_pinn_solution(
        pinn_model, X, Y, layout
    )

    # 3) Reference CFD (cached or recomputed).
    print('[2/5] Loading / computing reference CFD ...')
    if args.cfd_path and os.path.exists(args.cfd_path) and not args.compute_cfd:
        data = np.load(args.cfd_path)
        u_cfd, v_cfd, p_cfd = data['u'], data['v'], data['p']
    else:
        u_cfd, v_cfd, p_cfd, _ = compute_cfd_solution(args)
        np.savez(args.save_cfd, u=u_cfd, v=v_cfd, p=p_cfd)

    # 4) Residual + ETE -> R(x), median-normalised.
    print('[3/5] Computing residual + ETE ...')
    bc_error_local = compute_bc_error_field(
        bc_mask, bc_u, bc_v, u_pinn, v_pinn, layout
    )
    error_transport = solve_error_transport(
        u_pinn, v_pinn, bc_error_local, layout, nu=1.0 / args.Re,
        x_domain=(args.x_min, args.x_max),
        y_domain=(args.y_min, args.y_max),
    )

    residual_computer = PINNResidualComputer(pinn_model, nu=1.0 / args.Re, rho=1.0)
    res_field = residual_computer.compute_total_residual_with_bc(
        tf.constant(X, dtype=tf.float32),
        tf.constant(Y, dtype=tf.float32),
        tf.constant(bc_mask, dtype=tf.float32),
        tf.constant(bc_u, dtype=tf.float32),
        tf.constant(bc_v, dtype=tf.float32),
        {'continuity': 1.0, 'momentum': 1.0},
    ).numpy() * layout

    R = res_field + error_transport
    fluid = layout > 0
    med = float(np.median(R[fluid]))
    if med > 1e-10:
        R = R / med

    # 5) Naive threshold hybrid.
    print(f'[4/5] Running hybrid with mask = (R(x) >= {args.beta:.3f})'
          f'  morph={"off" if args.no_morph else f"on (k={args.morph_kernel})"} ...')
    t0 = time.time()
    if args.no_morph:
        # Build the mask directly without morphological opening so the
        # naive threshold rule is exposed end-to-end.
        cfd_mask = ((R >= float(args.beta)).astype(np.int32)
                    * layout.astype(np.int32))
        cfd_fraction = 100.0 * np.sum(cfd_mask) / np.sum(layout)
        print(f'  CFD region: {cfd_fraction:.1f}%')
        sim = CylinderFlowHybridSimulation(
            network=pinn_model, uv_func=compute_uv_direct, mask=cfd_mask,
            Re=args.Re, N=args.ny,
            max_iter=args.max_iter, tol=args.tol,
            x_domain=(args.x_min, args.x_max),
            y_domain=(args.y_min, args.y_max),
            cylinder_center=(args.cylinder_x, args.cylinder_y),
            cylinder_radius=args.cylinder_radius,
            inlet_velocity=args.inlet_velocity,
        )
        solve_t0 = time.time()
        u_h, v_h, p_h = sim.solve()
        hybrid_time = time.time() - solve_t0
        u_h, v_h, p_h = np.array(u_h), np.array(v_h), np.array(p_h)
    else:
        u_h, v_h, p_h, cfd_mask, hybrid_time = compute_hybrid_solution(
            pinn_model, R, layout, float(args.beta), args
        )
    coverage = float(np.mean(cfd_mask[fluid]))

    # RMSE on velocity magnitude (matches plot_coverage_metrics.py convention).
    pinn_mag = np.sqrt(u_pinn ** 2 + v_pinn ** 2)
    cfd_mag = np.sqrt(u_cfd ** 2 + v_cfd ** 2)
    hyb_mag = np.sqrt(u_h ** 2 + v_h ** 2)
    rmse_pinn = float(np.sqrt(np.mean((pinn_mag[fluid] - cfd_mag[fluid]) ** 2)))
    rmse_hybrid = float(np.sqrt(np.mean((hyb_mag[fluid] - cfd_mag[fluid]) ** 2)))

    # 6) Solution comparison plot (PINN | Hybrid | CFD).
    print('[5/5] Writing plots and stats ...')
    plot_solution_comparison(
        u_pinn, v_pinn, p_pinn,
        u_cfd, v_cfd, p_cfd,
        u_h, v_h, p_h,
        X, Y, layout, cfd_mask,
        cylinder_center=(args.cylinder_x, args.cylinder_y),
        cylinder_radius=args.cylinder_radius,
        save_path=os.path.join(args.output_dir, 'solution_comparison.png'),
    )

    stats_path = os.path.join(args.output_dir, 'stats.txt')
    with open(stats_path, 'w') as fp:
        fp.write('Naive-threshold sanity check (cylinder NSE)\n')
        fp.write('===========================================\n')
        fp.write(f'PINN path:           {args.pinn_path}\n')
        fp.write(f'beta:                {args.beta}\n')
        fp.write(f'cylinder (x,y,r):    ({args.cylinder_x}, {args.cylinder_y}, {args.cylinder_radius})\n')
        fp.write(f'Re:                  {args.Re}\n')
        fp.write(f'CFD coverage:        {coverage * 100:.2f}%\n')
        fp.write(f'PINN RMSE vs CFD:    {rmse_pinn:.6f}\n')
        fp.write(f'Hybrid RMSE vs CFD:  {rmse_hybrid:.6f}\n')
        fp.write(f'Hybrid solve time:   {hybrid_time:.2f} s\n')
        fp.write(f'Total wall time:     {time.time() - t0:.2f} s\n')
    print(f'-> {stats_path}')
    print(f'   PINN RMSE   = {rmse_pinn:.6f}')
    print(f'   Hybrid RMSE = {rmse_hybrid:.6f}   (coverage {coverage * 100:.1f}%)')


if __name__ == '__main__':
    main()
