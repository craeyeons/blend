"""Replot Abstention True Loss vs Coverage and Hybrid RMSE vs Coverage for
the held-out test cylinder (0.60, 0.55, 0.11), aggregating across multiple
trained router seeds to produce error bars.

For each router seed:
  - Compute the analytic abstention-loss curve via
    plot_coverage_metrics.compute_loss_vs_coverage (uses PINN residual field
    + that seed's router logits).
  - Sweep coverage in 10% increments and actually run a hybrid solve at each
    coverage level to get the RMSE-vs-coverage curve.
  - Find the per-seed optimal coverage from the abstention-loss curve and
    run one extra hybrid solve at exactly that coverage to get the optimal
    RMSE for the green star.

CFD reference is solved once (seed-independent geometry) and reused.

The two output figures share the same scienceplots style, figsize, fonts,
axis labels, error-bar style and green-star markers — no titles.
"""

import argparse
import contextlib
import io
import os

import numpy as np
import tensorflow as tf
from scipy.ndimage import binary_erosion

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
plt.style.use(['science', 'no-latex'])

from lib.router import (
    RouterCNN,
    PINNResidualComputer,
    compute_pinn_residual_field,
    create_router_input,
    compute_bc_error_field,
    solve_error_transport,
    create_cylinder_setup,
)
from lib.cylinder_flow import CylinderFlowSimulation, CylinderFlowHybridSimulation
from cylinder_network import Network as CylinderNetwork
from plot_coverage_metrics import compute_loss_vs_coverage
from time_cylinder import (
    _gauge_free_rmse,
    _interface_ring,
    threshold_for_coverage,
    apply_morph_open,
    compute_uv_direct,
)


FIGSIZE = (6, 4)
LABEL_FONTSIZE = 13
TICK_FONTSIZE = 11
LEGEND_FONTSIZE = 9
LINEWIDTH = 1.8
MARKERSIZE = 5
STAR_SIZE = 18
ERR_ALPHA = 0.25
LINE_COLOR = 'tab:blue'
STAR_COLOR = 'tab:green'


def make_hybrid(args, pinn_model, cfd_mask):
    return CylinderFlowHybridSimulation(
        network=pinn_model, uv_func=compute_uv_direct, mask=cfd_mask,
        Re=args.Re, N=args.ny, max_iter=args.max_iter, tol=args.tol,
        x_domain=(args.x_min, args.x_max),
        y_domain=(args.y_min, args.y_max),
        cylinder_center=(args.cylinder_x, args.cylinder_y),
        cylinder_radius=args.cylinder_radius,
        inlet_velocity=args.inlet_velocity,
    )


def make_cfd(args):
    return CylinderFlowSimulation(
        Re=args.Re, N=args.ny, max_iter=args.max_iter, tol=args.tol,
        x_domain=(args.x_min, args.x_max),
        y_domain=(args.y_min, args.y_max),
        cylinder_center=(args.cylinder_x, args.cylinder_y),
        cylinder_radius=args.cylinder_radius,
        inlet_velocity=args.inlet_velocity,
    )


def compute_router_field(router, setup, args):
    nu = 1.0 / args.Re
    X, Y = setup['X'], setup['Y']
    layout, bc_mask = setup['layout'], setup['bc_mask']
    bc_u, bc_v, bc_p = setup['bc_u'], setup['bc_v'], setup['bc_p']
    pu, pv, pp = setup['pu'], setup['pv'], setup['pp']
    bc_err = compute_bc_error_field(bc_mask, bc_u, bc_v, pu, pv, layout)
    ete = solve_error_transport(
        pu, pv, bc_err, layout, nu=nu,
        x_domain=(args.x_min, args.x_max),
        y_domain=(args.y_min, args.y_max),
    )
    pde_r = compute_pinn_residual_field(
        setup['pinn_model'], X, Y, layout, bc_mask, bc_u, bc_v, nu=nu
    )
    inputs = create_router_input(
        layout, bc_mask, bc_u, bc_v, bc_p, pu, pv, pp, ete, pde_r,
    )
    return router(inputs, training=False)[0, :, :, 0].numpy()


def hybrid_rmse_at_coverage(target_cov, router_output, setup, args,
                            u_cfd, v_cfd, p_cfd, dx, dy, gradp_scale,
                            interior):
    """Run one hybrid solve at the given coverage and return RMSE + actual cov."""
    layout = setup['layout']
    fluid = layout > 0
    thresh = threshold_for_coverage(router_output, layout, target_cov)
    mask = (router_output >= thresh).astype(np.int32) * layout.astype(np.int32)
    mask = apply_morph_open(mask, layout, args.morph_kernel)
    actual_cov = float(np.sum(mask) / np.sum(layout))

    sim = make_hybrid(args, setup['pinn_model'], mask)
    with contextlib.redirect_stdout(io.StringIO()):
        uh, vh, ph = sim.solve()
    uh, vh, ph = np.array(uh), np.array(vh), np.array(ph)
    ring = _interface_ring(mask.astype(bool) & fluid, iterations=1)
    valid = interior & ~ring
    rmse = _gauge_free_rmse(uh, vh, ph, u_cfd, v_cfd, p_cfd,
                            valid, dx, dy, gradp_scale)
    return rmse, actual_cov


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pinn-path', required=True)
    p.add_argument('--router-weights', nargs='+', required=True,
                   help='One or more router weight files (one per seed).')
    p.add_argument('--output-dir', default='./test_errorbars_output')
    p.add_argument('--nx', type=int, default=200)
    p.add_argument('--ny', type=int, default=100)
    p.add_argument('--x-min', type=float, default=0.0)
    p.add_argument('--x-max', type=float, default=2.0)
    p.add_argument('--y-min', type=float, default=0.0)
    p.add_argument('--y-max', type=float, default=1.0)
    p.add_argument('--cylinder-x', type=float, default=0.60)
    p.add_argument('--cylinder-y', type=float, default=0.55)
    p.add_argument('--cylinder-radius', type=float, default=0.11)
    p.add_argument('--inlet-velocity', type=float, default=1.0)
    p.add_argument('--Re', type=float, default=100)
    p.add_argument('--max-iter', type=int, default=200000)
    p.add_argument('--tol', type=float, default=1e-6)
    p.add_argument('--base-filters', type=int, default=32)
    p.add_argument('--temperature', type=float, default=0.5)
    p.add_argument('--beta', type=float, default=1.1)
    p.add_argument('--lambda-tv', type=float, default=1.0)
    p.add_argument('--morph-kernel', type=int, default=5)
    p.add_argument('--n-loss-points', type=int, default=200)
    p.add_argument('--cfd-path', type=str, default=None,
                   help='Optional cached CFD npz; if absent, CFD is solved.')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    nu = 1.0 / args.Re

    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass

    # ---- Setup (shared across seeds) ----
    X, Y, layout, bc_mask, bc_u, bc_v, bc_p = create_cylinder_setup(
        Nx=args.nx, Ny=args.ny,
        x_domain=(args.x_min, args.x_max),
        y_domain=(args.y_min, args.y_max),
        cylinder_center=(args.cylinder_x, args.cylinder_y),
        cylinder_radius=args.cylinder_radius,
        inlet_velocity=args.inlet_velocity,
    )

    network = CylinderNetwork()
    pinn_model = network.build(
        num_inputs=2, layers=[48, 48, 48, 48],
        activation='tanh', num_outputs=3,
    )
    pinn_model.load_weights(args.pinn_path)

    xy = np.stack([X.flatten(), Y.flatten()], axis=-1).astype(np.float32)
    uvp = pinn_model.predict(xy, batch_size=len(xy), verbose=0)
    pu = uvp[:, 0].reshape(X.shape).astype(np.float32) * layout
    pv = uvp[:, 1].reshape(X.shape).astype(np.float32) * layout
    pp = uvp[:, 2].reshape(X.shape).astype(np.float32) * layout

    # Residual field (PINN-only -> shared across seeds)
    rcomp = PINNResidualComputer(pinn_model, nu=nu, rho=1.0)
    pde_r = rcomp.compute_total_residual_with_bc(
        tf.constant(X, dtype=tf.float32),
        tf.constant(Y, dtype=tf.float32),
        tf.constant(bc_mask, dtype=tf.float32),
        tf.constant(bc_u, dtype=tf.float32),
        tf.constant(bc_v, dtype=tf.float32),
        {'continuity': 1.0, 'momentum': 1.0},
    ).numpy() * layout
    bc_err = compute_bc_error_field(bc_mask, bc_u, bc_v, pu, pv, layout)
    ete = solve_error_transport(
        pu, pv, bc_err, layout, nu=nu,
        x_domain=(args.x_min, args.x_max),
        y_domain=(args.y_min, args.y_max),
    )
    residual_field = pde_r + ete
    fluid_vals = residual_field[layout > 0]
    med = float(np.median(fluid_vals))
    if med > 1e-10:
        residual_field = residual_field / med

    setup = dict(
        X=X, Y=Y, layout=layout, bc_mask=bc_mask,
        bc_u=bc_u, bc_v=bc_v, bc_p=bc_p,
        pu=pu, pv=pv, pp=pp, pinn_model=pinn_model,
    )

    # ---- CFD reference (once) ----
    if args.cfd_path and os.path.isfile(args.cfd_path):
        d = np.load(args.cfd_path)
        u_cfd, v_cfd, p_cfd = d['u'], d['v'], d['p']
        print(f"  Loaded CFD reference from {args.cfd_path}")
    else:
        print("  Solving CFD reference ...")
        sim = make_cfd(args)
        with contextlib.redirect_stdout(io.StringIO()):
            u, v, pr = sim.solve()
        u_cfd, v_cfd, p_cfd = np.array(u), np.array(v), np.array(pr)
        cache = os.path.join(args.output_dir, 'cfd_ref.npz')
        np.savez(cache, u=u_cfd, v=v_cfd, p=p_cfd)
        print(f"  Saved CFD reference to {cache}")

    fluid = layout > 0
    dx = float(X[0, 1] - X[0, 0])
    dy = float(Y[1, 0] - Y[0, 0])
    pxc, pyc = np.gradient(p_cfd, dy, dx)
    gradp_scale = float(np.max(pxc[fluid] ** 2 + pyc[fluid] ** 2)) + 1e-10
    interior = binary_erosion(fluid, iterations=1)

    pinn_rmse = _gauge_free_rmse(pu, pv, pp, u_cfd, v_cfd, p_cfd,
                                 interior, dx, dy, gradp_scale)

    # ---- Sweep coverages ----
    target_covs = np.arange(0.1, 1.0, 0.1)  # 0.1 .. 0.9

    loss_curves = []
    rmse_curves = []
    actual_cov_curves = []
    opt_cov_list = []
    opt_loss_list = []
    opt_rmse_list = []

    n_seeds = len(args.router_weights)
    for s, rw in enumerate(args.router_weights):
        print(f"\n=== Seed {s} ({rw}) ===")
        router = RouterCNN(base_filters=args.base_filters,
                           temperature=args.temperature)
        # Build via dummy forward pass
        bc_err_local = compute_bc_error_field(bc_mask, bc_u, bc_v, pu, pv, layout)
        ete_local = solve_error_transport(
            pu, pv, bc_err_local, layout, nu=nu,
            x_domain=(args.x_min, args.x_max),
            y_domain=(args.y_min, args.y_max),
        )
        pde_r_local = compute_pinn_residual_field(
            pinn_model, X, Y, layout, bc_mask, bc_u, bc_v, nu=nu,
        )
        inputs = create_router_input(
            layout, bc_mask, bc_u, bc_v, bc_p,
            pu, pv, pp, ete_local, pde_r_local,
        )
        _ = router(inputs)
        router.load_weights(rw)
        router_output = compute_router_field(router, setup, args)

        # Loss curve (analytic)
        cov_grid, loss_grid, _, opt_info = compute_loss_vs_coverage(
            residual_field, router_output, layout, args.beta,
            n_points=args.n_loss_points, lambda_tv=args.lambda_tv,
        )
        loss_curves.append(loss_grid)
        opt_cov_list.append(opt_info['optimal_coverage'])
        opt_loss_list.append(opt_info['optimal_loss'])
        print(f"  optimal coverage = {opt_info['optimal_coverage']*100:.2f}%"
              f"  optimal loss = {opt_info['optimal_loss']:.4f}")

        # RMSE sweep at 10% increments
        rmse_seed = [pinn_rmse]
        cov_seed = [0.0]
        for tc in target_covs:
            rmse, actual_cov = hybrid_rmse_at_coverage(
                tc, router_output, setup, args,
                u_cfd, v_cfd, p_cfd, dx, dy, gradp_scale, interior,
            )
            rmse_seed.append(rmse)
            cov_seed.append(actual_cov)
            print(f"  cov={actual_cov*100:5.1f}%  RMSE={rmse:.5f}")
        rmse_seed.append(0.0)
        cov_seed.append(1.0)
        rmse_curves.append(np.array(rmse_seed))
        actual_cov_curves.append(np.array(cov_seed))

        # Optimal-coverage RMSE (extra solve)
        opt_rmse, opt_actual_cov = hybrid_rmse_at_coverage(
            opt_info['optimal_coverage'], router_output, setup, args,
            u_cfd, v_cfd, p_cfd, dx, dy, gradp_scale, interior,
        )
        opt_rmse_list.append(opt_rmse)
        print(f"  optimal RMSE @ cov={opt_actual_cov*100:.2f}%: {opt_rmse:.5f}")

    loss_curves = np.stack(loss_curves)         # (S, n_loss_points)
    rmse_curves = np.stack(rmse_curves)         # (S, 11)
    actual_cov_curves = np.stack(actual_cov_curves)  # (S, 11)

    np.savez(os.path.join(args.output_dir, 'errorbar_data.npz'),
             cov_grid=cov_grid,
             loss_curves=loss_curves,
             rmse_curves=rmse_curves,
             actual_cov_curves=actual_cov_curves,
             opt_cov=np.array(opt_cov_list),
             opt_loss=np.array(opt_loss_list),
             opt_rmse=np.array(opt_rmse_list),
             pinn_rmse=pinn_rmse,
             beta=args.beta)

    # ============================================================
    # PLOT 1: Abstention True Loss vs Coverage  (mean ± std)
    # ============================================================
    loss_mean = loss_curves.mean(axis=0)
    loss_std = loss_curves.std(axis=0)
    opt_cov_mean = float(np.mean(opt_cov_list))
    opt_cov_std = float(np.std(opt_cov_list))
    opt_loss_mean = float(np.mean(opt_loss_list))
    opt_loss_std = float(np.std(opt_loss_list))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(cov_grid * 100, loss_mean, '-', linewidth=LINEWIDTH,
            color=LINE_COLOR, zorder=2, label='Abstention true loss')
    ax.fill_between(cov_grid * 100,
                    loss_mean - loss_std, loss_mean + loss_std,
                    color=LINE_COLOR, alpha=ERR_ALPHA, zorder=1,
                    label=f'$\\pm$1 std ({n_seeds} seeds)')
    ax.errorbar([opt_cov_mean * 100], [opt_loss_mean],
                xerr=[opt_cov_std * 100], yerr=[opt_loss_std],
                fmt='none', ecolor='black', elinewidth=1.0,
                capsize=3, zorder=4)
    ax.plot(opt_cov_mean * 100, opt_loss_mean, '*',
            markersize=STAR_SIZE, color=STAR_COLOR,
            markeredgecolor='black', markeredgewidth=0.6, zorder=5,
            label=f'Optimal (cov={opt_cov_mean*100:.1f}%)')
    ax.set_xlabel('Coverage (% solved by CFD)', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(
        r'Abstention true loss  $\beta\,c + (1-c)\,\mathbb{E}[R\mid\mathrm{PINN}]$',
        fontsize=LABEL_FONTSIZE,
    )
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    ax.set_xlim(-3, 103)
    ax.legend(loc='upper right', frameon=False, fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    out_loss = os.path.join(args.output_dir, 'abstention_loss_vs_coverage.pdf')
    fig.savefig(out_loss, dpi=1200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {out_loss}")

    # ============================================================
    # PLOT 2: Hybrid RMSE vs Coverage  (mean ± std)
    # ============================================================
    rmse_mean = rmse_curves.mean(axis=0)
    rmse_std = rmse_curves.std(axis=0)
    cov_plot = actual_cov_curves.mean(axis=0)
    cov_plot_std = actual_cov_curves.std(axis=0)
    opt_rmse_mean = float(np.mean(opt_rmse_list))
    opt_rmse_std = float(np.std(opt_rmse_list))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(cov_plot * 100, rmse_mean, '-', linewidth=LINEWIDTH,
            color=LINE_COLOR, zorder=2, label='Hybrid RMSE')
    ax.errorbar(cov_plot * 100, rmse_mean,
                xerr=cov_plot_std * 100, yerr=rmse_std,
                fmt='o', markersize=MARKERSIZE, color=LINE_COLOR,
                ecolor=LINE_COLOR, elinewidth=1.0, capsize=2,
                zorder=3)
    ax.fill_between(cov_plot * 100,
                    rmse_mean - rmse_std, rmse_mean + rmse_std,
                    color=LINE_COLOR, alpha=ERR_ALPHA, zorder=1,
                    label=f'$\\pm$1 std ({n_seeds} seeds)')
    ax.errorbar([opt_cov_mean * 100], [opt_rmse_mean],
                xerr=[opt_cov_std * 100], yerr=[opt_rmse_std],
                fmt='none', ecolor='black', elinewidth=1.0,
                capsize=3, zorder=4)
    ax.plot(opt_cov_mean * 100, opt_rmse_mean, '*',
            markersize=STAR_SIZE, color=STAR_COLOR,
            markeredgecolor='black', markeredgewidth=0.6, zorder=5,
            label=f'Optimal (cov={opt_cov_mean*100:.1f}%)')
    ax.set_xlabel('Coverage (% solved by CFD)', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Hybrid RMSE (vs CFD)', fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    ax.set_xlim(-3, 103)
    ax.legend(loc='upper left', frameon=False, fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    out_rmse = os.path.join(args.output_dir, 'rmse_vs_coverage.pdf')
    fig.savefig(out_rmse, dpi=1200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_rmse}")


if __name__ == '__main__':
    main()
