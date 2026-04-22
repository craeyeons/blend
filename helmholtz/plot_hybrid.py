"""
Plot hybrid PINN+FEM results for Helmholtz Experiment 2.

Reads:
    {timing_dir}/timing_{tag}.json
    {timing_dir}/reference_{tag}.npz
    {router_history_dir}/router_meta_{tag}.json  (beta, lambda_tv)

Produces (under {plots_dir}):
    hybrid_{tag}.png                (6-panel summary: u, curves)
    solution_comparison_{tag}.png   (PINN | Hybrid | FEM, rejection outlined)
    loss_vs_coverage_{tag}.png      (training-loss curve with optimal star + bar breakdown)
    coverage_evolution_{tag}.png    (grid of hybrid u for 10%..100% rejection)
    metrics_exp2_{tag}.json

Usage:
    python plot_hybrid.py --tag exp2_hole_k4pi
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from lib.hybrid import solve_hybrid_schwarz, threshold_for_coverage, rel_l2
from lib.fem_solver import HelmholtzSolver
from lib.network import build_pinn
from lib.router import RouterCNN, create_router_input, median_normalize


# ---------------------------------------------------------------------------
# Setup: load solver, PINN, router once.
# ---------------------------------------------------------------------------

def _build_callables(summary):
    cx = float(summary.get('hole_center_x', 0.5))
    cy = float(summary.get('hole_center_y', 0.5))
    r_h = float(summary.get('hole_radius', 0.15))
    xs_ = float(summary['x_s']); ys_ = float(summary['y_s'])
    sigma = float(summary.get('sigma', 0.05))
    amp = float(summary.get('amplitude', 1.0))

    def dirichlet_predicate(x, y):
        return (x - cx) ** 2 + (y - cy) ** 2 <= r_h ** 2

    def f_callable(x, y):
        r2 = (x - xs_) ** 2 + (y - ys_) ** 2
        return amp * np.exp(-r2 / (2.0 * sigma ** 2))

    def g_callable(x, y):
        return np.zeros_like(x)

    return dirichlet_predicate, f_callable, g_callable


def _load_models(args, summary, ref):
    import tensorflow as tf

    k = float(summary['k'])
    layout = ref['layout']
    pinn_u = ref['pinn_u']
    residual = ref['residual']
    f_grid = ref['f_grid']
    ete = ref['ete'] if 'ete' in ref.files else None

    dirichlet_predicate, f_callable, g_callable = _build_callables(summary)

    solver = HelmholtzSolver(k=k, mesh_n=args.mesh_n,
                             dirichlet_predicate=dirichlet_predicate)

    with open(os.path.join(args.pinn_meta_dir,
                           f'training_meta_{args.tag}.json')) as fh:
        meta = json.load(fh)
    pinn = build_pinn(
        num_inputs=2,
        layers=tuple(meta.get('layers', [256, 256, 256, 256, 256])),
        activation='tanh',
        input_range=((0.0, 1.0), (0.0, 1.0)),
        fourier_m=int(meta.get('fourier_m', 64)),
        fourier_scale=float(meta.get('fourier_scale', k / (2 * np.pi))),
    )
    _ = pinn(tf.zeros((1, 2)))
    pinn.load_weights(os.path.join(
        args.pinn_dir, f'pinn_helmholtz_{args.tag}.weights.h5'))

    router = RouterCNN(base_filters=args.base_filters)
    dummy = create_router_input(layout, f_grid, pinn_u, residual, ete=ete)
    _ = router(tf.constant(dummy, dtype=tf.float32))
    router.load_weights(os.path.join(
        args.router_dir, f'router_helmholtz_{args.tag}.weights.h5'))

    return solver, pinn, router, f_callable, g_callable


# ---------------------------------------------------------------------------
# Plot 1: legacy 6-panel summary.
# ---------------------------------------------------------------------------

def _plot_summary(args, summary, ref, u_hybrid):
    X, Y = ref['X'], ref['Y']
    u_fem = ref['u_fem']
    layout = ref['layout']
    mask = layout > 0
    u_fem_m = np.where(mask, u_fem, np.nan)
    u_hyb_m = np.where(mask, u_hybrid, np.nan)
    err = np.where(mask, u_hybrid - u_fem, np.nan)

    sweep = summary['coverage_sweep']
    cov = np.array([s['actual_coverage_pct'] for s in sweep])
    errs = np.array([s['rel_l2_vs_fem'] for s in sweep])
    wall = np.array([s['hybrid_total_s'] for s in sweep])
    fem_mean = summary['fem_mean_s']
    speedup = fem_mean / np.maximum(wall, 1e-12)
    pinn_only = summary['pinn_rel_l2_vs_fem']

    fig = plt.figure(figsize=(18, 10))
    vmax = float(np.nanmax(np.abs(u_fem_m)))

    ax = fig.add_subplot(2, 3, 1)
    im = ax.pcolormesh(X, Y, u_fem_m, cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, shading='auto')
    ax.set_title('Full FEM  u'); ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(2, 3, 2)
    im = ax.pcolormesh(X, Y, u_hyb_m, cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, shading='auto')
    ax.set_title('Hybrid  u  (threshold=0)'); ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(2, 3, 3)
    ev = float(np.nanmax(np.abs(err))) + 1e-30
    im = ax.pcolormesh(X, Y, err, cmap='RdBu_r',
                       vmin=-ev, vmax=ev, shading='auto')
    rl = rel_l2(u_hybrid, u_fem)
    ax.set_title(f'u_hybrid - u_FEM  (rel L2 = {rl:.3e})')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(2, 3, 4)
    ax.plot(cov, errs, 'o-', label='Hybrid rel L2')
    ax.axhline(pinn_only, ls='--', color='C3',
               label=f'PINN-only ({pinn_only:.2e})')
    ax.set_xlabel('FEM coverage (%)'); ax.set_yscale('log')
    ax.set_ylabel('Rel L2 vs full FEM')
    ax.set_title('Accuracy vs coverage')
    ax.grid(True, alpha=0.3); ax.legend()

    ax = fig.add_subplot(2, 3, 5)
    ax.plot(cov, speedup, 'o-', color='C2')
    ax.axhline(1.0, ls='--', color='k', alpha=0.5, label='FEM baseline')
    ax.set_xlabel('FEM coverage (%)')
    ax.set_ylabel('Speedup  (FEM / hybrid wall)')
    ax.set_title('Speedup vs coverage')
    ax.grid(True, alpha=0.3); ax.legend()

    ax = fig.add_subplot(2, 3, 6)
    ax.plot(cov, wall * 1000, 'o-', color='C1', label='Hybrid total')
    ax.axhline(fem_mean * 1000, ls='--', color='k',
               label=f'FEM baseline ({fem_mean*1000:.1f} ms)')
    ax.set_xlabel('FEM coverage (%)')
    ax.set_ylabel('Wall time (ms)')
    ax.set_title('Wall time vs coverage')
    ax.grid(True, alpha=0.3); ax.legend()

    fig.suptitle(
        f'Helmholtz Exp 2  tag={args.tag}  k={summary["k"]:.3f}  '
        f'FEM={fem_mean*1000:.1f} ms  '
        f'hybrid@0={summary["hybrid_mean_s"]*1000:.1f} ms  '
        f'speedup={summary["speedup"]:.2f}x')
    fig.tight_layout()
    out = os.path.join(args.plots_dir, f'hybrid_{args.tag}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')
    return rl


# ---------------------------------------------------------------------------
# Plot 2: PINN | Hybrid | FEM side-by-side with rejection contour outline.
# ---------------------------------------------------------------------------

def _plot_solution_comparison(args, summary, ref, u_hybrid, accept_mask):
    X, Y = ref['X'], ref['Y']
    u_fem = ref['u_fem']
    layout = ref['layout']
    pinn_u = ref['pinn_u']
    mask = layout > 0

    def _masked(a): return np.where(mask, a, np.nan)
    u_fem_m = _masked(u_fem)
    u_hyb_m = _masked(u_hybrid)
    u_pinn_m = _masked(pinn_u)

    vmax = float(np.nanmax(np.abs(u_fem_m)))
    reject_field = (~accept_mask.astype(bool) & mask).astype(np.float32)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    # Row 1: solutions.
    for ax, data, title in zip(
            axes[0],
            [u_pinn_m, u_hyb_m, u_fem_m],
            ['PINN', 'Hybrid  (FEM region outlined)', 'Full FEM']):
        im = ax.pcolormesh(X, Y, data, cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax, shading='auto')
        ax.contour(X, Y, reject_field, levels=[0.5],
                   colors='lime', linewidths=1.5)
        ax.set_title(title); ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Row 2: signed errors vs full FEM.
    err_pinn = _masked(pinn_u - u_fem)
    err_hyb = _masked(u_hybrid - u_fem)
    err_fem = _masked(np.zeros_like(u_fem))
    all_errs = np.concatenate([np.abs(err_pinn[mask]),
                               np.abs(err_hyb[mask])])
    evmax = float(np.nanmax(all_errs)) + 1e-30
    for ax, data, title in zip(
            axes[1],
            [err_pinn, err_hyb, err_fem],
            ['PINN - FEM', 'Hybrid - FEM', 'FEM - FEM  (reference)']):
        im = ax.pcolormesh(X, Y, data, cmap='RdBu_r',
                           vmin=-evmax, vmax=evmax, shading='auto')
        ax.contour(X, Y, reject_field, levels=[0.5],
                   colors='lime', linewidths=1.5)
        ax.set_title(title); ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, fraction=0.046)

    rl_pinn = rel_l2(pinn_u, u_fem)
    rl_hyb = rel_l2(u_hybrid, u_fem)
    cov_pct = 100.0 * reject_field.sum() / max(mask.sum(), 1)
    fig.suptitle(
        f'Solution comparison  k={summary["k"]:.3f}  '
        f'FEM coverage={cov_pct:.1f}%  '
        f'PINN rel L2={rl_pinn:.2e}  Hybrid rel L2={rl_hyb:.2e}')
    fig.tight_layout()
    out = os.path.join(args.plots_dir, f'solution_comparison_{args.tag}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


# ---------------------------------------------------------------------------
# Plot 3: training-loss curve vs coverage with optimal star + bar breakdown.
# ---------------------------------------------------------------------------

def _training_loss_curve(logits, residual, layout, beta):
    """Evaluate the router's decision-version training loss at many thresholds.

    At inference, the router makes a hard accept/reject decision per cell.
    Replacing softplus(s), softplus(-s) with the indicators 1_{reject},
    1_{accept} gives the deployed loss:

        L(thr) = mean_solid [ beta * 1_{logit >= thr}
                              + R_bar * 1_{logit < thr} ].

    Returns (coverage_fracs, total_loss, reject_cost, accept_cost).
    """
    solid = layout > 0
    lv = logits[solid]
    r_bar = median_normalize(residual, layout)[solid]

    # Sweep thresholds at each sorted logit percentile to get a smooth curve.
    sorted_logits = np.sort(lv)
    thresholds = np.concatenate([
        [sorted_logits[0] - 1e-6],
        sorted_logits,
        [sorted_logits[-1] + 1e-6],
    ])
    # Remove duplicates for a monotone x-axis.
    thresholds = np.unique(thresholds)

    cov_fracs = np.empty_like(thresholds)
    reject_cost = np.empty_like(thresholds)
    accept_cost = np.empty_like(thresholds)
    for i, thr in enumerate(thresholds):
        rej = lv >= thr
        cov_fracs[i] = rej.mean()
        reject_cost[i] = beta * rej.mean()
        accept_cost[i] = (r_bar * (~rej)).mean()
    total = reject_cost + accept_cost
    return cov_fracs, total, reject_cost, accept_cost


def _plot_loss_vs_coverage(args, summary, ref, beta):
    logits = ref['logits']
    residual = ref['residual']
    layout = ref['layout']

    cov, total, rej_c, acc_c = _training_loss_curve(
        logits, residual, layout, beta)
    cov_pct = cov * 100.0

    # Endpoints: all-PINN (cov=0, loss = mean(R_bar)) and all-FEM (cov=1, loss=beta).
    solid = layout > 0
    r_bar_mean = float(median_normalize(residual, layout)[solid].mean())
    all_pinn = r_bar_mean
    all_fem = beta

    i_opt = int(np.argmin(total))
    opt_cov = cov_pct[i_opt]
    opt_loss = total[i_opt]
    opt_rej = rej_c[i_opt]
    opt_acc = acc_c[i_opt]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 6))

    axL.plot(cov_pct, total, color='royalblue', lw=2, label='Loss curve')
    axL.scatter([0.0], [all_pinn], color='purple', s=80, zorder=5)
    axL.scatter([100.0], [all_fem], color='teal', s=80, zorder=5)
    axL.axhline(all_pinn, ls='--', color='purple', alpha=0.6,
                label=f'All PINN: {all_pinn:.3f}')
    axL.axhline(all_fem, ls='--', color='teal', alpha=0.6,
                label=f'All FEM: {all_fem:.3f}')
    axL.scatter([opt_cov], [opt_loss], color='green', s=260, marker='*',
                edgecolor='darkgreen', zorder=6,
                label=f'Opt: {opt_cov:.1f}%  L={opt_loss:.3f}')
    axL.annotate(f'All PINN\n{all_pinn:.3f}', (0, all_pinn),
                 xytext=(5, 8), textcoords='offset points', color='purple')
    axL.annotate(f'All FEM\n{all_fem:.3f}', (100, all_fem),
                 xytext=(-80, 8), textcoords='offset points', color='teal')
    axL.set_xlabel('Coverage (% solved by FEM)')
    axL.set_ylabel('Training loss  (decision version)')
    axL.set_title(f'Training loss vs coverage  (\u03b2 = {beta})')
    axL.grid(True, alpha=0.3); axL.legend()

    bars = axR.bar(['FEM cost\n(\u03b2 · cov)', 'Residual cost', 'TOTAL'],
                   [opt_rej, opt_acc, opt_loss],
                   color=['steelblue', 'darkorange', 'forestgreen'],
                   edgecolor='black')
    for b, v in zip(bars, [opt_rej, opt_acc, opt_loss]):
        axR.text(b.get_x() + b.get_width() / 2, v + 0.01,
                 f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
    axR.set_ylabel('Loss value')
    axR.set_title(f'Loss breakdown at optimum ({opt_cov:.1f}% FEM)')
    axR.grid(True, axis='y', alpha=0.3)

    fig.suptitle(f'Router coverage metrics  tag={args.tag}  '
                 f'k={summary["k"]:.3f}')
    fig.tight_layout()
    out = os.path.join(args.plots_dir, f'loss_vs_coverage_{args.tag}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')
    return opt_cov, opt_loss


# ---------------------------------------------------------------------------
# Plot 4: hybrid-u grid across coverage levels 10%..100%.
# ---------------------------------------------------------------------------

def _plot_coverage_evolution(args, summary, ref, solver, pinn, router,
                             f_callable, g_callable):
    X, Y = ref['X'], ref['Y']
    u_fem = ref['u_fem']
    layout = ref['layout']
    pinn_u = ref['pinn_u']
    residual = ref['residual']
    f_grid = ref['f_grid']
    logits = ref['logits']
    mask = layout > 0
    vmax = float(np.nanmax(np.abs(np.where(mask, u_fem, np.nan))))

    covs = np.arange(0.1, 1.01, 0.1)  # 10% .. 100%
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    for ax, cov in zip(axes.ravel(), covs):
        thr = threshold_for_coverage(logits, layout, float(cov))
        res = solve_hybrid_schwarz(solver, pinn, router, f_callable, g_callable,
                                   X, Y, layout, f_grid, pinn_u, residual,
                                   ete_grid=(ref['ete'] if 'ete' in ref.files
                                             else None),
                                   threshold=thr, reuse_logits=logits)
        u = np.where(mask, res['u_grid'], np.nan)
        im = ax.pcolormesh(X, Y, u, cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax, shading='auto')
        reject_field = (~res['accept_mask'].astype(bool) & mask).astype(
            np.float32)
        ax.contour(X, Y, reject_field, levels=[0.5],
                   colors='lime', linewidths=1.0)
        errl2 = rel_l2(res['u_grid'], u_fem)
        ax.set_title(
            f'target {cov*100:.0f}%  actual {res["coverage_pct"]:.1f}%\n'
            f'rel L2 = {errl2:.2e}')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f'Hybrid solution vs FEM coverage  tag={args.tag}  '
                 f'k={summary["k"]:.3f}')
    fig.tight_layout()
    out = os.path.join(args.plots_dir, f'coverage_evolution_{args.tag}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', type=str, required=True)
    p.add_argument('--timing-dir', type=str, default='./timing')
    p.add_argument('--plots-dir', type=str, default='./plots')
    p.add_argument('--pinn-dir', type=str, default='./models')
    p.add_argument('--pinn-meta-dir', type=str, default='./history')
    p.add_argument('--router-dir', type=str, default='./router_models')
    p.add_argument('--router-history-dir', type=str, default='./router_history')
    p.add_argument('--mesh-n', type=int, default=129)
    p.add_argument('--base-filters', type=int, default=32)
    p.add_argument('--beta', type=float, default=None,
                   help='Override beta for loss plot; default reads router meta')
    args = p.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    with open(os.path.join(args.timing_dir, f'timing_{args.tag}.json')) as f:
        summary = json.load(f)
    ref = np.load(os.path.join(args.timing_dir, f'reference_{args.tag}.npz'))

    # Router beta (for loss plot).
    beta = args.beta
    if beta is None:
        rmeta_path = os.path.join(args.router_history_dir,
                                  f'router_meta_{args.tag}.json')
        if os.path.exists(rmeta_path):
            with open(rmeta_path) as f:
                beta = float(json.load(f).get('beta', 0.1))
        else:
            beta = 0.1

    # Load models once.
    solver, pinn, router, f_callable, g_callable = _load_models(
        args, summary, ref)

    # Hybrid at threshold=0 for comparison plots.
    X, Y = ref['X'], ref['Y']
    layout = ref['layout']
    pinn_u = ref['pinn_u']
    residual = ref['residual']
    f_grid = ref['f_grid']
    res0 = solve_hybrid_schwarz(solver, pinn, router, f_callable, g_callable,
                                X, Y, layout, f_grid, pinn_u, residual,
                                ete_grid=(ref['ete'] if 'ete' in ref.files
                                          else None),
                                threshold=0.0, reuse_logits=ref['logits'])
    u_hybrid0 = res0['u_grid']
    accept_mask0 = res0['accept_mask']

    # Plots.
    rl = _plot_summary(args, summary, ref, u_hybrid0)
    _plot_solution_comparison(args, summary, ref, u_hybrid0, accept_mask0)
    opt_cov, opt_loss = _plot_loss_vs_coverage(args, summary, ref, beta)
    _plot_coverage_evolution(args, summary, ref, solver, pinn, router,
                             f_callable, g_callable)

    metrics = {
        'tag': args.tag,
        'k': summary['k'],
        'fem_baseline_ms': summary['fem_mean_s'] * 1000,
        'hybrid_threshold0_ms': summary['hybrid_mean_s'] * 1000,
        'speedup_at_threshold0': summary['speedup'],
        'pinn_only_rel_l2': summary['pinn_rel_l2_vs_fem'],
        'hybrid_threshold0_rel_l2': float(rl),
        'beta': float(beta),
        'optimal_coverage_pct': float(opt_cov),
        'optimal_training_loss': float(opt_loss),
        'coverage_pct': [s['actual_coverage_pct']
                         for s in summary['coverage_sweep']],
        'hybrid_rel_l2': [s['rel_l2_vs_fem']
                          for s in summary['coverage_sweep']],
        'hybrid_wall_ms': [s['hybrid_total_s'] * 1000
                           for s in summary['coverage_sweep']],
    }
    out_json = os.path.join(args.plots_dir, f'metrics_exp2_{args.tag}.json')
    with open(out_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'Saved metrics: {out_json}')


if __name__ == '__main__':
    main()
