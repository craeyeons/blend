"""
Plot RMSE-vs-coverage for the held-out cylinder test.

Reads the actual-hybrid sweep from time_cylinder.py's timing_sweep.npz, splices
in the optimal-threshold point so the line passes through the star, and
renders with SciencePlots styling.

Usage:
    python plot_test_rmse_vs_coverage.py \
        --run-dir ./solution_test_rerun \
        --output  ./solution_test_rerun/rmse_vs_coverage.png

If --run-dir is omitted, the hard-coded sweep below (from the held-out test
log) is used so this script is self-contained.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (registers the 'science' style)

plt.style.use(['science', 'no-latex'])


# ---- Hard-coded fallback data (from the held-out test slurm log). ----
# Fill in the missing 10%–60% values when you have them; the script will use
# whatever is present and still pass the line through the star.
FALLBACK_SWEEP_COV = np.array([
    0.000,   # PINN only
    0.097,   # ~10%
    0.198,   # ~20%
    0.298,   # ~30%
    0.399,   # ~40%
    0.500,   # ~50%
    0.600,   # ~60%
    0.700,   # ~70%
    0.799,   # ~80%
    0.900,   # ~90%
    1.000,   # CFD only
])
FALLBACK_SWEEP_RMSE = np.array([
    0.1212,
    0.124339,
    0.099376,
    0.078327,
    0.057381,
    0.032783,
    0.032102,
    0.072165,
    0.084395,
    0.334774,
    0.000,   # CFD ground truth
])
FALLBACK_OPT_COV = 0.4766
FALLBACK_OPT_RMSE = 0.022706
FALLBACK_BETA = 1.1


def load_sweep(run_dir):
    sweep_path = os.path.join(run_dir, 'timing_sweep.npz')
    metrics_path = os.path.join(run_dir, 'metrics_results.npz')
    sweep = np.load(sweep_path)
    metrics = np.load(metrics_path)
    cov = np.asarray(sweep['coverage'], dtype=float)
    rmse = np.asarray(sweep['rmse'], dtype=float)
    opt_cov = float(metrics['full_loss_optimal_coverage'])
    opt_rmse = float(metrics['hybrid_rmse'])
    beta = float(metrics['cfg_beta']) if 'cfg_beta' in metrics.files else 1.1
    return cov, rmse, opt_cov, opt_rmse, beta


def splice_optimal(cov, rmse, opt_cov, opt_rmse):
    """Insert (opt_cov, opt_rmse) into sorted (cov, rmse) so the line passes
    through it, removing any duplicate at the same coverage."""
    keep = np.abs(cov - opt_cov) > 1e-6
    cov = np.append(cov[keep], opt_cov)
    rmse = np.append(rmse[keep], opt_rmse)
    order = np.argsort(cov)
    return cov[order], rmse[order]


def plot(cov, rmse, opt_cov, opt_rmse, beta, out_path):
    fig, ax = plt.subplots(figsize=(6.0, 4.2))

    ax.plot(cov * 100, rmse, '-', linewidth=1.6, color='tab:blue',
            zorder=2, label='Hybrid sweep')
    # Markers for the sweep points (excluding the optimal — that gets a star).
    sweep_mask = np.abs(cov - opt_cov) > 1e-6
    ax.plot(cov[sweep_mask] * 100, rmse[sweep_mask], 'o',
            markersize=4.5, color='tab:blue', zorder=3)
    # Optimal star, on the line.
    ax.plot(opt_cov * 100, opt_rmse, '*', markersize=14, color='tab:green',
            markeredgecolor='black', markeredgewidth=0.6, zorder=4,
            label=fr'Optimal $\tau^\star$ ({opt_cov*100:.1f}\%)'
                  if plt.rcParams.get('text.usetex')
                  else f'Optimal τ* ({opt_cov*100:.1f}%)')

    # Reference lines for the two endpoints.
    ax.axhline(rmse[0], color='purple', linestyle='--', linewidth=1.0,
               alpha=0.6, label=f'PINN only: {rmse[0]:.3f}')
    ax.axhline(rmse[-1], color='teal', linestyle='--', linewidth=1.0,
               alpha=0.6, label=f'CFD only: {rmse[-1]:.3f}')

    ax.set_xlabel('Coverage (\\% solved by CFD)' if plt.rcParams.get('text.usetex')
                  else 'Coverage (% solved by CFD)')
    ax.set_ylabel('Hybrid RMSE (vs CFD)')
    title = (f'RMSE vs Coverage  (held-out test, $\\beta={beta}$)'
             if plt.rcParams.get('text.usetex')
             else f'RMSE vs Coverage  (held-out test, β = {beta})')
    ax.set_title(title)
    ax.set_xlim(-3, 103)
    ymin = float(min(rmse.min(), opt_rmse))
    ymax = float(rmse.max())
    pad = 0.05 * (ymax - ymin if ymax > ymin else max(ymax, 1e-3))
    ax.set_ylim(max(0.0, ymin - pad), ymax + pad)
    ax.legend(loc='upper left', frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=str, default=None,
                   help='Dir containing timing_sweep.npz + metrics_results.npz.')
    p.add_argument('--output', type=str,
                   default='./rmse_vs_coverage_test.png')
    args = p.parse_args()

    if args.run_dir and os.path.isdir(args.run_dir):
        cov, rmse, opt_cov, opt_rmse, beta = load_sweep(args.run_dir)
    else:
        cov, rmse = FALLBACK_SWEEP_COV, FALLBACK_SWEEP_RMSE
        opt_cov, opt_rmse, beta = (FALLBACK_OPT_COV,
                                   FALLBACK_OPT_RMSE,
                                   FALLBACK_BETA)
        print('Using hard-coded fallback data (no --run-dir).')

    cov, rmse = splice_optimal(cov, rmse, opt_cov, opt_rmse)
    plot(cov, rmse, opt_cov, opt_rmse, beta, args.output)


if __name__ == '__main__':
    main()
