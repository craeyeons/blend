"""
Aggregate Table 2 (cylinder NSE summary) across all configs.

Iterates over every entry in `configs_multi_cylinder.json` (train + test),
runs `plot_coverage_metrics.py` for each one in its own output directory, then
reads the resulting `metrics_results.npz` files and assembles a single CSV +
LaTeX table with the columns reported in the paper:

    name | role | (x_c, y_c, r) | inlet | Re | PINN RMSE | Hybrid RMSE |
    Coverage | t_PINN | t_Hybrid | t_CFD

Usage:
    python compute_summary_table.py \
        --configs configs_multi_cylinder.json \
        --router-weights ./router_output_multi/router.weights.h5 \
        --output-root ./summary_table_runs \
        --beta 1.1

Pass --skip-runs to only aggregate from previously-completed output dirs.
"""

import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np


def config_name(entry, idx, role):
    """Stable per-row key derived from geometry + inlet."""
    return (f"{role}_{idx:02d}_x{entry['cylinder_x']}"
            f"_y{entry['cylinder_y']}_r{entry['cylinder_radius']}"
            f"_u{entry['inlet_velocity']}")


def run_one_config(entry, output_dir, args):
    """Invoke plot_coverage_metrics.py for a single config."""
    cmd = [
        sys.executable, 'plot_coverage_metrics.py',
        '--pinn-path', entry['pinn_path'],
        '--router-weights', args.router_weights,
        '--compute-cfd',
        '--save-cfd', os.path.join(output_dir, 'cfd.npz'),
        '--beta', str(args.beta),
        '--cylinder-x', str(entry['cylinder_x']),
        '--cylinder-y', str(entry['cylinder_y']),
        '--cylinder-radius', str(entry['cylinder_radius']),
        '--inlet-velocity', str(entry['inlet_velocity']),
        '--Re', str(entry.get('Re', args.Re)),
        '--output-dir', output_dir,
    ]
    print('\n' + '=' * 70)
    print('Running:', ' '.join(cmd))
    print('=' * 70)
    subprocess.run(cmd, check=True)


def read_one_npz(path):
    """Pull the columns we need for the table."""
    z = np.load(path, allow_pickle=False)
    keys = ['cfg_cylinder_x', 'cfg_cylinder_y', 'cfg_cylinder_radius',
            'cfg_inlet_velocity', 'cfg_Re', 'cfg_beta',
            'pinn_rmse', 'hybrid_rmse',
            'full_loss_optimal_coverage', 'full_loss_optimal_threshold',
            'pinn_inference_time', 'hybrid_solve_time', 'cfd_time']
    out = {}
    for k in keys:
        if k in z.files:
            out[k] = float(z[k])
        else:
            out[k] = float('nan')
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--configs', type=str, default='configs_multi_cylinder.json')
    p.add_argument('--router-weights', type=str,
                   default='./router_output_multi/router.weights.h5')
    p.add_argument('--output-root', type=str, default='./summary_table_runs')
    p.add_argument('--beta', type=float, default=1.1)
    p.add_argument('--Re', type=float, default=100.0,
                   help='Default Re if a config does not specify one.')
    p.add_argument('--skip-runs', action='store_true',
                   help='Only aggregate from existing output dirs; do not run '
                        'plot_coverage_metrics.py.')
    p.add_argument('--include-test', action='store_true', default=True,
                   help='Include the test split (default true).')
    p.add_argument('--include-train', action='store_true', default=True,
                   help='Include the train split (default true).')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)

    with open(args.configs) as fp:
        cfgs = json.load(fp)

    rows = []
    splits = []
    if args.include_train:
        splits.append(('train', cfgs.get('train', [])))
    if args.include_test:
        splits.append(('test', cfgs.get('test', [])))

    for role, entries in splits:
        for idx, entry in enumerate(entries):
            name = config_name(entry, idx, role)
            out_dir = os.path.join(args.output_root, name)
            os.makedirs(out_dir, exist_ok=True)

            if not args.skip_runs:
                run_one_config(entry, out_dir, args)

            npz_path = os.path.join(out_dir, 'metrics_results.npz')
            if not os.path.exists(npz_path):
                print(f'[skip] {name}: no metrics_results.npz')
                continue
            data = read_one_npz(npz_path)
            data['name'] = name
            data['role'] = role
            rows.append(data)

    if not rows:
        print('No rows aggregated.')
        return

    # Write CSV.
    csv_path = os.path.join(args.output_root, 'summary_table.csv')
    fields = ['name', 'role',
              'cfg_cylinder_x', 'cfg_cylinder_y', 'cfg_cylinder_radius',
              'cfg_inlet_velocity', 'cfg_Re', 'cfg_beta',
              'pinn_rmse', 'hybrid_rmse',
              'full_loss_optimal_coverage', 'full_loss_optimal_threshold',
              'pinn_inference_time', 'hybrid_solve_time', 'cfd_time']
    with open(csv_path, 'w', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})
    print(f'\nWrote {csv_path}')

    # Write LaTeX table fragment.
    tex_path = os.path.join(args.output_root, 'summary_table.tex')
    with open(tex_path, 'w') as fp:
        fp.write('% Auto-generated by compute_summary_table.py\n')
        fp.write('\\begin{tabular}{llccccccc}\n')
        fp.write('\\toprule\n')
        fp.write('Role & $(x_c, y_c, r)$ & $u_0$ & Re & '
                 'PINN RMSE & Hybrid RMSE & Cov & '
                 '$t_{\\text{PINN}}$ & $t_{\\text{Hyb}}$ / $t_{\\text{CFD}}$ \\\\\n')
        fp.write('\\midrule\n')
        for r in rows:
            fp.write(
                f"{r['role']} & "
                f"({r['cfg_cylinder_x']:g}, {r['cfg_cylinder_y']:g}, "
                f"{r['cfg_cylinder_radius']:g}) & "
                f"{r['cfg_inlet_velocity']:g} & "
                f"{r['cfg_Re']:g} & "
                f"{r['pinn_rmse']:.3e} & {r['hybrid_rmse']:.3e} & "
                f"{r['full_loss_optimal_coverage']*100:.1f}\\% & "
                f"{r['pinn_inference_time']:.2f}s & "
                f"{r['hybrid_solve_time']:.2f}s / {r['cfd_time']:.2f}s \\\\\n"
            )
        fp.write('\\bottomrule\n\\end{tabular}\n')
    print(f'Wrote {tex_path}')

    # Print to stdout.
    print('\nSummary:')
    print(f"{'name':<45} {'role':<6} {'pinn_rmse':>10} {'hyb_rmse':>10} "
          f"{'cov':>6} {'t_pinn':>8} {'t_hyb':>8} {'t_cfd':>8}")
    for r in rows:
        print(f"{r['name']:<45} {r['role']:<6} "
              f"{r['pinn_rmse']:>10.3e} {r['hybrid_rmse']:>10.3e} "
              f"{r['full_loss_optimal_coverage']*100:>5.1f}% "
              f"{r['pinn_inference_time']:>7.2f}s "
              f"{r['hybrid_solve_time']:>7.2f}s "
              f"{r['cfd_time']:>7.2f}s")


if __name__ == '__main__':
    main()
