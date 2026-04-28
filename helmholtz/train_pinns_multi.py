"""
Train one independent single-case PINN per config in a multi-cylinder-style
configs JSON. Each entry becomes one subprocess call to train_pinn.py (Exp 2
setup) with per-config (k, x_s, y_s). Weights land at the `pinn_path` given
in the JSON (tag is derived from basename).

Usage:
    python train_pinns_multi.py --configs configs_exp3b.json [--skip-existing]
        [--also-train-test] [--epochs 50000] [--dry-run]
"""

import argparse
import json
import os
import subprocess
import sys
import time


def tag_from_pinn_path(pinn_path):
    """./models/pinn_helmholtz_k2pi_p1.weights.h5 -> k2pi_p1"""
    base = os.path.basename(pinn_path)
    prefix = 'pinn_helmholtz_'
    suffix = '.weights.h5'
    assert base.startswith(prefix) and base.endswith(suffix), (
        f"pinn_path basename must match {prefix}<tag>{suffix}, got {base}")
    return base[len(prefix):-len(suffix)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--configs', type=str, required=True)
    p.add_argument('--skip-existing', action='store_true',
                   help='Skip configs whose pinn_path already exists.')
    p.add_argument('--also-train-test', action='store_true',
                   help='Train PINNs for entries under "test" as well.')
    p.add_argument('--epochs', type=int, default=50000)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--n-domain', type=int, default=20000)
    p.add_argument('--n-boundary', type=int, default=4000)
    p.add_argument('--w-bc', type=float, default=100.0)
    p.add_argument('--layers', type=int, nargs='+',
                   default=[256, 256, 256, 256, 256])
    p.add_argument('--fourier-m', type=int, default=128)
    p.add_argument('--sigma', type=float, default=0.05)
    p.add_argument('--amplitude', type=float, default=250.0)
    p.add_argument('--hole-center-x', type=float, default=0.5)
    p.add_argument('--hole-center-y', type=float, default=0.5)
    p.add_argument('--hole-radius', type=float, default=0.15)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--output-dir', type=str, default='./models')
    p.add_argument('--history-dir', type=str, default='./history')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    with open(args.configs) as f:
        cfg = json.load(f)

    train_list = cfg.get('train', [])
    test_list = cfg.get('test', [])
    targets = list(train_list)
    if args.also_train_test:
        targets += list(test_list)
    else:
        missing = [e for e in test_list
                   if not os.path.exists(e['pinn_path'])]
        if missing:
            print("WARNING: test entries have no weights on disk "
                  "(use --also-train-test to train them here):")
            for e in missing:
                print(f"  - {e['name']}  expected at {e['pinn_path']}")

    print(f"Will train {len(targets)} PINNs "
          f"(train={len(train_list)}"
          f"{', +test='+str(len(test_list)) if args.also_train_test else ''}).")

    t_start = time.perf_counter()
    for i, entry in enumerate(targets):
        name = entry['name']
        k = float(entry['k'])
        xs = float(entry['x_s']); ys = float(entry['y_s'])
        pinn_path = entry['pinn_path']
        tag = tag_from_pinn_path(pinn_path)

        print("\n" + "=" * 70)
        print(f"[{i+1}/{len(targets)}]  {name}  "
              f"k={k:.4f}  x_s={xs:.3f}  y_s={ys:.3f}  tag={tag}")
        print("=" * 70)

        if args.skip_existing and os.path.exists(pinn_path):
            print(f"  skip: weights exist at {pinn_path}")
            continue

        # PINNs are always trained against the global hole. Per-entry
        # `hole_center_*` fields in the JSON are TEST-time geometry overrides
        # consumed by train_router_multi.py — not by training.
        cmd = [
            sys.executable, 'train_pinn.py',
            '--k', f'{k:.10f}',
            '--domain', 'square_hole',
            '--hole-center-x', f'{args.hole_center_x}',
            '--hole-center-y', f'{args.hole_center_y}',
            '--hole-radius', f'{args.hole_radius}',
            '--source', 'gaussian',
            '--x-s', f'{xs:.10f}',
            '--y-s', f'{ys:.10f}',
            '--sigma', f'{args.sigma}',
            '--amplitude', f'{args.amplitude}',
            '--epochs', str(args.epochs),
            '--lr', f'{args.lr}',
            '--n-domain', str(args.n_domain),
            '--n-boundary', str(args.n_boundary),
            '--w-bc', f'{args.w_bc}',
            '--layers', *[str(w) for w in args.layers],
            '--fourier-m', str(args.fourier_m),
            '--grad-clip', f'{args.grad_clip}',
            '--tag', tag,
            '--output-dir', args.output_dir,
            '--history-dir', args.history_dir,
        ]
        print("  $ " + " ".join(cmd))
        if args.dry_run:
            continue

        t0 = time.perf_counter()
        rc = subprocess.call(cmd)
        elapsed = time.perf_counter() - t0
        print(f"  -> exit {rc}  ({elapsed:.1f}s)")
        if rc != 0:
            print(f"  FAILED: continuing to next config.")
        else:
            if not os.path.exists(pinn_path):
                print(f"  WARNING: expected {pinn_path} not found after run.")

    print(f"\nTotal wall-clock: {time.perf_counter() - t_start:.1f}s")


if __name__ == '__main__':
    main()
