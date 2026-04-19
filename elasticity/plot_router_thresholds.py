"""
Plot router separation at multiple thresholds for elasticity problems.

Usage:
    python plot_router_thresholds.py --problem plate_with_hole \
        --router-path ./router_output/plate_with_hole/beta_0.1/router.weights.h5 \
            --pinn-path ./models/pinn_plate_with_hole.weights.h5
"""

import argparse
import os
import numpy as np
import cv2
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

import matplotlib.pyplot as plt
try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
except ImportError:
    pass
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

from lib.network import Network
from lib.domains import create_plate_with_hole, create_l_bracket
from train_pinn_decomposed import load_decomposed_pinn, blend_solutions
from lib.router import (
    RouterCNN,
    create_router_input,
    compute_bc_error_field,
    solve_error_transport,
)


def apply_morph_open(mask, layout, kernel_size):
    """Apply morphological opening to smooth the binary mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return opened.astype(np.int32) * layout.astype(np.int32)


def plot_separation_at_threshold(r, X, Y, layout, threshold,
                                  show_hole, save_path, morph_kernel=5):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Continuous output with threshold line
    ax = axes[0]
    r_masked = np.ma.masked_where(layout == 0, r)
    cf = ax.contourf(X, Y, r_masked, levels=50, cmap='RdBu_r',
                     norm=Normalize(vmin=0, vmax=1))
    ax.contour(X, Y, r_masked, levels=[threshold], colors='green',
               linewidths=2, linestyles='--')
    plt.colorbar(cf, ax=ax, label='Router Score')
    if show_hole:
        ax.add_patch(plt.Circle(show_hole[:2], show_hole[2], color='gray', fill=True))
    ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(f'Continuous (threshold={threshold:.2f})')

    # 2. Binary mask (with morphological opening)
    ax = axes[1]
    mask = (r >= threshold).astype(np.int32) * layout.astype(np.int32)
    mask = apply_morph_open(mask, layout, morph_kernel)
    combined = np.zeros_like(r)
    combined[layout == 0] = 0
    combined[(layout == 1) & (mask == 0)] = 1
    combined[(layout == 1) & (mask == 1)] = 2
    ax.contourf(X, Y, combined, levels=[-0.5, 0.5, 1.5, 2.5],
                colors=['gray', 'blue', 'red'], alpha=0.7)
    legend_elements = [
        Patch(facecolor='gray', label='Void'),
        Patch(facecolor='blue', alpha=0.7, label='PINN'),
        Patch(facecolor='red', alpha=0.7, label='FDM'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    if show_hole:
        ax.add_patch(plt.Circle(show_hole[:2], show_hole[2], color='gray', fill=True))
    ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')
    fdm_frac = np.sum(mask) / np.sum(layout == 1) * 100
    ax.set_title(f'Binary (threshold={threshold:.2f}, FDM={fdm_frac:.1f}%)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, default='plate_with_hole',
                        choices=['plate_with_hole', 'l_bracket'])
    parser.add_argument('--router-path', type=str,
                        default='./router_output/plate_with_hole/beta_0.1/router.weights.h5')
    parser.add_argument('--pinn-path', type=str,
                        default='./models/pinn_plate_with_hole.weights.h5')
    parser.add_argument('--pinn-vbar-path', type=str, default=None,
                        help='V-bar PINN weights for decomposed L-bracket')
    parser.add_argument('--pinn-hbar-path', type=str, default=None,
                        help='H-bar PINN weights for decomposed L-bracket')
    parser.add_argument('--output-dir', type=str, default='./threshold_plots')

    parser.add_argument('--threshold-start', type=float, default=0.01)
    parser.add_argument('--threshold-end', type=float, default=0.11)
    parser.add_argument('--num-thresholds', type=int, default=11)

    parser.add_argument('--nx', type=int, default=200)
    parser.add_argument('--ny', type=int, default=200)
    parser.add_argument('--x-min', type=float, default=-2.0)
    parser.add_argument('--x-max', type=float, default=2.0)
    parser.add_argument('--y-min', type=float, default=-2.0)
    parser.add_argument('--y-max', type=float, default=2.0)
    parser.add_argument('--hole-x', type=float, default=0.0)
    parser.add_argument('--hole-y', type=float, default=0.0)
    parser.add_argument('--hole-radius', type=float, default=0.5)
    parser.add_argument('--applied-stress', type=float, default=10.0)
    parser.add_argument('--corner-x', type=float, default=1.0)
    parser.add_argument('--corner-y', type=float, default=1.0)
    parser.add_argument('--fillet-radius', type=float, default=0.04,
                        help='Fillet radius at L-bracket re-entrant corner (0=sharp)')
    parser.add_argument('--E', type=float, default=1.0)
    parser.add_argument('--nu', type=float, default=0.3)
    parser.add_argument('--layers', type=int, nargs='+', default=[128, 128, 128, 128])
    parser.add_argument('--base-filters', type=int, default=32)
    parser.add_argument('--morph-kernel', type=int, default=5,
                        help='Kernel size for morphological opening of mask')

    args = parser.parse_args()

    if args.problem == 'l_bracket' and args.x_min == -2.0:
        args.x_min, args.y_min = 0.0, 0.0

    os.makedirs(args.output_dir, exist_ok=True)

    # Domain
    if args.problem == 'plate_with_hole':
        X, Y, layout, dbc, tbc, bux, buy, btx, bty = create_plate_with_hole(
            Nx=args.nx, Ny=args.ny,
            x_domain=(args.x_min, args.x_max), y_domain=(args.y_min, args.y_max),
            hole_center=(args.hole_x, args.hole_y), hole_radius=args.hole_radius,
            applied_stress=args.applied_stress)
        show_hole = (args.hole_x, args.hole_y, args.hole_radius)
    else:
        X, Y, layout, dbc, tbc, bux, buy, btx, bty = create_l_bracket(
            Nx=args.nx, Ny=args.ny,
            x_domain=(args.x_min, args.x_max), y_domain=(args.y_min, args.y_max),
            corner_x=args.corner_x, corner_y=args.corner_y,
            applied_stress=args.applied_stress,
            fillet_radius=args.fillet_radius)
        show_hole = None

    # PINN
    print("Loading PINN...")
    use_decomposed = (args.pinn_vbar_path is not None and
                      args.pinn_hbar_path is not None)
    if use_decomposed:
        model_v, model_h = load_decomposed_pinn(
            args.pinn_vbar_path, args.pinn_hbar_path,
            layers=args.layers, activation='tanh',
            x_min=args.x_min, x_max=args.x_max,
            y_min=args.y_min, y_max=args.y_max,
            corner_x=args.corner_x, corner_y=args.corner_y)
        pinn_ux, pinn_uy = blend_solutions(
            model_v, model_h, X, Y, layout,
            args.corner_x, args.corner_y)
    else:
        network = Network()
        input_range = [(args.x_min, args.x_max), (args.y_min, args.y_max)]
        hard_bc_params = None
        if args.problem == 'l_bracket':
            hard_bc_params = {'corner_x': args.corner_x, 'corner_y': args.corner_y}
        pinn_model = network.build(num_inputs=2, layers=args.layers,
                                   activation='tanh', num_outputs=2,
                                   input_range=input_range,
                                   hard_bc=args.problem,
                                   hard_bc_params=hard_bc_params)
        pinn_model.load_weights(args.pinn_path)
        xy = np.stack([X.flatten(), Y.flatten()], axis=-1).astype(np.float32)
        out = pinn_model.predict(xy, batch_size=len(xy), verbose=0)
        pinn_ux = out[:, 0].reshape(X.shape).astype(np.float32) * layout
        pinn_uy = out[:, 1].reshape(X.shape).astype(np.float32) * layout

    # Von Mises from PINN
    dx = (args.x_max - args.x_min) / (args.nx - 1)
    dy = (args.y_max - args.y_min) / (args.ny - 1)
    from lib.solver import compute_stress_field
    _, _, _, pinn_vm = compute_stress_field(
        pinn_ux, pinn_uy, layout, dx, dy, E=args.E, nu=args.nu)
    pinn_vm = pinn_vm.astype(np.float32) * layout

    bc_error = compute_bc_error_field(dbc, bux, buy, pinn_ux, pinn_uy, layout)
    error_transport = solve_error_transport(
        bc_error, layout, E=args.E, nu=args.nu,
        x_domain=(args.x_min, args.x_max),
        y_domain=(args.y_min, args.y_max),
    )

    inputs = create_router_input(layout, dbc, bux, buy, tbc,
                                  pinn_ux, pinn_uy, pinn_vm, error_transport)

    # Router
    print("Loading router...")
    router = RouterCNN(base_filters=args.base_filters)
    _ = router(inputs)
    router.load_weights(args.router_path)

    r = router(tf.constant(inputs, dtype=tf.float32), training=False)
    r = r[0, :, :, 0].numpy()

    print(f"Router output: min={r.min():.6f}, max={r.max():.6f}, mean={r.mean():.6f}")

    # Thresholds
    thresholds = np.linspace(args.threshold_start, args.threshold_end,
                             args.num_thresholds)
    print(f"Plotting {len(thresholds)} thresholds...")

    for t in thresholds:
        path = os.path.join(args.output_dir, f'threshold_{t:.4f}.png')
        plot_separation_at_threshold(r, X, Y, layout, t, show_hole, path,
                                     morph_kernel=args.morph_kernel)

    # Summary grid
    n_cols = 4
    n_rows = (len(thresholds) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, t in enumerate(thresholds):
        ax = axes[i]
        mask = (r >= t).astype(np.int32) * layout.astype(np.int32)
        mask = apply_morph_open(mask, layout, args.morph_kernel)
        combined = np.zeros_like(r)
        combined[layout == 0] = 0
        combined[(layout == 1) & (mask == 0)] = 1
        combined[(layout == 1) & (mask == 1)] = 2
        ax.contourf(X, Y, combined, levels=[-0.5, 0.5, 1.5, 2.5],
                    colors=['gray', 'blue', 'red'], alpha=0.7)
        if show_hole:
            ax.add_patch(plt.Circle(show_hole[:2], show_hole[2], color='gray', fill=True))
        ax.set_aspect('equal')
        fdm_frac = np.sum(mask) / np.sum(layout == 1) * 100
        ax.set_title(f't={t:.2f}, FDM={fdm_frac:.1f}%')
        ax.set_xticks([]); ax.set_yticks([])

    for j in range(len(thresholds), len(axes)):
        axes[j].axis('off')

    legend_elements = [Patch(facecolor='gray', label='Void'),
                       Patch(facecolor='blue', alpha=0.7, label='PINN'),
                       Patch(facecolor='red', alpha=0.7, label='FDM')]
    fig.legend(handles=legend_elements, loc='lower right', fontsize=12)
    plt.suptitle(f'Router Thresholds ({args.problem})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'summary_all_thresholds.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nDone! Saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
