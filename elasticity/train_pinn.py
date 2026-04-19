"""
Train a PINN for 2D static linear elasticity.

Supports:
    - plate_with_hole: Rectangular plate with circular hole under tension
    - l_bracket: L-shaped bracket under load

Usage:
    python train_pinn.py --problem plate_with_hole --epochs 10000
    python train_pinn.py --problem l_bracket --epochs 10000
"""

import argparse
import os
import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

from lib.network import Network
from lib.domains import create_plate_with_hole, create_l_bracket


def create_training_data(problem, n_domain=10000, n_boundary=2000, args=None):
    """
    Create collocation points and boundary data for PINN training.

    Returns
    -------
    xy_domain : ndarray (n_domain, 2)
        Interior collocation points.
    xy_bc_disp : ndarray (n_disp, 2)
        Displacement BC points.
    bc_disp_vals : ndarray (n_disp, 2)
        Prescribed displacements [ux, uy].
    xy_bc_trac : ndarray (n_trac, 2)
        Traction BC points.
    bc_trac_vals : ndarray (n_trac, 2)
        Prescribed tractions [tx, ty].
    bc_trac_normals : ndarray (n_trac, 2)
        Outward normals at traction points.
    """
    def _sample_rows(arr, n):
        if len(arr) == 0:
            return arr
        replace = len(arr) < n
        idx = np.random.choice(len(arr), size=n, replace=replace)
        return arr[idx]

    if problem == 'plate_with_hole':
        X, Y, layout, disp_bc_mask, trac_bc_mask, bc_ux, bc_uy, bc_tx, bc_ty = \
            create_plate_with_hole(
                Nx=200, Ny=200,
                x_domain=(args.x_min, args.x_max),
                y_domain=(args.y_min, args.y_max),
                hole_center=(args.hole_x, args.hole_y),
                hole_radius=args.hole_radius,
                applied_stress=args.applied_stress,
            )
    elif problem == 'l_bracket':
        X, Y, layout, disp_bc_mask, trac_bc_mask, bc_ux, bc_uy, bc_tx, bc_ty = \
            create_l_bracket(
                Nx=200, Ny=200,
                x_domain=(args.x_min, args.x_max),
                y_domain=(args.y_min, args.y_max),
                corner_x=args.corner_x,
                corner_y=args.corner_y,
                applied_stress=args.applied_stress,
                fillet_radius=args.fillet_radius,
            )
    else:
        raise ValueError(f"Unknown problem: {problem}")

    # PDE collocation points: sample from interior material points
    interior_mask = (layout > 0) & (disp_bc_mask == 0) & (trac_bc_mask == 0)
    iy, ix = np.where(interior_mask)
    xy_interior = np.stack([X[iy, ix], Y[iy, ix]], axis=-1).astype(np.float32)
    xy_domain = _sample_rows(xy_interior, n_domain).astype(np.float32)

    # Displacement BC points/values from the exact same masks used by FDM
    dy_disp, dx_disp = np.where((layout > 0) & (disp_bc_mask > 0))
    xy_disp_all = np.stack([X[dy_disp, dx_disp], Y[dy_disp, dx_disp]], axis=-1).astype(np.float32)
    bc_disp_all = np.stack([bc_ux[dy_disp, dx_disp], bc_uy[dy_disp, dx_disp]], axis=-1).astype(np.float32)

    if len(xy_disp_all) > 0:
        replace = len(xy_disp_all) < max(1, n_boundary)
        idx_disp = np.random.choice(len(xy_disp_all), size=max(1, n_boundary), replace=replace)
        xy_bc_disp = xy_disp_all[idx_disp].astype(np.float32)
        bc_disp_vals = bc_disp_all[idx_disp].astype(np.float32)
    else:
        xy_bc_disp = np.zeros((0, 2), dtype=np.float32)
        bc_disp_vals = np.zeros((0, 2), dtype=np.float32)

    # Traction BC points/values from the exact same masks used by FDM
    dy_trac, dx_trac = np.where((layout > 0) & (trac_bc_mask > 0))
    xy_trac_all = np.stack([X[dy_trac, dx_trac], Y[dy_trac, dx_trac]], axis=-1).astype(np.float32)
    bc_trac_all = np.stack([bc_tx[dy_trac, dx_trac], bc_ty[dy_trac, dx_trac]], axis=-1).astype(np.float32)

    if len(xy_trac_all) > 0:
        replace = len(xy_trac_all) < max(1, n_boundary)
        idx_trac = np.random.choice(len(xy_trac_all), size=max(1, n_boundary), replace=replace)
        xy_bc_trac = xy_trac_all[idx_trac].astype(np.float32)
        bc_trac_vals = bc_trac_all[idx_trac].astype(np.float32)
    else:
        xy_bc_trac = np.zeros((0, 2), dtype=np.float32)
        bc_trac_vals = np.zeros((0, 2), dtype=np.float32)

    # Approximate outward normals on sampled traction points
    eps_x = (X[0, 1] - X[0, 0]) * 0.6
    eps_y = (Y[1, 0] - Y[0, 0]) * 0.6
    normals = np.zeros_like(xy_bc_trac, dtype=np.float32)

    if problem == 'plate_with_hole' and len(xy_bc_trac) > 0:
        x = xy_bc_trac[:, 0]
        y = xy_bc_trac[:, 1]
        x_min, x_max = args.x_min, args.x_max
        y_min, y_max = args.y_min, args.y_max
        cx, cy = args.hole_x, args.hole_y
        R = args.hole_radius

        on_right = np.abs(x - x_max) < eps_x
        on_top = np.abs(y - y_max) < eps_y
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        on_hole = np.abs(dist - R) < max(eps_x, eps_y) * 1.5

        normals[on_right, 0] = 1.0
        normals[on_top, 1] = 1.0
        hole_idx = on_hole & (~on_right) & (~on_top)
        if np.any(hole_idx):
            vx = x[hole_idx] - cx
            vy = y[hole_idx] - cy
            vn = np.sqrt(vx ** 2 + vy ** 2) + 1e-12
            normals[hole_idx, 0] = vx / vn
            normals[hole_idx, 1] = vy / vn

        # fallback for any unclassified points
        unclassified = np.linalg.norm(normals, axis=1) < 1e-8
        normals[unclassified, 1] = 1.0

    elif problem == 'l_bracket' and len(xy_bc_trac) > 0:
        x = xy_bc_trac[:, 0]
        y = xy_bc_trac[:, 1]
        x_min, x_max = args.x_min, args.x_max
        y_max = args.y_max
        cx, cy = args.corner_x, args.corner_y
        R = getattr(args, 'fillet_radius', 0.0)

        on_right_lower = (np.abs(x - x_max) < eps_x) & (y <= cy + eps_y)
        on_left = np.abs(x - x_min) < eps_x
        on_top = (np.abs(y - y_max) < eps_y) & (x <= cx + eps_x)
        on_inner_h = (np.abs(y - cy) < eps_y) & (x >= cx - eps_x)
        on_inner_v = (np.abs(x - cx) < eps_x) & (y >= cy - eps_y)

        # Fillet arc: points near the quarter-circle centered at (cx+R, cy+R)
        if R > 0:
            fcx, fcy = cx + R, cy + R
            dist_fc = np.sqrt((x - fcx) ** 2 + (y - fcy) ** 2)
            on_fillet = ((x >= cx - eps_x) & (x <= cx + R + eps_x) &
                         (y >= cy - eps_y) & (y <= cy + R + eps_y) &
                         (np.abs(dist_fc - R) < max(eps_x, eps_y) * 1.5))
            # Exclude points already matched to straight edges
            on_fillet = on_fillet & (~on_inner_h) & (~on_inner_v)
            # Trim the straight inner edges at the fillet tangent points
            on_inner_h = on_inner_h & (x >= cx + R + eps_x)
            on_inner_v = on_inner_v & (y >= cy + R + eps_y)
        else:
            on_fillet = np.zeros_like(x, dtype=bool)

        normals[on_right_lower, 0] = 1.0
        normals[on_left, 0] = -1.0
        normals[on_top, 1] = 1.0
        normals[on_inner_h, 1] = 1.0
        normals[on_inner_v, 0] = 1.0

        # Fillet arc: outward normal points toward arc center (concave)
        if np.any(on_fillet):
            vx = (cx + R) - x[on_fillet]
            vy = (cy + R) - y[on_fillet]
            vn = np.sqrt(vx ** 2 + vy ** 2) + 1e-12
            normals[on_fillet, 0] = vx / vn
            normals[on_fillet, 1] = vy / vn

        unclassified = np.linalg.norm(normals, axis=1) < 1e-8
        normals[unclassified, 1] = 1.0

    bc_trac_normals = normals.astype(np.float32)

    return (xy_domain.astype(np.float32), xy_bc_disp.astype(np.float32), bc_disp_vals.astype(np.float32),
            xy_bc_trac.astype(np.float32), bc_trac_vals.astype(np.float32), bc_trac_normals.astype(np.float32))


class ElasticityPINNTrainer:
    """
    PINN trainer for 2D plane-stress linear elasticity.

    Loss = w_pde * L_equilibrium + w_disp * L_disp_bc + w_trac * L_trac_bc

    Equilibrium:
        d(sigma_xx)/dx + d(sigma_xy)/dy = 0
        d(sigma_xy)/dx + d(sigma_yy)/dy = 0

    Constitutive (plane stress):
        sigma_xx = C11 * dux/dx + C12 * duy/dy
        sigma_yy = C12 * dux/dx + C11 * duy/dy
        sigma_xy = C66 * (dux/dy + duy/dx)
    """

    def __init__(self, model, E=1.0, nu=0.3, lr=1e-3, epochs=10000,
                 w_pde=1.0, w_disp=10.0, w_trac=1.0,
                 grad_clip_norm=1.0):
        self.model = model
        self.E = E
        self.nu = nu
        self.C11 = E / (1.0 - nu ** 2)
        self.C12 = nu * E / (1.0 - nu ** 2)
        self.C66 = E / (2.0 * (1.0 + nu))
        self.w_pde = w_pde
        self.w_disp = w_disp
        self.w_trac = w_trac
        self.grad_clip_norm = grad_clip_norm

        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=epochs,
            alpha=1e-2,  # final lr = lr * 1e-2
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.history = {'total': [], 'pde': [], 'disp_bc': [], 'trac_bc': [], 'grad_norm': []}

    @tf.function
    def train_step(self, xy_domain, xy_disp, disp_vals,
                   xy_trac, trac_vals, trac_normals):
        with tf.GradientTape() as tape:
            # PDE loss: equilibrium residual at collocation points
            pde_loss = self._pde_loss(xy_domain)

            # Displacement BC loss
            disp_loss = self._disp_bc_loss(xy_disp, disp_vals)

            # Traction BC loss
            trac_loss = self._trac_bc_loss(xy_trac, trac_vals, trac_normals)

            total = self.w_pde * pde_loss + self.w_disp * disp_loss + self.w_trac * trac_loss

        grads = tape.gradient(total, self.model.trainable_variables)

        safe_grads = []
        for grad, var in zip(grads, self.model.trainable_variables):
            if grad is None:
                safe_grads.append(tf.zeros_like(var))
            else:
                safe_grads.append(tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad)))

        if self.grad_clip_norm is not None:
            safe_grads, grad_norm = tf.clip_by_global_norm(safe_grads, self.grad_clip_norm)
        else:
            grad_norm = tf.linalg.global_norm(safe_grads)

        self.optimizer.apply_gradients(zip(safe_grads, self.model.trainable_variables))

        return total, pde_loss, disp_loss, trac_loss, grad_norm

    def _pde_loss(self, xy):
        """Compute equilibrium residual loss."""
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(xy)
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(xy)
                uv = self.model(xy, training=True)
                ux = uv[:, 0]
                uy = uv[:, 1]

            grad_ux = tape1.gradient(ux, xy)
            grad_uy = tape1.gradient(uy, xy)

            dux_dx = grad_ux[:, 0]
            dux_dy = grad_ux[:, 1]
            duy_dx = grad_uy[:, 0]
            duy_dy = grad_uy[:, 1]

            sxx = self.C11 * dux_dx + self.C12 * duy_dy
            syy = self.C12 * dux_dx + self.C11 * duy_dy
            sxy = self.C66 * (dux_dy + duy_dx)

        grad_sxx = tape2.gradient(sxx, xy)
        grad_syy = tape2.gradient(syy, xy)
        grad_sxy = tape2.gradient(sxy, xy)

        dsxx_dx = grad_sxx[:, 0] if grad_sxx is not None else tf.zeros_like(ux)
        dsxy_dy = grad_sxy[:, 1] if grad_sxy is not None else tf.zeros_like(ux)
        dsxy_dx = grad_sxy[:, 0] if grad_sxy is not None else tf.zeros_like(ux)
        dsyy_dy = grad_syy[:, 1] if grad_syy is not None else tf.zeros_like(ux)

        del tape1, tape2

        eq_x = dsxx_dx + dsxy_dy  # should be 0
        eq_y = dsxy_dx + dsyy_dy  # should be 0

        return tf.reduce_mean(eq_x ** 2 + eq_y ** 2)

    def _disp_bc_loss(self, xy, vals):
        """Displacement BC loss (component-wise, NaN = free)."""
        uv = self.model(xy, training=True)
        mask = tf.math.is_finite(vals)  # True for constrained, False for free
        diff = tf.where(mask, uv - vals, tf.zeros_like(uv))
        return tf.reduce_sum(diff ** 2) / (tf.cast(tf.reduce_sum(tf.cast(mask, tf.int32)), tf.float32) + 1e-10)

    def _trac_bc_loss(self, xy, trac_vals, normals):
        """Traction BC loss: sigma . n = t at boundary."""
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xy)
            uv = self.model(xy, training=True)
            ux = uv[:, 0]
            uy = uv[:, 1]

        grad_ux = tape.gradient(ux, xy)
        grad_uy = tape.gradient(uy, xy)
        del tape

        dux_dx = grad_ux[:, 0]
        dux_dy = grad_ux[:, 1]
        duy_dx = grad_uy[:, 0]
        duy_dy = grad_uy[:, 1]

        sxx = self.C11 * dux_dx + self.C12 * duy_dy
        syy = self.C12 * dux_dx + self.C11 * duy_dy
        sxy = self.C66 * (dux_dy + duy_dx)

        nx = normals[:, 0]
        ny = normals[:, 1]

        # Traction = sigma . n
        tx_pred = sxx * nx + sxy * ny
        ty_pred = sxy * nx + syy * ny

        tx_true = trac_vals[:, 0]
        ty_true = trac_vals[:, 1]

        return tf.reduce_mean((tx_pred - tx_true) ** 2 + (ty_pred - ty_true) ** 2)

    def train(self, data, epochs=10000, print_every=500):
        """
        Train the PINN.

        Parameters
        ----------
        data : tuple
            (xy_domain, xy_bc_disp, bc_disp_vals,
             xy_bc_trac, bc_trac_vals, bc_trac_normals)
        epochs : int
        print_every : int
        """
        (xy_domain, xy_bc_disp, bc_disp_vals,
         xy_bc_trac, bc_trac_vals, bc_trac_normals) = data

        xy_domain = tf.constant(xy_domain, dtype=tf.float32)
        xy_bc_disp = tf.constant(xy_bc_disp, dtype=tf.float32)
        bc_disp_vals = tf.constant(bc_disp_vals, dtype=tf.float32)
        xy_bc_trac = tf.constant(xy_bc_trac, dtype=tf.float32)
        bc_trac_vals = tf.constant(bc_trac_vals, dtype=tf.float32)
        bc_trac_normals = tf.constant(bc_trac_normals, dtype=tf.float32)

        for epoch in range(epochs):
            total, pde, disp, trac, grad_norm = self.train_step(
                xy_domain, xy_bc_disp, bc_disp_vals,
                xy_bc_trac, bc_trac_vals, bc_trac_normals
            )

            self.history['total'].append(float(total))
            self.history['pde'].append(float(pde))
            self.history['disp_bc'].append(float(disp))
            self.history['trac_bc'].append(float(trac))
            self.history['grad_norm'].append(float(grad_norm))

            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Total: {total:.6e}, "
                      f"PDE: {pde:.6e}, "
                      f"Disp BC: {disp:.6e}, "
                      f"Trac BC: {trac:.6e}, "
                      f"|g|: {grad_norm:.6e}")

        return self.history


def main():
    parser = argparse.ArgumentParser(
        description='Train PINN for 2D linear elasticity'
    )
    parser.add_argument('--problem', type=str, default='plate_with_hole',
                        choices=['plate_with_hole', 'l_bracket'])
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Global norm clipping for PINN gradients; <=0 disables clipping')
    parser.add_argument('--n-domain', type=int, default=10000)
    parser.add_argument('--n-boundary', type=int, default=2000)
    parser.add_argument('--layers', type=int, nargs='+', default=[128, 128, 128, 128])
    parser.add_argument('--activation', type=str, default='tanh')
    parser.add_argument('--output-dir', type=str, default='./models')

    # Material properties
    parser.add_argument('--E', type=float, default=1.0, help='Youngs modulus')
    parser.add_argument('--nu', type=float, default=0.3, help='Poissons ratio')
    parser.add_argument('--applied-stress', type=float, default=10.0)

    # Domain parameters (plate with hole)
    parser.add_argument('--x-min', type=float, default=-2.0)
    parser.add_argument('--x-max', type=float, default=2.0)
    parser.add_argument('--y-min', type=float, default=-2.0)
    parser.add_argument('--y-max', type=float, default=2.0)
    parser.add_argument('--hole-x', type=float, default=0.0)
    parser.add_argument('--hole-y', type=float, default=0.0)
    parser.add_argument('--hole-radius', type=float, default=0.5)

    # L-bracket parameters
    parser.add_argument('--corner-x', type=float, default=1.0)
    parser.add_argument('--corner-y', type=float, default=1.0)
    parser.add_argument('--fillet-radius', type=float, default=0.04,
                        help='Fillet radius at L-bracket re-entrant corner (0=sharp)')

    # Loss weights
    parser.add_argument('--w-pde', type=float, default=1.0)
    parser.add_argument('--w-disp', type=float, default=10.0)
    parser.add_argument('--w-trac', type=float, default=1.0)
    parser.add_argument('--no-hard-bc', action='store_true',
                        help='Disable hard displacement BC enforcement')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"PINN TRAINING: {args.problem}")
    print("=" * 60)

    # Adjust domain defaults for L-bracket
    if args.problem == 'l_bracket' and args.x_min == -2.0:
        args.x_min = 0.0
        args.y_min = 0.0

    # Build model
    input_range = [(args.x_min, args.x_max), (args.y_min, args.y_max)]
    hard_bc = args.problem if not args.no_hard_bc else None
    hard_bc_params = None
    if args.problem == 'l_bracket':
        hard_bc_params = {'corner_x': args.corner_x, 'corner_y': args.corner_y}
    network = Network()
    model = network.build(
        num_inputs=2,
        layers=args.layers,
        activation=args.activation,
        num_outputs=2,
        input_range=input_range,
        hard_bc=hard_bc,
        hard_bc_params=hard_bc_params,
    )
    model.summary()

    # Create training data
    print("\nGenerating training data...")
    data = create_training_data(args.problem, args.n_domain, args.n_boundary, args)
    print(f"  Domain points: {len(data[0])}")
    print(f"  Disp BC points: {len(data[1])}")
    print(f"  Trac BC points: {len(data[3])}")

    # Train — when hard BCs are active, displacement BC is exact; zero its weight
    w_disp = 0.0 if hard_bc else args.w_disp
    trainer = ElasticityPINNTrainer(
        model, E=args.E, nu=args.nu, lr=args.lr, epochs=args.epochs,
        w_pde=args.w_pde, w_disp=w_disp, w_trac=args.w_trac,
        grad_clip_norm=args.grad_clip if args.grad_clip > 0 else None,
    )

    print("\nTraining...")
    history = trainer.train(data, epochs=args.epochs)

    # Save model
    model_path = os.path.join(args.output_dir, f'pinn_{args.problem}.weights.h5')
    model.save_weights(model_path)
    print(f"\nSaved model to {model_path}")

    # Save history
    history_path = os.path.join(args.output_dir, f'pinn_{args.problem}_history.npz')
    np.savez(history_path, **history)
    print(f"Saved history to {history_path}")

    # Quick visualization
    try:
        import matplotlib.pyplot as plt
        try:
            import scienceplots
            plt.style.use(['science', 'no-latex'])
        except ImportError:
            pass

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.semilogy(history['total'], label='Total')
        ax.semilogy(history['pde'], label='PDE')
        ax.semilogy(history['disp_bc'], label='Disp BC')
        ax.semilogy(history['trac_bc'], label='Trac BC')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'PINN Training: {args.problem}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_path = os.path.join(args.output_dir, f'pinn_{args.problem}_loss.png')
        plt.savefig(fig_path, dpi=150)
        print(f"Saved loss plot to {fig_path}")
        plt.show()
    except Exception as e:
        print(f"Plotting skipped: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
