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
    if problem == 'plate_with_hole':
        x_domain = (args.x_min, args.x_max)
        y_domain = (args.y_min, args.y_max)
        cx, cy = args.hole_x, args.hole_y
        R = args.hole_radius

        # Sample interior points (reject those inside hole)
        xy_all = []
        while len(xy_all) < n_domain:
            x = np.random.uniform(x_domain[0], x_domain[1], n_domain * 2)
            y = np.random.uniform(y_domain[0], y_domain[1], n_domain * 2)
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            valid = dist > R
            pts = np.stack([x[valid], y[valid]], axis=-1)
            xy_all.append(pts)
        xy_domain = np.concatenate(xy_all, axis=0)[:n_domain]

        # Displacement BCs
        # Use NaN for the free component in roller BCs so the loss
        # only penalises the constrained component.
        disp_pts = []
        disp_vals = []

        # Left edge: ux = 0, uy free (roller)
        y_left = np.random.uniform(y_domain[0], y_domain[1], n_boundary // 4)
        x_left = np.full_like(y_left, x_domain[0])
        disp_pts.append(np.stack([x_left, y_left], axis=-1))
        disp_vals.append(np.stack([np.zeros_like(y_left),
                                   np.full_like(y_left, np.nan)], axis=-1))

        # Bottom edge: uy = 0, ux free (roller)
        x_bot = np.random.uniform(x_domain[0], x_domain[1], n_boundary // 4)
        y_bot = np.full_like(x_bot, y_domain[0])
        # Reject points inside hole
        dist_bot = np.sqrt((x_bot - cx) ** 2 + (y_bot - cy) ** 2)
        valid_bot = dist_bot > R
        x_bot, y_bot = x_bot[valid_bot], y_bot[valid_bot]
        disp_pts.append(np.stack([x_bot, y_bot], axis=-1))
        disp_vals.append(np.stack([np.full_like(y_bot, np.nan),
                                   np.zeros_like(y_bot)], axis=-1))

        xy_bc_disp = np.concatenate(disp_pts, axis=0).astype(np.float32)
        bc_disp_vals = np.concatenate(disp_vals, axis=0).astype(np.float32)

        # Traction BCs
        trac_pts = []
        trac_vals = []
        trac_normals = []

        # Right edge: tx = applied_stress, ty = 0
        y_right = np.random.uniform(y_domain[0], y_domain[1], n_boundary // 4)
        x_right = np.full_like(y_right, x_domain[1])
        trac_pts.append(np.stack([x_right, y_right], axis=-1))
        trac_vals.append(np.stack([np.full_like(y_right, args.applied_stress),
                                   np.zeros_like(y_right)], axis=-1))
        trac_normals.append(np.stack([np.ones_like(y_right), np.zeros_like(y_right)], axis=-1))

        # Top edge: traction-free
        x_top = np.random.uniform(x_domain[0], x_domain[1], n_boundary // 4)
        y_top = np.full_like(x_top, y_domain[1])
        dist_top = np.sqrt((x_top - cx) ** 2 + (y_top - cy) ** 2)
        valid_top = dist_top > R
        x_top, y_top = x_top[valid_top], y_top[valid_top]
        trac_pts.append(np.stack([x_top, y_top], axis=-1))
        trac_vals.append(np.zeros((len(x_top), 2), dtype=np.float32))
        trac_normals.append(np.stack([np.zeros_like(x_top), np.ones_like(x_top)], axis=-1))

        # Hole surface: traction-free
        theta = np.random.uniform(0, 2 * np.pi, n_boundary // 2)
        x_hole = cx + R * np.cos(theta)
        y_hole = cy + R * np.sin(theta)
        # Keep only points inside domain
        in_domain = ((x_hole >= x_domain[0]) & (x_hole <= x_domain[1])
                     & (y_hole >= y_domain[0]) & (y_hole <= y_domain[1]))
        x_hole, y_hole, theta_h = x_hole[in_domain], y_hole[in_domain], theta[in_domain]
        trac_pts.append(np.stack([x_hole, y_hole], axis=-1))
        trac_vals.append(np.zeros((len(x_hole), 2), dtype=np.float32))
        # Outward normal points away from center
        trac_normals.append(np.stack([np.cos(theta_h), np.sin(theta_h)], axis=-1))

        xy_bc_trac = np.concatenate(trac_pts, axis=0).astype(np.float32)
        bc_trac_vals = np.concatenate(trac_vals, axis=0).astype(np.float32)
        bc_trac_normals = np.concatenate(trac_normals, axis=0).astype(np.float32)

    elif problem == 'l_bracket':
        x_domain = (args.x_min, args.x_max)
        y_domain = (args.y_min, args.y_max)
        corner_x, corner_y = args.corner_x, args.corner_y

        # Sample interior points (reject upper-right block)
        xy_all = []
        while len(xy_all) < n_domain:
            x = np.random.uniform(x_domain[0], x_domain[1], n_domain * 2)
            y = np.random.uniform(y_domain[0], y_domain[1], n_domain * 2)
            valid = ~((x > corner_x) & (y > corner_y))
            pts = np.stack([x[valid], y[valid]], axis=-1)
            xy_all.append(pts)
        xy_domain = np.concatenate(xy_all, axis=0)[:n_domain]

        # Displacement BCs: bottom edge fixed
        disp_pts = []
        disp_vals = []

        x_bot = np.random.uniform(x_domain[0], x_domain[1], n_boundary // 2)
        y_bot = np.full_like(x_bot, y_domain[0])
        disp_pts.append(np.stack([x_bot, y_bot], axis=-1))
        disp_vals.append(np.zeros((len(x_bot), 2), dtype=np.float32))

        xy_bc_disp = np.concatenate(disp_pts, axis=0).astype(np.float32)
        bc_disp_vals = np.concatenate(disp_vals, axis=0).astype(np.float32)

        # Traction BCs
        trac_pts = []
        trac_vals = []
        trac_normals = []

        # Right edge of lower arm (x=x_max, y < corner_y): applied traction
        y_right = np.random.uniform(y_domain[0], corner_y, n_boundary // 4)
        x_right = np.full_like(y_right, x_domain[1])
        trac_pts.append(np.stack([x_right, y_right], axis=-1))
        trac_vals.append(np.stack([np.full_like(y_right, args.applied_stress),
                                   np.zeros_like(y_right)], axis=-1))
        trac_normals.append(np.stack([np.ones_like(y_right), np.zeros_like(y_right)], axis=-1))

        # Left edge: traction-free
        y_left = np.random.uniform(y_domain[0], y_domain[1], n_boundary // 6)
        x_left = np.full_like(y_left, x_domain[0])
        trac_pts.append(np.stack([x_left, y_left], axis=-1))
        trac_vals.append(np.zeros((len(y_left), 2), dtype=np.float32))
        trac_normals.append(np.stack([-np.ones_like(y_left), np.zeros_like(y_left)], axis=-1))

        # Top edge (y=y_max, x < corner_x): traction-free
        x_top = np.random.uniform(x_domain[0], corner_x, n_boundary // 6)
        y_top = np.full_like(x_top, y_domain[1])
        trac_pts.append(np.stack([x_top, y_top], axis=-1))
        trac_vals.append(np.zeros((len(x_top), 2), dtype=np.float32))
        trac_normals.append(np.stack([np.zeros_like(x_top), np.ones_like(x_top)], axis=-1))

        # Re-entrant corner edges: horizontal inner edge (y=corner_y, x > corner_x)
        x_inner_h = np.random.uniform(corner_x, x_domain[1], n_boundary // 8)
        y_inner_h = np.full_like(x_inner_h, corner_y)
        trac_pts.append(np.stack([x_inner_h, y_inner_h], axis=-1))
        trac_vals.append(np.zeros((len(x_inner_h), 2), dtype=np.float32))
        trac_normals.append(np.stack([np.zeros_like(x_inner_h), np.ones_like(x_inner_h)], axis=-1))

        # Vertical inner edge (x=corner_x, y > corner_y)
        y_inner_v = np.random.uniform(corner_y, y_domain[1], n_boundary // 8)
        x_inner_v = np.full_like(y_inner_v, corner_x)
        trac_pts.append(np.stack([x_inner_v, y_inner_v], axis=-1))
        trac_vals.append(np.zeros((len(y_inner_v), 2), dtype=np.float32))
        trac_normals.append(np.stack([np.ones_like(y_inner_v), np.zeros_like(y_inner_v)], axis=-1))

        xy_bc_trac = np.concatenate(trac_pts, axis=0).astype(np.float32)
        bc_trac_vals = np.concatenate(trac_vals, axis=0).astype(np.float32)
        bc_trac_normals = np.concatenate(trac_normals, axis=0).astype(np.float32)

    else:
        raise ValueError(f"Unknown problem: {problem}")

    return (xy_domain.astype(np.float32), xy_bc_disp, bc_disp_vals,
            xy_bc_trac, bc_trac_vals, bc_trac_normals)


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

    def __init__(self, model, E=1.0, nu=0.3, lr=1e-3,
                 w_pde=1.0, w_disp=10.0, w_trac=1.0):
        self.model = model
        self.E = E
        self.nu = nu
        self.C11 = E / (1.0 - nu ** 2)
        self.C12 = nu * E / (1.0 - nu ** 2)
        self.C66 = E / (2.0 * (1.0 + nu))
        self.w_pde = w_pde
        self.w_disp = w_disp
        self.w_trac = w_trac

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.history = {'total': [], 'pde': [], 'disp_bc': [], 'trac_bc': []}

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
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return total, pde_loss, disp_loss, trac_loss

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
            total, pde, disp, trac = self.train_step(
                xy_domain, xy_bc_disp, bc_disp_vals,
                xy_bc_trac, bc_trac_vals, bc_trac_normals
            )

            self.history['total'].append(float(total))
            self.history['pde'].append(float(pde))
            self.history['disp_bc'].append(float(disp))
            self.history['trac_bc'].append(float(trac))

            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Total: {total:.6e}, "
                      f"PDE: {pde:.6e}, "
                      f"Disp BC: {disp:.6e}, "
                      f"Trac BC: {trac:.6e}")

        return self.history


def main():
    parser = argparse.ArgumentParser(
        description='Train PINN for 2D linear elasticity'
    )
    parser.add_argument('--problem', type=str, default='plate_with_hole',
                        choices=['plate_with_hole', 'l_bracket'])
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n-domain', type=int, default=10000)
    parser.add_argument('--n-boundary', type=int, default=2000)
    parser.add_argument('--layers', type=int, nargs='+', default=[64, 64, 64, 64])
    parser.add_argument('--activation', type=str, default='tanh')
    parser.add_argument('--output-dir', type=str, default='./models')

    # Material properties
    parser.add_argument('--E', type=float, default=1.0, help='Youngs modulus')
    parser.add_argument('--nu', type=float, default=0.3, help='Poissons ratio')
    parser.add_argument('--applied-stress', type=float, default=1.0)

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

    # Loss weights
    parser.add_argument('--w-pde', type=float, default=1.0)
    parser.add_argument('--w-disp', type=float, default=10.0)
    parser.add_argument('--w-trac', type=float, default=1.0)

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
    network = Network()
    model = network.build(
        num_inputs=2,
        layers=args.layers,
        activation=args.activation,
        num_outputs=2,
    )
    model.summary()

    # Create training data
    print("\nGenerating training data...")
    data = create_training_data(args.problem, args.n_domain, args.n_boundary, args)
    print(f"  Domain points: {len(data[0])}")
    print(f"  Disp BC points: {len(data[1])}")
    print(f"  Trac BC points: {len(data[3])}")

    # Train
    trainer = ElasticityPINNTrainer(
        model, E=args.E, nu=args.nu, lr=args.lr,
        w_pde=args.w_pde, w_disp=args.w_disp, w_trac=args.w_trac,
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
