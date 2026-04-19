"""
JAX-based finite difference solver for 2D static linear elasticity.

Solves Navier's displacement equations on a regular grid using
Jacobi iteration (direct analogue of the pressure Poisson solver
in the fluid code).

Plane stress formulation:
    C11 d2ux/dx2 + C66 d2ux/dy2 + (C12+C66) d2uy/dxdy = 0
    (C12+C66) d2ux/dxdy + C66 d2uy/dx2 + C11 d2uy/dy2 = 0

where C11 = E/(1-nu^2), C12 = nu*E/(1-nu^2), C66 = E/(2(1+nu)).
"""

import jax
import jax.numpy as jnp
from jax import jit, lax
import numpy as np

jax.config.update("jax_enable_x64", True)


class ElasticitySolver:
    """
    Jacobi-iteration solver for 2D plane-stress linear elasticity.

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    x_domain, y_domain : tuple
        Physical domain bounds (min, max).
    Nx, Ny : int
        Grid dimensions.
    max_iter : int
        Maximum Jacobi iterations.
    tol : float
        Convergence tolerance on displacement increment.
    """

    def __init__(self, E=1.0, nu=0.3,
                 x_domain=(-2.0, 2.0), y_domain=(-2.0, 2.0),
                 Nx=200, Ny=200, max_iter=200000, tol=1e-8,
                 relaxation=0.2):
        self.E = E
        self.nu = nu
        self.Nx = Nx
        self.Ny = Ny
        self.max_iter = max_iter
        self.tol = tol
        self.relaxation = relaxation

        x_min, x_max = x_domain
        y_min, y_max = y_domain
        self.dx = (x_max - x_min) / (Nx - 1)
        self.dy = (y_max - y_min) / (Ny - 1)

        x = np.linspace(x_min, x_max, Nx)
        y = np.linspace(y_min, y_max, Ny)
        self.X, self.Y = np.meshgrid(x, y)

        # Plane-stress stiffness coefficients
        self.C11 = E / (1.0 - nu ** 2)
        self.C12 = nu * E / (1.0 - nu ** 2)
        self.C66 = E / (2.0 * (1.0 + nu))

        print(f"JAX devices: {jax.devices()}")
        print(f"ElasticitySolver: {Nx}x{Ny}, E={E}, nu={nu}, omega={relaxation}")

    def solve(self, layout, disp_bc_mask, bc_ux, bc_uy,
              trac_bc_mask=None, bc_tx=None, bc_ty=None,
              fdm_mask=None, initial_ux=None, initial_uy=None):
        """
        Solve for displacement field using Jacobi iteration.

        Parameters
        ----------
        layout : ndarray (Ny, Nx)
            1 = material, 0 = void.
        disp_bc_mask : ndarray (Ny, Nx)
            1 where displacement BCs are prescribed.
        bc_ux, bc_uy : ndarray (Ny, Nx)
            Prescribed displacement values.
        trac_bc_mask : ndarray (Ny, Nx), optional
            1 where traction BCs are prescribed.
        bc_tx, bc_ty : ndarray (Ny, Nx), optional
            Prescribed traction values.

        Returns
        -------
        ux, uy : ndarray (Ny, Nx)
            Displacement fields.
        sxx, syy, sxy : ndarray (Ny, Nx)
            Stress fields.
        """
        dx, dy = self.dx, self.dy
        C11, C12, C66 = self.C11, self.C12, self.C66
        Ny, Nx = self.Ny, self.Nx

        layout_j = jnp.array(layout, dtype=jnp.float64)
        disp_mask_j = jnp.array(disp_bc_mask, dtype=jnp.float64)
        bc_ux_j = jnp.array(bc_ux, dtype=jnp.float64)
        bc_uy_j = jnp.array(bc_uy, dtype=jnp.float64)

        if trac_bc_mask is not None:
            trac_mask_j = jnp.array(trac_bc_mask, dtype=jnp.float64)
            bc_tx_j = jnp.array(bc_tx, dtype=jnp.float64)
            bc_ty_j = jnp.array(bc_ty, dtype=jnp.float64)
        else:
            trac_mask_j = jnp.zeros_like(layout_j)
            bc_tx_j = jnp.zeros_like(layout_j)
            bc_ty_j = jnp.zeros_like(layout_j)

        # Hybrid mode: pin PINN values in non-FDM regions
        if fdm_mask is not None:
            fdm_mask_j = jnp.array(fdm_mask, dtype=jnp.float64) * layout_j
            # Always solve physically constrained boundary nodes with FDM,
            # regardless of router selection.
            bc_forced_mask = (disp_mask_j > 0) | (trac_mask_j > 0)
            fdm_mask_j = jnp.where(bc_forced_mask, 1.0, fdm_mask_j)
            init_ux_j = jnp.array(initial_ux, dtype=jnp.float64) * layout_j
            init_uy_j = jnp.array(initial_uy, dtype=jnp.float64) * layout_j
        else:
            fdm_mask_j = layout_j
            init_ux_j = jnp.zeros_like(layout_j)
            init_uy_j = jnp.zeros_like(layout_j)

        # Precompute coefficients for Jacobi update
        # From the discretized Navier equations:
        #   C11/dx^2 (ux[i,j+1] + ux[i,j-1]) + C66/dy^2 (ux[i+1,j] + ux[i-1,j])
        #   + (C12+C66)/(4*dx*dy) (uy[i+1,j+1] - uy[i+1,j-1] - uy[i-1,j+1] + uy[i-1,j-1])
        #   = a_P * ux[i,j]
        # where a_P = 2*C11/dx^2 + 2*C66/dy^2

        a_x = C11 / dx ** 2  # coefficient for ux neighbours in x
        a_y = C66 / dy ** 2  # coefficient for ux neighbours in y

        b_x = C66 / dx ** 2  # coefficient for uy neighbours in x
        b_y = C11 / dy ** 2  # coefficient for uy neighbours in y

        c_cross = (C12 + C66) / (4.0 * dx * dy)

        # Layout-aware neighbor weights.  When a neighbor is void we mirror
        # the current cell (zero-gradient / implicit Neumann), which means
        # that neighbor drops out of both the numerator and the diagonal a_P.
        w_xp = jnp.zeros_like(layout_j).at[1:-1, 1:-1].set(layout_j[1:-1, 2:])
        w_xm = jnp.zeros_like(layout_j).at[1:-1, 1:-1].set(layout_j[1:-1, :-2])
        w_yp = jnp.zeros_like(layout_j).at[1:-1, 1:-1].set(layout_j[2:, 1:-1])
        w_ym = jnp.zeros_like(layout_j).at[1:-1, 1:-1].set(layout_j[:-2, 1:-1])

        # Per-point diagonal for ux equation
        a_P_ux = a_x * (w_xp + w_xm) + a_y * (w_yp + w_ym)
        a_P_ux = jnp.where(a_P_ux < 1e-30, 1.0, a_P_ux)

        # Per-point diagonal for uy equation
        a_P_uy = b_x * (w_xp + w_xm) + b_y * (w_yp + w_ym)
        a_P_uy = jnp.where(a_P_uy < 1e-30, 1.0, a_P_uy)

        # Diagonal-neighbor layout weights for cross-derivative stencil
        w_pp = jnp.zeros_like(layout_j).at[1:-1, 1:-1].set(layout_j[2:, 2:])
        w_pm = jnp.zeros_like(layout_j).at[1:-1, 1:-1].set(layout_j[2:, :-2])
        w_mp = jnp.zeros_like(layout_j).at[1:-1, 1:-1].set(layout_j[:-2, 2:])
        w_mm = jnp.zeros_like(layout_j).at[1:-1, 1:-1].set(layout_j[:-2, :-2])

        @jit
        def jacobi_step(ux, uy):
            """Single Jacobi iteration for coupled ux, uy."""
            # --- Update ux ---
            # Neighbour contributions, masked by layout so void gives 0
            ux_xp = jnp.zeros_like(ux).at[1:-1, 1:-1].set(ux[1:-1, 2:])  * w_xp
            ux_xm = jnp.zeros_like(ux).at[1:-1, 1:-1].set(ux[1:-1, :-2]) * w_xm
            ux_yp = jnp.zeros_like(ux).at[1:-1, 1:-1].set(ux[2:, 1:-1])  * w_yp
            ux_ym = jnp.zeros_like(ux).at[1:-1, 1:-1].set(ux[:-2, 1:-1]) * w_ym

            # Cross-derivative of uy (layout-masked diagonal neighbors)
            uy_cross = jnp.zeros_like(uy)
            uy_cross = uy_cross.at[1:-1, 1:-1].set(
                uy[2:, 2:]   * w_pp[1:-1, 1:-1]
              - uy[2:, :-2]  * w_pm[1:-1, 1:-1]
              - uy[:-2, 2:]  * w_mp[1:-1, 1:-1]
              + uy[:-2, :-2] * w_mm[1:-1, 1:-1]
            )

            ux_new = jnp.zeros_like(ux)
            ux_new = ux_new.at[1:-1, 1:-1].set(
                (a_x * (ux_xp[1:-1, 1:-1] + ux_xm[1:-1, 1:-1])
                 + a_y * (ux_yp[1:-1, 1:-1] + ux_ym[1:-1, 1:-1])
                 + c_cross * uy_cross[1:-1, 1:-1]) / a_P_ux[1:-1, 1:-1]
            )

            # --- Update uy ---
            uy_xp = jnp.zeros_like(uy).at[1:-1, 1:-1].set(uy[1:-1, 2:])  * w_xp
            uy_xm = jnp.zeros_like(uy).at[1:-1, 1:-1].set(uy[1:-1, :-2]) * w_xm
            uy_yp = jnp.zeros_like(uy).at[1:-1, 1:-1].set(uy[2:, 1:-1])  * w_yp
            uy_ym = jnp.zeros_like(uy).at[1:-1, 1:-1].set(uy[:-2, 1:-1]) * w_ym

            # Cross-derivative of ux (layout-masked diagonal neighbors)
            ux_cross = jnp.zeros_like(ux)
            ux_cross = ux_cross.at[1:-1, 1:-1].set(
                ux[2:, 2:]   * w_pp[1:-1, 1:-1]
              - ux[2:, :-2]  * w_pm[1:-1, 1:-1]
              - ux[:-2, 2:]  * w_mp[1:-1, 1:-1]
              + ux[:-2, :-2] * w_mm[1:-1, 1:-1]
            )

            uy_new = jnp.zeros_like(uy)
            uy_new = uy_new.at[1:-1, 1:-1].set(
                (b_x * (uy_xp[1:-1, 1:-1] + uy_xm[1:-1, 1:-1])
                 + b_y * (uy_yp[1:-1, 1:-1] + uy_ym[1:-1, 1:-1])
                 + c_cross * ux_cross[1:-1, 1:-1]) / a_P_uy[1:-1, 1:-1]
            )

            # Apply displacement BCs (NaN = free component, skip)
            ux_new = jnp.where((disp_mask_j > 0) & jnp.isfinite(bc_ux_j), bc_ux_j, ux_new)
            uy_new = jnp.where((disp_mask_j > 0) & jnp.isfinite(bc_uy_j), bc_uy_j, uy_new)

            # Apply traction BCs (Neumann): copy from interior neighbour
            # For simplicity, use zero-gradient (free surface) where traction=0
            # and offset where traction != 0
            # Right edge traction: sigma_xx = C11*dux/dx + C12*duy/dy = tx
            # Approximate: ux[boundary] = ux[interior] + tx*dx/C11
            # This is a first-order Neumann BC implementation
            ux_new = jnp.where(
                (trac_mask_j > 0) & (disp_mask_j == 0),
                self._apply_traction_ux(ux_new, uy_new, bc_tx_j, bc_ty_j,
                                        trac_mask_j, layout_j),
                ux_new
            )
            uy_new = jnp.where(
                (trac_mask_j > 0) & (disp_mask_j == 0),
                self._apply_traction_uy(ux_new, uy_new, bc_tx_j, bc_ty_j,
                                        trac_mask_j, layout_j),
                uy_new
            )

            # Hybrid: pin PINN values in non-FDM regions
            keep_computed = fdm_mask_j > 0
            ux_new = jnp.where(keep_computed, ux_new, init_ux_j)
            uy_new = jnp.where(keep_computed, uy_new, init_uy_j)

            # Zero out void regions
            ux_new = ux_new * layout_j
            uy_new = uy_new * layout_j

            return ux_new, uy_new

        # Traction BC helpers (Neumann via ghost-cell approach)
        #
        # On a vertical boundary (normal = +x):
        #   sigma_xx = C11 dux/dx + C12 duy/dy = tx
        #   sigma_xy = C66 (dux/dy + duy/dx) = ty
        # Approximate dux/dx ~ (ux_ghost - ux_interior) / dx
        #   => ux_ghost = ux_interior + (tx - C12 duy/dy) * dx / C11
        # For simplicity, neglect the cross-term C12*duy/dy (it couples
        # and converges via iteration). This gives:
        #   ux_ghost ~ ux_interior + tx * dx / C11
        #   uy_ghost ~ uy_interior + ty * dy / C11   (y-normal face)
        #
        # On a horizontal boundary (normal = +y):
        #   sigma_yy = C12 dux/dx + C11 duy/dy = ty
        #   sigma_xy = C66 (dux/dy + duy/dx) = tx
        #   => uy_ghost ~ uy_interior + ty * dy / C11
        #   => ux_ghost ~ ux_interior + tx * dy / C66

        def _pad_zeros(arr):
            """Pad with zeros (void) so domain edges see void outside."""
            return jnp.pad(arr, 1, mode='constant', constant_values=0.0)

        def _interior_avg(field, tmask, layout):
            """Average of interior (non-traction, non-void) neighbours."""
            fp = _pad_zeros(field)
            lp = _pad_zeros(layout)
            tp = _pad_zeros(tmask)

            f_l = fp[1:-1, :-2]
            f_r = fp[1:-1, 2:]
            f_d = fp[:-2, 1:-1]
            f_u = fp[2:, 1:-1]

            w_l = lp[1:-1, :-2] * (1 - tp[1:-1, :-2])
            w_r = lp[1:-1, 2:] * (1 - tp[1:-1, 2:])
            w_d = lp[:-2, 1:-1] * (1 - tp[:-2, 1:-1])
            w_u = lp[2:, 1:-1] * (1 - tp[2:, 1:-1])

            count = w_l + w_r + w_d + w_u
            total = f_l * w_l + f_r * w_r + f_d * w_d + f_u * w_u
            return total / jnp.maximum(count, 1.0)

        def _boundary_is_horizontal(tmask, layout):
            """1 where the traction face has a y-facing normal."""
            lp = _pad_zeros(layout)
            has_void_above = (1 - lp[2:, 1:-1])
            has_void_below = (1 - lp[:-2, 1:-1])
            return jnp.clip(has_void_above + has_void_below, 0.0, 1.0)

        def _apply_trac_ux(ux, uy, tx, ty, tmask, layout):
            """Neumann BC for ux at traction boundaries."""
            avg = _interior_avg(ux, tmask, layout)
            is_horiz = _boundary_is_horizontal(tmask, layout)
            # Vertical face: ux_ghost = avg + tx*dx/C11
            # Horizontal face: ux_ghost = avg + tx*dy/C66  (from sigma_xy = tx)
            offset = (1 - is_horiz) * tx * dx / C11 + is_horiz * tx * dy / C66
            return avg + offset

        def _apply_trac_uy(ux, uy, tx, ty, tmask, layout):
            """Neumann BC for uy at traction boundaries."""
            avg = _interior_avg(uy, tmask, layout)
            is_horiz = _boundary_is_horizontal(tmask, layout)
            # Horizontal face: uy_ghost = avg + ty*dy/C11
            # Vertical face: uy_ghost = avg + ty*dx/C66  (from sigma_xy = ty)
            offset = is_horiz * ty * dy / C11 + (1 - is_horiz) * ty * dx / C66
            return avg + offset

        self._apply_traction_ux = jit(_apply_trac_ux)
        self._apply_traction_uy = jit(_apply_trac_uy)

        # Initialize displacement fields
        if fdm_mask is not None:
            ux = init_ux_j
            uy = init_uy_j
        else:
            ux = jnp.zeros((Ny, Nx), dtype=jnp.float64)
            uy = jnp.zeros((Ny, Nx), dtype=jnp.float64)

        # Apply initial BCs (NaN = free component, skip)
        ux = jnp.where((disp_mask_j > 0) & jnp.isfinite(bc_ux_j), bc_ux_j, ux)
        uy = jnp.where((disp_mask_j > 0) & jnp.isfinite(bc_uy_j), bc_uy_j, uy)

        # Apply traction BCs to the initial state as well, so hybrid
        # initialization is boundary-consistent before the first iteration.
        ux = jnp.where(
            (trac_mask_j > 0) & (disp_mask_j == 0),
            self._apply_traction_ux(ux, uy, bc_tx_j, bc_ty_j, trac_mask_j, layout_j),
            ux
        )
        uy = jnp.where(
            (trac_mask_j > 0) & (disp_mask_j == 0),
            self._apply_traction_uy(ux, uy, bc_tx_j, bc_ty_j, trac_mask_j, layout_j),
            uy
        )

        @jit
        def compute_residual(ux_new, ux_old, uy_new, uy_old):
            diff_ux = jnp.sum((ux_new - ux_old) ** 2)
            diff_uy = jnp.sum((uy_new - uy_old) ** 2)
            return jnp.sqrt(diff_ux + diff_uy)

        print(f"Solving 2D linear elasticity (Jacobi iteration)...")
        print(f"Grid: {Nx}x{Ny}, max_iter: {self.max_iter}, tol: {self.tol}")
        print("-" * 50)

        prev_residual = np.inf
        for n in range(self.max_iter):
            ux_old, uy_old = ux, uy
            ux_new, uy_new = jacobi_step(ux, uy)

            # Under-relaxation for stability in coupled elasticity Jacobi.
            # omega=1 is plain Jacobi; smaller omega damps divergent modes.
            omega = jnp.asarray(self.relaxation, dtype=ux.dtype)
            ux = (1.0 - omega) * ux_old + omega * ux_new
            uy = (1.0 - omega) * uy_old + omega * uy_new

            if n % 100 == 0:
                residual = float(compute_residual(ux, ux_old, uy, uy_old))
                print(f"Iteration {n}, Residual: {residual:.6e}")

                if not np.isfinite(residual) or (np.isfinite(prev_residual) and residual > prev_residual * 100.0):
                    raise RuntimeError(
                        "Elasticity solver diverged. "
                        f"Residual jumped from {prev_residual:.6e} to {residual:.6e}. "
                        "Try smaller --omega (e.g. 0.1) or lower applied stress."
                    )

                if residual < self.tol:
                    print(f"\nConverged at iteration {n}")
                    break
                prev_residual = residual
        else:
            print(f"\nReached maximum iterations ({self.max_iter})")

        # Compute stresses from displacements
        ux_np = np.array(ux)
        uy_np = np.array(uy)
        sxx, syy, sxy = self.compute_stress(ux_np, uy_np)

        return ux_np, uy_np, sxx, syy, sxy

    def compute_stress(self, ux, uy):
        """
        Compute stress field from displacement using central differences.

        Returns
        -------
        sxx, syy, sxy : ndarray (Ny, Nx)
        """
        dx, dy = self.dx, self.dy
        C11, C12, C66 = self.C11, self.C12, self.C66

        # Strain via central differences
        exx = np.zeros_like(ux)
        eyy = np.zeros_like(uy)
        exy = np.zeros_like(ux)

        exx[1:-1, 1:-1] = (ux[1:-1, 2:] - ux[1:-1, :-2]) / (2 * dx)
        eyy[1:-1, 1:-1] = (uy[2:, 1:-1] - uy[:-2, 1:-1]) / (2 * dy)
        exy[1:-1, 1:-1] = 0.5 * (
            (ux[2:, 1:-1] - ux[:-2, 1:-1]) / (2 * dy)
            + (uy[1:-1, 2:] - uy[1:-1, :-2]) / (2 * dx)
        )

        sxx = C11 * exx + C12 * eyy
        syy = C12 * exx + C11 * eyy
        sxy = 2 * C66 * exy

        return sxx, syy, sxy

    def compute_von_mises(self, sxx, syy, sxy):
        """Compute von Mises stress."""
        return np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3 * sxy ** 2)
