"""
scikit-fem FEM solver for the 2D Helmholtz equation

    -Delta u - k^2 u = f   on [0,1]^2,
    u = g                 on the boundary.

Uses structured triangular mesh + linear Lagrange (P1) elements. No gmsh
dependency needed for the square domain — `MeshTri.init_tensor` gives a
regular triangulation of the square directly.

For the square-with-hole (Experiment 2) we will swap in a gmsh mesh; this
file can be extended then.
"""

import time
import numpy as np
import skfem
from skfem import MeshTri, ElementTriP1, Basis, BilinearForm, LinearForm
from skfem.helpers import dot, grad
from scipy.sparse.linalg import spsolve


@BilinearForm
def _stiffness(u, v, w):
    return dot(grad(u), grad(v))


@BilinearForm
def _mass(u, v, w):
    return u * v


class HelmholtzSolver:
    """Solve -Delta u - k^2 u = f with Dirichlet BC on [0, 1]^2."""

    def __init__(self, k, mesh_n=129):
        """
        Parameters
        ----------
        k : float
            Wavenumber.
        mesh_n : int
            Nodes per side of the structured tensor mesh.
            Total nodes = mesh_n^2, elements ~= 2 * (mesh_n - 1)^2.
        """
        self.k = float(k)
        xs = np.linspace(0.0, 1.0, mesh_n)
        ys = np.linspace(0.0, 1.0, mesh_n)
        self.mesh = MeshTri.init_tensor(xs, ys)
        self.element = ElementTriP1()
        self.basis = Basis(self.mesh, self.element)

    def solve(self, f_callable, g_callable):
        """
        Parameters
        ----------
        f_callable : callable (x, y) -> values  (arrays of same shape)
            Forcing in `-Delta u - k^2 u = f`.
        g_callable : callable (x, y) -> values
            Dirichlet data on the boundary.

        Returns
        -------
        u_dof : ndarray, nodal DOF vector of shape (n_nodes,)
        solve_time_s : float, wall-clock time of assemble + solve
        """
        t0 = time.perf_counter()

        K = _stiffness.assemble(self.basis)
        M = _mass.assemble(self.basis)
        # PDE is -Delta u - k^2 u = f; weak form: (grad u, grad v) - k^2 (u, v) = (f, v)
        A = K - (self.k ** 2) * M

        @LinearForm
        def _rhs(v, w):
            x, y = w.x[0], w.x[1]
            return f_callable(x, y) * v
        b = _rhs.assemble(self.basis)

        # Dirichlet: set boundary DOFs to g(x, y).
        bd_dofs = self.basis.get_dofs()
        bd_node_ids = bd_dofs.nodal['u']
        bd_xy = self.mesh.p[:, bd_node_ids]
        g_vals = g_callable(bd_xy[0], bd_xy[1])

        u = np.zeros(self.basis.N)
        u[bd_node_ids] = g_vals

        A_c, b_c, u_c, I = skfem.condense(A, b, x=u, D=bd_node_ids)
        u_c = spsolve(A_c, b_c)
        u[I] = u_c

        solve_time_s = time.perf_counter() - t0
        return u.astype(np.float32), solve_time_s

    def interp_to_grid(self, u_dof, X, Y):
        """Interpolate nodal FEM solution onto a regular (Ny, Nx) grid.

        Linear FEM ⇒ nodal values are the DOF values at mesh vertices.
        Using scipy.interpolate.griddata with linear interpolation over the
        triangulation, which exactly matches the P1 FEM field.
        """
        from scipy.interpolate import griddata
        pts = self.mesh.p.T  # (n_nodes, 2)
        target = np.stack([X.ravel(), Y.ravel()], axis=-1)
        vals = griddata(pts, u_dof, target, method='linear')
        # Fill any NaNs from hull-edge points via nearest fallback.
        if np.any(np.isnan(vals)):
            vals_near = griddata(pts, u_dof, target, method='nearest')
            mask = np.isnan(vals)
            vals[mask] = vals_near[mask]
        return vals.reshape(X.shape).astype(np.float32)
