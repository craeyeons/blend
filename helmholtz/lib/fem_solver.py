"""
scikit-fem FEM solver for the 2D Helmholtz equation

    -Delta u - k^2 u = f   on [0,1]^2,
    u = g                 on a prescribed Dirichlet set.

Uses structured triangular mesh + linear Lagrange (P1) elements. The
Dirichlet set defaults to the 4 outer edges of the unit square but can
be extended by passing `dirichlet_predicate` at construction (used to
represent the Experiment-2 circular hole as an all-Dirichlet region).

`solve_with_extra_dirichlet` additionally pins an arbitrary set of DOFs
(from the hybrid router's "accept PINN" decision) — solved system is
condensed on (boundary ∪ extra), yielding a smaller interior system.
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

    def __init__(self, k, mesh_n=129, dirichlet_predicate=None):
        """
        Parameters
        ----------
        k : float
            Wavenumber.
        mesh_n : int
            Nodes per side of the structured tensor mesh.
        dirichlet_predicate : callable (x, y) -> bool array, optional
            If provided, marks any mesh vertex satisfying the predicate as
            Dirichlet (in addition to the outer boundary). Used to encode
            the circular hole for Experiment 2.
        """
        self.k = float(k)
        xs = np.linspace(0.0, 1.0, mesh_n)
        ys = np.linspace(0.0, 1.0, mesh_n)
        self.mesh = MeshTri.init_tensor(xs, ys)
        self.element = ElementTriP1()
        self.basis = Basis(self.mesh, self.element)

        # Outer-boundary DOFs (always Dirichlet).
        bd_dofs = self.basis.get_dofs()
        outer_ids = np.asarray(bd_dofs.nodal['u'], dtype=np.int64)

        # Predicate-satisfying interior DOFs (hole, etc.).
        if dirichlet_predicate is not None:
            pts = self.mesh.p  # (2, n_nodes)
            mask = dirichlet_predicate(pts[0], pts[1])
            extra_ids = np.where(mask)[0].astype(np.int64)
        else:
            extra_ids = np.array([], dtype=np.int64)

        self.outer_dof_ids = outer_ids
        self.predicate_dof_ids = extra_ids
        self.fixed_dof_ids = np.unique(
            np.concatenate([outer_ids, extra_ids])
        )

        # Cache stiffness and mass matrices: geometry-dependent only.
        self._K = _stiffness.assemble(self.basis)
        self._M = _mass.assemble(self.basis)
        self._A = self._K - (self.k ** 2) * self._M

    def _assemble_rhs(self, f_callable):
        @LinearForm
        def _rhs(v, w):
            x, y = w.x[0], w.x[1]
            return f_callable(x, y) * v
        return _rhs.assemble(self.basis)

    def solve(self, f_callable, g_callable):
        """Full FEM solve with the default Dirichlet set.

        Returns
        -------
        u_dof : ndarray, nodal DOF vector of shape (n_nodes,)
        solve_time_s : float, wall-clock of assemble-RHS + condense + spsolve.
        """
        t0 = time.perf_counter()
        b = self._assemble_rhs(f_callable)

        bd_xy = self.mesh.p[:, self.fixed_dof_ids]
        g_vals = g_callable(bd_xy[0], bd_xy[1])

        u = np.zeros(self.basis.N)
        u[self.fixed_dof_ids] = g_vals

        A_c, b_c, u_c, I = skfem.condense(
            self._A, b, x=u, D=self.fixed_dof_ids)
        u_c = spsolve(A_c, b_c)
        u[I] = u_c

        solve_time_s = time.perf_counter() - t0
        return u.astype(np.float32), solve_time_s

    def solve_with_extra_dirichlet(self, f_callable, g_callable,
                                   extra_dof_ids, extra_values):
        """Hybrid path: solve with an *extended* Dirichlet set.

        The system matrix A and mass/stiffness caches are reused; only the
        RHS is assembled per call. The condense set is
        (outer ∪ predicate ∪ extra_dof_ids), so the interior system solved
        by spsolve is smaller than in :meth:`solve` by |extra_dof_ids|.

        Parameters
        ----------
        f_callable, g_callable :
            Same as :meth:`solve`; g_callable is evaluated only at the
            outer + predicate DOFs (typically homogeneous here).
        extra_dof_ids : (M,) int array
            Mesh vertex ids at which to pin the provided values.
        extra_values : (M,) float array
            Values to pin (from the PINN at those vertex coordinates).

        Returns
        -------
        u_dof, solve_time_s
        """
        t0 = time.perf_counter()
        b = self._assemble_rhs(f_callable)

        u = np.zeros(self.basis.N)
        # Original Dirichlet values
        base_xy = self.mesh.p[:, self.fixed_dof_ids]
        u[self.fixed_dof_ids] = g_callable(base_xy[0], base_xy[1])

        # Extra Dirichlet values (overwrite any overlap).
        extra_dof_ids = np.asarray(extra_dof_ids, dtype=np.int64)
        extra_values = np.asarray(extra_values, dtype=np.float64)
        u[extra_dof_ids] = extra_values

        D = np.unique(np.concatenate([self.fixed_dof_ids, extra_dof_ids]))
        A_c, b_c, u_c, I = skfem.condense(self._A, b, x=u, D=D)
        u_c = spsolve(A_c, b_c)
        u[I] = u_c

        solve_time_s = time.perf_counter() - t0
        return u.astype(np.float32), solve_time_s

    def interp_to_grid(self, u_dof, X, Y):
        """Interpolate nodal FEM solution onto a regular (Ny, Nx) grid."""
        from scipy.interpolate import griddata
        pts = self.mesh.p.T  # (n_nodes, 2)
        target = np.stack([X.ravel(), Y.ravel()], axis=-1)
        vals = griddata(pts, u_dof, target, method='linear')
        if np.any(np.isnan(vals)):
            vals_near = griddata(pts, u_dof, target, method='nearest')
            mask = np.isnan(vals)
            vals[mask] = vals_near[mask]
        return vals.reshape(X.shape).astype(np.float32)

    def vertex_ids_for_grid(self, X, Y):
        """Return (Ny, Nx) array of nearest mesh vertex ids for each grid cell.

        Useful for the hybrid router -> DOF mapping: the router decision on
        cell (i,j) is applied to `vertex_ids_for_grid()[i,j]`.
        """
        from scipy.spatial import cKDTree
        tree = cKDTree(self.mesh.p.T)
        target = np.stack([X.ravel(), Y.ravel()], axis=-1)
        _, idx = tree.query(target, k=1)
        return idx.reshape(X.shape).astype(np.int64)
