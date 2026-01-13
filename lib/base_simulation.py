"""
Base simulation class for PINN-CFD hybrid solvers.
"""

import jax
import jax.numpy as jnp
from jax import jit, lax
import numpy as np
from abc import ABC, abstractmethod


class BaseSimulation(ABC):
    """
    Abstract base class for fluid dynamics simulations using hybrid PINN-CFD.
    
    Attributes:
        Re: Reynolds number
        N: Number of grid points per side
        max_iter: Maximum iterations
        tol: Convergence tolerance
        nu: Kinematic viscosity
        dx, dy, dt: Grid spacing and time step
    """
    
    def __init__(self, Re=100, N=100, max_iter=200000, tol=1e-6):
        """
        Initialize simulation parameters.
        
        Parameters:
        -----------
        Re : float
            Reynolds number
        N : int
            Number of grid points per side
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        """
        self.Re = Re
        self.N = N
        self.max_iter = max_iter
        self.tol = tol
        self.nu = 1.0 / Re
        
        # Grid setup
        self.L = 1.0
        self.dx = self.L / (N - 1)
        self.dy = self.dx
        self.dt = 0.001 * min(self.dx, self.dy)
        
        # Create coordinate grid
        x = np.linspace(0, self.L, N)
        y = np.linspace(0, self.L, N)
        self.X, self.Y = np.meshgrid(x, y)
        self.xy = np.stack([self.X.flatten(), self.Y.flatten()], axis=-1)
        
        print(f"JAX devices: {jax.devices()}")
        print(f"Using device: {jax.devices()[0]}")
    
    @abstractmethod
    def apply_boundary_conditions(self, u, v):
        """Apply velocity boundary conditions specific to the simulation."""
        pass
    
    @abstractmethod
    def get_simulation_name(self):
        """Return the name of the simulation."""
        pass
    
    @abstractmethod
    def get_output_filename(self):
        """Return the base filename for output."""
        pass
    
    def _create_jit_functions(self):
        """Create JIT-compiled helper functions."""
        dx, dy = self.dx, self.dy
        
        @jit
        def laplacian(f):
            """Compute Laplacian using central differences"""
            lap = jnp.zeros_like(f)
            lap = lap.at[1:-1, 1:-1].set(
                (f[1:-1, 2:] - 2*f[1:-1, 1:-1] + f[1:-1, :-2]) / dx**2 +
                (f[2:, 1:-1] - 2*f[1:-1, 1:-1] + f[:-2, 1:-1]) / dy**2
            )
            return lap
        
        @jit
        def divergence(u, v):
            """Compute divergence of velocity field"""
            div = jnp.zeros_like(u)
            div = div.at[1:-1, 1:-1].set(
                (u[1:-1, 2:] - u[1:-1, :-2]) / (2*dx) +
                (v[2:, 1:-1] - v[:-2, 1:-1]) / (2*dy)
            )
            return div
        
        @jit
        def convection(u, v, f):
            """Compute convective term (u·∇)f"""
            dfdx = jnp.zeros_like(f)
            dfdy = jnp.zeros_like(f)
            
            dfdx = dfdx.at[1:-1, 1:-1].set(
                (f[1:-1, 2:] - f[1:-1, :-2]) / (2*dx)
            )
            dfdy = dfdy.at[1:-1, 1:-1].set(
                (f[2:, 1:-1] - f[:-2, 1:-1]) / (2*dy)
            )
            
            return u * dfdx + v * dfdy
        
        @jit
        def pressure_gradient(p):
            """Compute pressure gradient"""
            dpdx = jnp.zeros_like(p)
            dpdy = jnp.zeros_like(p)
            dpdx = dpdx.at[1:-1, 1:-1].set((p[1:-1, 2:] - p[1:-1, :-2]) / (2*dx))
            dpdy = dpdy.at[1:-1, 1:-1].set((p[2:, 1:-1] - p[:-2, 1:-1]) / (2*dy))
            return dpdx, dpdy
        
        return laplacian, divergence, convection, pressure_gradient
    
    def solve(self):
        """
        Solve Navier-Stokes equations using fractional step method.
        
        Returns:
        --------
        u, v, p : ndarray
            Velocity and pressure fields
        """
        N = self.N
        dx, dy, dt = self.dx, self.dy, self.dt
        nu = self.nu
        
        # Initialize fields
        u = jnp.zeros((N, N))
        v = jnp.zeros((N, N))
        p = jnp.zeros((N, N))
        
        # Apply initial boundary conditions
        u, v = self.apply_boundary_conditions(u, v)
        
        # Get JIT-compiled functions
        laplacian, divergence, convection, pressure_gradient = self._create_jit_functions()
        apply_bc = jit(self.apply_boundary_conditions)
        
        @jit
        def pressure_poisson_iteration(p, rhs):
            """Single Jacobi iteration for pressure Poisson equation"""
            p_new = jnp.zeros_like(p)
            p_new = p_new.at[1:-1, 1:-1].set(
                0.25 * (
                    p[1:-1, 2:] + p[1:-1, :-2] +
                    p[2:, 1:-1] + p[:-2, 1:-1] -
                    dx**2 * rhs[1:-1, 1:-1]
                )
            )
            # Neumann BC: dp/dn = 0 on walls
            p_new = p_new.at[0, :].set(p_new[1, :])
            p_new = p_new.at[-1, :].set(p_new[-2, :])
            p_new = p_new.at[:, 0].set(p_new[:, 1])
            p_new = p_new.at[:, -1].set(p_new[:, -2])
            return p_new
        
        @jit
        def solve_pressure_poisson(p, rhs, n_iter=100):
            """Solve pressure Poisson equation with Jacobi iterations"""
            def body_fn(i, p):
                return pressure_poisson_iteration(p, rhs)
            return lax.fori_loop(0, n_iter, body_fn, p)
        
        @jit
        def step(u, v, p):
            """Single time step using fractional step method"""
            # Step 1: Compute intermediate velocity (without pressure)
            conv_u = convection(u, v, u)
            conv_v = convection(u, v, v)
            lap_u = laplacian(u)
            lap_v = laplacian(v)
            
            u_star = u + dt * (-conv_u + nu * lap_u)
            v_star = v + dt * (-conv_v + nu * lap_v)
            
            # Apply BCs to intermediate velocity
            u_star, v_star = apply_bc(u_star, v_star)
            
            # Step 2: Solve pressure Poisson equation
            div_star = divergence(u_star, v_star)
            rhs = div_star / dt
            p_new = solve_pressure_poisson(p, rhs)
            
            # Step 3: Correct velocity
            dpdx, dpdy = pressure_gradient(p_new)
            u_new = u_star - dt * dpdx
            v_new = v_star - dt * dpdy
            
            # Apply BCs to corrected velocity
            u_new, v_new = apply_bc(u_new, v_new)
            
            return u_new, v_new, p_new
        
        @jit
        def compute_residual(u_new, u_old, v_new, v_old):
            """Compute convergence residual"""
            diff_u = jnp.sum((u_new - u_old)**2)
            diff_v = jnp.sum((v_new - v_old)**2)
            return jnp.sqrt(diff_u + diff_v)
        
        print(f"Solving {self.get_simulation_name()} with JAX (Fractional Step method)...")
        print(f"Reynolds number: {self.Re}")
        print(f"Grid size: {N}x{N}")
        print(f"Time step: {dt:.6e}")
        print("-" * 50)
        
        # Time stepping loop
        for n in range(self.max_iter):
            u_old, v_old = u, v
            u, v, p = step(u, v, p)
            
            # Check convergence every 50 iterations
            if n % 50 == 0:
                residual = compute_residual(u, u_old, v, v_old)
                residual_val = float(residual)
                print(f"Iteration {n}, Residual: {residual_val:.6e}")
                
                if residual_val < self.tol:
                    print(f"\nConverged at iteration {n}")
                    break
        else:
            print(f"\nReached maximum iterations ({self.max_iter})")
        
        return np.array(u), np.array(v), np.array(p)
