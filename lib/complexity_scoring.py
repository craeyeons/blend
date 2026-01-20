"""
Dynamic segregation based on local complexity scoring for hybrid PINN-CFD solvers.

This module implements the computation of local diagnostic quantities and
constructs a complexity score to automatically segregate the domain into
CFD-dominated and PINN-dominated regions.

References:
-----------
Local diagnostic quantities characterize shear, rotation, and the degree to which
governing equations are difficult to satisfy, enabling physics-informed segregation.
"""

import numpy as np
import jax.numpy as jnp
from jax import jit
import tensorflow as tf


class ComplexityScorer:
    """
    Computes local complexity scores based on flow field diagnostics.
    
    The complexity score is a weighted combination of:
    - Strain-rate magnitude (characterizes local shear)
    - Vorticity magnitude (characterizes local rotation)
    - Momentum equation residual (indicates PDE violation)
    - Continuity residual (indicates mass conservation violation)
    
    Attributes:
        weights: Weights for each diagnostic component
        ref_scales: Reference scales for normalization
    """
    
    def __init__(self, weights=None, normalization='mean'):
        """
        Initialize complexity scorer.
        
        Parameters:
        -----------
        weights : dict, optional
            Weights for [strain_rate, vorticity, momentum_residual, continuity_residual].
            Default: {'strain': 1.0, 'vorticity': 1.0, 'momentum': 2.0, 'continuity': 2.0}
            Residual-based terms receive higher priority.
        normalization : str
            Normalization method: 'mean', 'max', or 'percentile' (95th)
        """
        self.weights = weights or {
            'strain': 1.0,
            'vorticity': 1.0,
            'momentum': 2.0,
            'continuity': 2.0
        }
        self.normalization = normalization
        self.ref_scales = {}
        
    def compute_strain_rate_magnitude(self, u, v, dx, dy):
        """
        Compute strain-rate magnitude.
        
        ∥S∥ = √(2 S_ij S_ij)
        
        where S_ij = 1/2 * (∂_j u_i + ∂_i u_j)
        
        Parameters:
        -----------
        u, v : ndarray of shape (Ny, Nx)
            Velocity components
        dx, dy : float
            Grid spacing
            
        Returns:
        --------
        strain_magnitude : ndarray of shape (Ny, Nx)
            Local strain-rate magnitude
        """
        # Compute velocity gradients using central differences (interior points)
        # Pad boundaries with zeros
        du_dx = np.zeros_like(u)
        du_dy = np.zeros_like(u)
        dv_dx = np.zeros_like(v)
        dv_dy = np.zeros_like(v)
        
        # Interior points: central differences
        du_dx[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)
        du_dy[1:-1, 1:-1] = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy)
        dv_dx[1:-1, 1:-1] = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)
        dv_dy[1:-1, 1:-1] = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)
        
        # Boundary points: forward/backward differences
        # Left boundary (x=0)
        du_dx[:, 0] = (u[:, 1] - u[:, 0]) / dx
        dv_dx[:, 0] = (v[:, 1] - v[:, 0]) / dx
        
        # Right boundary (x=Nx-1)
        du_dx[:, -1] = (u[:, -1] - u[:, -2]) / dx
        dv_dx[:, -1] = (v[:, -1] - v[:, -2]) / dx
        
        # Bottom boundary (y=0)
        du_dy[0, :] = (u[1, :] - u[0, :]) / dy
        dv_dy[0, :] = (v[1, :] - v[0, :]) / dy
        
        # Top boundary (y=Ny-1)
        du_dy[-1, :] = (u[-1, :] - u[-2, :]) / dy
        dv_dy[-1, :] = (v[-1, :] - v[-2, :]) / dy
        
        # Strain-rate tensor components
        # S_xx = ∂u/∂x
        # S_yy = ∂v/∂y
        # S_xy = S_yx = 1/2 * (∂u/∂y + ∂v/∂x)
        S_xx = du_dx
        S_yy = dv_dy
        S_xy = 0.5 * (du_dy + dv_dx)
        
        # ∥S∥ = √(2 * S_ij * S_ij) = √(2 * (S_xx^2 + 2*S_xy^2 + S_yy^2))
        strain_magnitude = np.sqrt(2.0 * (S_xx**2 + 2*S_xy**2 + S_yy**2))
        
        return strain_magnitude
    
    def compute_vorticity_magnitude(self, u, v, dx, dy):
        """
        Compute vorticity magnitude.
        
        |ω| = |∂v/∂x - ∂u/∂y|
        
        Parameters:
        -----------
        u, v : ndarray of shape (Ny, Nx)
            Velocity components
        dx, dy : float
            Grid spacing
            
        Returns:
        --------
        vorticity_magnitude : ndarray of shape (Ny, Nx)
            Local vorticity magnitude
        """
        # Compute velocity gradients using central differences
        dv_dx = np.zeros_like(v)
        du_dy = np.zeros_like(u)
        
        # Interior points: central differences
        dv_dx[1:-1, 1:-1] = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)
        du_dy[1:-1, 1:-1] = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy)
        
        # Boundary points: forward/backward differences
        dv_dx[:, 0] = (v[:, 1] - v[:, 0]) / dx
        dv_dx[:, -1] = (v[:, -1] - v[:, -2]) / dx
        du_dy[0, :] = (u[1, :] - u[0, :]) / dy
        du_dy[-1, :] = (u[-1, :] - u[-2, :]) / dy
        
        vorticity = dv_dx - du_dy
        vorticity_magnitude = np.abs(vorticity)
        
        return vorticity_magnitude
    
    def compute_momentum_residual(self, u, v, p, nu, dx, dy, rho=1.0):
        """
        Compute momentum equation residual.
        
        R_m = √(r_u^2 + r_v^2)
        
        where:
        r_u = ρ(u ∂u/∂x + v ∂u/∂y) + ∂p/∂x - μ(∂²u/∂x² + ∂²u/∂y²)
        r_v = ρ(u ∂v/∂x + v ∂v/∂y) + ∂p/∂y - μ(∂²v/∂x² + ∂²v/∂y²)
        
        Parameters:
        -----------
        u, v : ndarray of shape (Ny, Nx)
            Velocity components
        p : ndarray of shape (Ny, Nx)
            Pressure field
        nu : float
            Kinematic viscosity
        dx, dy : float
            Grid spacing
        rho : float
            Fluid density (default: 1.0)
            
        Returns:
        --------
        momentum_residual : ndarray of shape (Ny, Nx)
            Local momentum equation residual
        """
        # Compute first-order derivatives (interior points only)
        Ny, Nx = u.shape
        
        du_dx = np.zeros_like(u)
        du_dy = np.zeros_like(u)
        dv_dx = np.zeros_like(v)
        dv_dy = np.zeros_like(v)
        dp_dx = np.zeros_like(p)
        dp_dy = np.zeros_like(p)
        
        # Central differences for interior
        du_dx[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)
        du_dy[1:-1, 1:-1] = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy)
        dv_dx[1:-1, 1:-1] = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)
        dv_dy[1:-1, 1:-1] = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)
        dp_dx[1:-1, 1:-1] = (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dx)
        dp_dy[1:-1, 1:-1] = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dy)
        
        # Compute second-order derivatives (interior points only)
        d2u_dx2 = np.zeros_like(u)
        d2u_dy2 = np.zeros_like(u)
        d2v_dx2 = np.zeros_like(v)
        d2v_dy2 = np.zeros_like(v)
        
        d2u_dx2[1:-1, 1:-1] = (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dx**2
        d2u_dy2[1:-1, 1:-1] = (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dy**2
        d2v_dx2[1:-1, 1:-1] = (v[1:-1, 2:] - 2*v[1:-1, 1:-1] + v[1:-1, :-2]) / dx**2
        d2v_dy2[1:-1, 1:-1] = (v[2:, 1:-1] - 2*v[1:-1, 1:-1] + v[:-2, 1:-1]) / dy**2
        
        # Momentum residuals (only interior points are meaningful)
        r_u = (rho * (u * du_dx + v * du_dy) + dp_dx - 
               nu * (d2u_dx2 + d2u_dy2))
        r_v = (rho * (u * dv_dx + v * dv_dy) + dp_dy - 
               nu * (d2v_dx2 + d2v_dy2))
        
        momentum_residual = np.sqrt(r_u**2 + r_v**2)
        
        return momentum_residual
    
    def compute_continuity_residual(self, u, v, dx, dy):
        """
        Compute continuity equation residual (divergence).
        
        R_c = |∂u/∂x + ∂v/∂y|
        
        Parameters:
        -----------
        u, v : ndarray of shape (Ny, Nx)
            Velocity components
        dx, dy : float
            Grid spacing
            
        Returns:
        --------
        continuity_residual : ndarray of shape (Ny, Nx)
            Local continuity residual
        """
        du_dx = np.zeros_like(u)
        dv_dy = np.zeros_like(v)
        
        # Central differences for interior
        du_dx[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)
        dv_dy[1:-1, 1:-1] = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)
        
        divergence = du_dx + dv_dy
        continuity_residual = np.abs(divergence)
        
        return continuity_residual
    
    def compute_reference_scales(self, strain, vorticity, momentum, continuity):
        """
        Compute reference scales for normalization.
        
        Parameters:
        -----------
        strain, vorticity, momentum, continuity : ndarray
            Diagnostic quantities (interior points only)
            
        Returns:
        --------
        dict : Reference scales for each quantity
        """
        ref_scales = {}
        
        # Use only interior points (exclude boundaries where derivatives are less accurate)
        interior = slice(1, -1), slice(1, -1)
        
        if self.normalization == 'mean':
            ref_scales['strain'] = np.mean(strain[interior]) + 1e-14
            ref_scales['vorticity'] = np.mean(vorticity[interior]) + 1e-14
            ref_scales['momentum'] = np.mean(momentum[interior]) + 1e-14
            ref_scales['continuity'] = np.mean(continuity[interior]) + 1e-14
        elif self.normalization == 'max':
            ref_scales['strain'] = np.max(strain[interior]) + 1e-14
            ref_scales['vorticity'] = np.max(vorticity[interior]) + 1e-14
            ref_scales['momentum'] = np.max(momentum[interior]) + 1e-14
            ref_scales['continuity'] = np.max(continuity[interior]) + 1e-14
        elif self.normalization == 'percentile':
            ref_scales['strain'] = np.percentile(strain[interior], 95) + 1e-14
            ref_scales['vorticity'] = np.percentile(vorticity[interior], 95) + 1e-14
            ref_scales['momentum'] = np.percentile(momentum[interior], 95) + 1e-14
            ref_scales['continuity'] = np.percentile(continuity[interior], 95) + 1e-14
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization}")
        
        self.ref_scales = ref_scales
        return ref_scales
    
    def compute_complexity_score(self, u, v, p, nu, dx, dy, rho=1.0):
        """
        Compute local complexity score.
        
        D(x,y) = w1 * ∥S∥/S_ref + w2 * |ω|/ω_ref + 
                  w3 * R_m/R_m,ref + w4 * R_c/R_c,ref
        
        Parameters:
        -----------
        u, v : ndarray of shape (Ny, Nx)
            Velocity components
        p : ndarray of shape (Ny, Nx)
            Pressure field
        nu : float
            Kinematic viscosity
        dx, dy : float
            Grid spacing
        rho : float
            Fluid density (default: 1.0)
            
        Returns:
        --------
        complexity_score : ndarray of shape (Ny, Nx)
            Local complexity score at each grid point
        """
        # Compute diagnostic quantities
        strain = self.compute_strain_rate_magnitude(u, v, dx, dy)
        vorticity = self.compute_vorticity_magnitude(u, v, dx, dy)
        momentum = self.compute_momentum_residual(u, v, p, nu, dx, dy, rho)
        continuity = self.compute_continuity_residual(u, v, dx, dy)
        
        # Compute reference scales
        self.compute_reference_scales(strain, vorticity, momentum, continuity)
        
        # Normalize and combine
        complexity_score = (
            self.weights['strain'] * strain / self.ref_scales['strain'] +
            self.weights['vorticity'] * vorticity / self.ref_scales['vorticity'] +
            self.weights['momentum'] * momentum / self.ref_scales['momentum'] +
            self.weights['continuity'] * continuity / self.ref_scales['continuity']
        )
        
        return complexity_score
    
    def get_diagnostics(self, u, v, p, nu, dx, dy, rho=1.0):
        """
        Get all diagnostic quantities for analysis.
        
        Returns:
        --------
        dict : Diagnostic quantities and reference scales
        """
        strain = self.compute_strain_rate_magnitude(u, v, dx, dy)
        vorticity = self.compute_vorticity_magnitude(u, v, dx, dy)
        momentum = self.compute_momentum_residual(u, v, p, nu, dx, dy, rho)
        continuity = self.compute_continuity_residual(u, v, dx, dy)
        
        self.compute_reference_scales(strain, vorticity, momentum, continuity)
        
        return {
            'strain': strain,
            'vorticity': vorticity,
            'momentum': momentum,
            'continuity': continuity,
            'ref_scales': self.ref_scales
        }


def create_dynamic_mask(u, v, p, nu, dx, dy, threshold=1.0, 
                       weights=None, normalization='mean', rho=1.0,
                       invert=False, boundary_width=1, obstacle_mask=None,
                       merge_distance=0):
    """
    Create binary mask based on complexity score threshold.
    
    Parameters:
    -----------
    u, v : ndarray of shape (Ny, Nx)
        Velocity components
    p : ndarray of shape (Ny, Nx)
        Pressure field
    nu : float
        Kinematic viscosity
    dx, dy : float
        Grid spacing
    threshold : float
        Complexity score threshold for CFD assignment
        - D(x,y) > threshold: assign to CFD (mask=1)
        - D(x,y) ≤ threshold: assign to PINN (mask=0)
    weights : dict, optional
        Weights for diagnostic components
    normalization : str
        Normalization method
    rho : float
        Fluid density
    invert : bool
        If True, invert the mask (swap CFD and PINN regions)
    boundary_width : int
        Width of forced CFD region at domain boundaries (default: 1).
        Set to 0 to disable boundary forcing.
    obstacle_mask : ndarray of shape (Ny, Nx), optional
        Binary mask where 1 indicates obstacle (e.g., cylinder).
        These regions and their immediate neighbors will be forced to CFD.
    merge_distance : int
        Number of grid cells for merging nearby CFD regions (default: 0).
        Uses morphological closing: dilate then erode by this amount.
        This connects CFD regions that are within 2*merge_distance cells of each other.
        
    Returns:
    --------
    mask : ndarray of shape (Ny, Nx) with dtype int32
        Binary mask (1=CFD, 0=PINN)
    complexity_score : ndarray of shape (Ny, Nx)
        Local complexity scores
    """
    from scipy import ndimage
    
    scorer = ComplexityScorer(weights=weights, normalization=normalization)
    
    # Compute complexity score
    complexity_score = scorer.compute_complexity_score(u, v, p, nu, dx, dy, rho)
    
    # Create mask based on threshold
    if invert:
        mask = (complexity_score <= threshold).astype(np.int32)
    else:
        mask = (complexity_score > threshold).astype(np.int32)
    
    Ny, Nx = mask.shape
    
    # Force domain boundaries to CFD
    if boundary_width > 0:
        mask[:boundary_width, :] = 1   # Bottom
        mask[-boundary_width:, :] = 1  # Top
        mask[:, :50] = 1   # Left
        mask[:, -boundary_width:] = 1  # Right
    
    # Force obstacle region and surrounding cells to CFD
    if obstacle_mask is not None:
        # Dilate obstacle to include boundary layer region
        dilated_obstacle = ndimage.binary_dilation(
            obstacle_mask > 0, iterations=2
        ).astype(np.int32)
        mask = np.maximum(mask, dilated_obstacle)
    
    # Merge nearby CFD regions using morphological closing
    if merge_distance > 0:
        # Binary closing = dilation followed by erosion
        # This connects regions within 2*merge_distance of each other
        mask = ndimage.binary_opening(
            mask > 0, iterations=merge_distance
        ).astype(np.int32)
        
        # Re-enforce boundaries and obstacle after closing
        if boundary_width > 0:
            mask[:boundary_width, :] = 1   # Bottom
            mask[-boundary_width:, :] = 1  # Top
            mask[:, :50] = 1   # Left
            mask[:, -boundary_width:] = 1  # Right
        
        if obstacle_mask is not None:
            mask = np.maximum(mask, dilated_obstacle)
    
    return mask, complexity_score


def compute_mask_statistics(mask, complexity_score):
    """
    Compute statistics about the segregation.
    
    Parameters:
    -----------
    mask : ndarray of shape (Ny, Nx)
        Binary mask
    complexity_score : ndarray of shape (Ny, Nx)
        Complexity scores
        
    Returns:
    --------
    dict : Statistics including region sizes and complexity statistics
    """
    Ny, Nx = mask.shape
    total_points = Ny * Nx
    
    cfd_region = mask == 1
    pinn_region = mask == 0
    
    cfd_count = np.sum(cfd_region)
    pinn_count = np.sum(pinn_region)
    
    stats = {
        'total_points': total_points,
        'cfd_count': cfd_count,
        'pinn_count': pinn_count,
        'cfd_percentage': 100.0 * cfd_count / total_points,
        'pinn_percentage': 100.0 * pinn_count / total_points,
        'cfd_avg_complexity': np.mean(complexity_score[cfd_region]) if cfd_count > 0 else 0,
        'pinn_avg_complexity': np.mean(complexity_score[pinn_region]) if pinn_count > 0 else 0,
        'complexity_min': np.min(complexity_score),
        'complexity_max': np.max(complexity_score),
        'complexity_mean': np.mean(complexity_score),
        'complexity_std': np.std(complexity_score),
    }
    
    return stats
