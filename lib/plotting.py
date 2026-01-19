"""
Plotting utilities for fluid dynamics simulations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec


def generate_filename(simulation_type, mode, Re, suffix='', output_dir='./'):
    """
    Generate a standardized filename for saving figures.
    
    Parameters:
    -----------
    simulation_type : str
        Type of simulation ('cavity', 'cylinder')
    mode : str
        Solver mode ('cfd', 'hybrid', 'pinn')
    Re : float
        Reynolds number
    suffix : str
        Additional suffix for the filename
    output_dir : str
        Output directory for saving files
    
    Returns:
    --------
    str : Full path to save the file
    """
    os.makedirs(output_dir, exist_ok=True)
    Re_str = f"Re{int(Re)}" if Re == int(Re) else f"Re{Re:.1f}"
    suffix_str = f"_{suffix}" if suffix else ""
    filename = f"{simulation_type}_{mode}_{Re_str}{suffix_str}.png"
    return os.path.join(output_dir, filename)


def plot_contour(x, y, z, title, levels=50, ax=None, add_colorbar=True,
                 cmap='rainbow', show_circle=None):
    """
    Create a contour plot.
    
    Parameters:
    -----------
    x, y : ndarray
        Coordinate arrays
    z : ndarray
        Field values to plot
    title : str
        Plot title
    levels : int
        Number of contour levels
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes.
    add_colorbar : bool
        Whether to add a colorbar
    cmap : str
        Colormap name
    show_circle : tuple, optional
        (cx, cy, r) to draw a filled circle (for cylinder visualization)
    
    Returns:
    --------
    cf : contour object
    """
    if ax is None:
        ax = plt.gca()
    
    vmin = np.min(z)
    vmax = np.max(z)
    
    cf = ax.contourf(x, y, z, levels=levels, cmap=cmap,
                     norm=Normalize(vmin=vmin, vmax=vmax))
    
    if show_circle is not None:
        cx, cy, r = show_circle
        circle = plt.Circle((cx, cy), r, fc='black')
        ax.add_patch(circle)
    
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    if add_colorbar:
        plt.colorbar(cf, ax=ax)
    
    return cf


def plot_solution(u, v, p, x=None, y=None, title_prefix='', save_path=None,
                  show_circle=None, simulation_type=None, mode=None, Re=None):
    """
    Plot velocity and pressure fields in a 2x2 grid.
    
    Parameters:
    -----------
    u, v : ndarray
        Velocity components
    p : ndarray
        Pressure field
    x, y : ndarray, optional
        Coordinate arrays. If None, will create unit grid.
    title_prefix : str
        Prefix for titles
    save_path : str, optional
        Path to save figure. If None and simulation_type/mode/Re provided, auto-generates.
    show_circle : tuple, optional
        (cx, cy, r) for cylinder visualization
    simulation_type : str, optional
        Type of simulation for auto-generating filename
    mode : str, optional
        Solver mode for auto-generating filename
    Re : float, optional
        Reynolds number for auto-generating filename
    """
    Ny, Nx = u.shape
    if x is None or y is None:
        x = np.linspace(0, 1, Nx)
        y = np.linspace(0, 1, Ny)
        x, y = np.meshgrid(x, y)
    
    vel_mag = np.sqrt(u**2 + v**2)
    
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 2)
    
    ax1 = fig.add_subplot(gs[0, 0])
    plot_contour(x, y, vel_mag, f'{title_prefix}|u| (velocity magnitude)', 
                 ax=ax1, cmap='rainbow', show_circle=show_circle)
    
    ax2 = fig.add_subplot(gs[0, 1])
    plot_contour(x, y, p, f'{title_prefix}p (pressure)', 
                 ax=ax2, cmap='rainbow', show_circle=show_circle)
    
    ax3 = fig.add_subplot(gs[1, 0])
    plot_contour(x, y, u, f'{title_prefix}u (x-velocity)', 
                 ax=ax3, cmap='rainbow', show_circle=show_circle)
    
    ax4 = fig.add_subplot(gs[1, 1])
    plot_contour(x, y, v, f'{title_prefix}v (y-velocity)', 
                 ax=ax4, cmap='rainbow', show_circle=show_circle)
    
    plt.tight_layout()
    
    # Auto-generate save path if not provided
    if save_path is None and simulation_type and mode and Re is not None:
        save_path = generate_filename(simulation_type, mode, Re, 'solution')
    
    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Saved figure to {save_path}")
    
    plt.close(fig)


def plot_single_field(x, y, z, title, save_path=None, show_circle=None,
                      figsize=(16, 8), simulation_type=None, mode=None, Re=None, suffix='field'):
    """
    Plot a single field.
    
    Parameters:
    -----------
    x, y : ndarray
        Coordinate arrays
    z : ndarray
        Field values
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    show_circle : tuple, optional
        (cx, cy, r) for cylinder visualization
    figsize : tuple
        Figure size
    simulation_type, mode, Re : optional
        For auto-generating filename
    suffix : str
        Suffix for auto-generated filename
    """
    fig = plt.figure(figsize=figsize)
    plot_contour(x, y, z, title, show_circle=show_circle)
    plt.tight_layout()
    
    if save_path is None and simulation_type and mode and Re is not None:
        save_path = generate_filename(simulation_type, mode, Re, suffix)
    
    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Saved figure to {save_path}")
    
    plt.close(fig)


def plot_hybrid_solution(u, v, p, mask, x=None, y=None, save_path=None,
                         show_circle=None, simulation_type=None, mode=None, Re=None):
    """
    Plot velocity and pressure fields with mask overlay showing PINN/CFD regions.
    
    Parameters:
    -----------
    u, v : ndarray
        Velocity components
    p : ndarray
        Pressure field
    mask : ndarray
        Binary mask (1 = CFD, 0 = PINN)
    x, y : ndarray, optional
        Coordinate arrays
    save_path : str, optional
        Path to save figure
    show_circle : tuple, optional
        (cx, cy, r) for cylinder visualization
    simulation_type, mode, Re : optional
        For auto-generating filename
    """
    Ny, Nx = u.shape
    if x is None or y is None:
        x = np.linspace(0, 1, Nx)
        y = np.linspace(0, 1, Ny)
        x, y = np.meshgrid(x, y)
    
    vel_mag = np.sqrt(u**2 + v**2)
    
    def contour_with_mask(ax, X, Y, z, title, mask):
        cf = ax.contourf(X, Y, z, levels=50, cmap='rainbow')
        ax.contour(X, Y, mask, levels=[0.5], colors='white', 
                   linewidths=2, linestyles='--')
        if show_circle:
            cx, cy, r = show_circle
            circle = plt.Circle((cx, cy), r, fc='black')
            ax.add_patch(circle)
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax)
    
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(2, 2)
    
    ax1 = fig.add_subplot(gs[0, 0])
    contour_with_mask(ax1, x, y, vel_mag, '|u| (velocity magnitude)', mask)
    
    ax2 = fig.add_subplot(gs[0, 1])
    contour_with_mask(ax2, x, y, p, 'p (pressure)', mask)
    
    ax3 = fig.add_subplot(gs[1, 0])
    contour_with_mask(ax3, x, y, u, 'u (x-velocity)', mask)
    
    ax4 = fig.add_subplot(gs[1, 1])
    contour_with_mask(ax4, x, y, v, 'v (y-velocity)', mask)
    
    fig.text(0.5, 0.02, 'White dashed line: PINN/CFD interface', 
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    
    if save_path is None and simulation_type and mode and Re is not None:
        save_path = generate_filename(simulation_type, mode, Re, 'hybrid_mask')
    
    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Saved figure to {save_path}")
    
    plt.close(fig)


def plot_comparison(solutions, labels, x=None, y=None, save_path=None,
                    show_circle=None, simulation_type=None, Re=None):
    """
    Plot comparison of multiple solutions.
    
    Parameters:
    -----------
    solutions : list of tuples
        List of (u, v, p) solution tuples
    labels : list of str
        Labels for each solution
    x, y : ndarray, optional
        Coordinate arrays
    save_path : str, optional
        Path to save figure
    show_circle : tuple, optional
        (cx, cy, r) for cylinder visualization
    simulation_type, Re : optional
        For auto-generating filename
    """
    n_solutions = len(solutions)
    
    if x is None or y is None:
        Ny, Nx = solutions[0][0].shape
        x = np.linspace(0, 1, Nx)
        y = np.linspace(0, 1, Ny)
        x, y = np.meshgrid(x, y)
    
    fig, axes = plt.subplots(n_solutions, 3, figsize=(15, 5*n_solutions))
    
    if n_solutions == 1:
        axes = axes.reshape(1, -1)
    
    for i, ((u, v, p), label) in enumerate(zip(solutions, labels)):
        vel_mag = np.sqrt(u**2 + v**2)
        
        for j, (field, name) in enumerate([(vel_mag, '|u|'), (u, 'u'), (p, 'p')]):
            ax = axes[i, j]
            cf = ax.contourf(x, y, field, levels=50, cmap='rainbow')
            if show_circle:
                cx, cy, r = show_circle
                circle = plt.Circle((cx, cy), r, fc='black')
                ax.add_patch(circle)
            ax.set_title(f'{label}: {name}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')
            plt.colorbar(cf, ax=ax)
    
    plt.tight_layout()
    
    if save_path is None and simulation_type and Re is not None:
        save_path = generate_filename(simulation_type, 'comparison', Re, 'comparison')
    
    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Saved figure to {save_path}")
    
    plt.close(fig)


def plot_streamlines(u, v, x=None, y=None, density=1.5, save_path=None,
                     show_circle=None, title='Streamlines',
                     simulation_type=None, mode=None, Re=None):
    """
    Plot streamlines of the velocity field.
    
    Parameters:
    -----------
    u, v : ndarray
        Velocity components
    x, y : ndarray, optional
        Coordinate arrays
    density : float
        Streamline density
    save_path : str, optional
        Path to save figure
    show_circle : tuple, optional
        (cx, cy, r) for cylinder visualization
    title : str
        Plot title
    simulation_type, mode, Re : optional
        For auto-generating filename
    """
    Ny, Nx = u.shape
    if x is None or y is None:
        x = np.linspace(0, 1, Nx)
        y = np.linspace(0, 1, Ny)
        X, Y = np.meshgrid(x, y)
    else:
        X, Y = x, y
        x = X[0, :]
        y = Y[:, 0]
    
    vel_mag = np.sqrt(u**2 + v**2)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Background contour of velocity magnitude
    cf = ax.contourf(X, Y, vel_mag, levels=50, cmap='rainbow', alpha=0.8)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    plt.colorbar(cf, ax=ax, label='Velocity magnitude')
    
    # Streamlines using interior-smoothed velocity
    ax.streamplot(x, y, u, v, density=density, color='white', linewidth=0.5)
    
    if show_circle:
        cx, cy, r = show_circle
        circle = plt.Circle((cx, cy), r, fc='black', ec='white', linewidth=2)
        ax.add_patch(circle)
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path is None and simulation_type and mode and Re is not None:
        save_path = generate_filename(simulation_type, mode, Re, 'streamlines')
    
    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Saved figure to {save_path}")
    
    plt.close(fig)


def plot_region_mask(mask, x=None, y=None, save_path=None, show_circle=None,
                     title='Solver Region Assignment', figsize=(12, 8),
                     simulation_type=None, mode=None, Re=None):
    """
    Plot black and white visualization of CFD vs PINN regions.
    
    Parameters:
    -----------
    mask : ndarray
        Binary mask (1 = CFD region, 0 = PINN region)
    x, y : ndarray, optional
        Coordinate arrays. If None, uses indices.
    save_path : str, optional
        Path to save figure
    show_circle : tuple, optional
        (cx, cy, r) for cylinder visualization
    title : str
        Plot title
    figsize : tuple
        Figure size
    simulation_type, mode, Re : optional
        For auto-generating filename
    """
    Ny, Nx = mask.shape
    if x is None or y is None:
        x = np.linspace(0, 1, Nx)
        y = np.linspace(0, 1, Ny)
        X, Y = np.meshgrid(x, y)
    else:
        X, Y = x, y
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot mask: white = PINN (mask=1), black = CFD (mask=0)
    im = ax.pcolormesh(X, Y, mask, cmap='binary', vmin=0, vmax=1, shading='auto')
    
    # Add contour line at interface
    ax.contour(X, Y, mask, levels=[0.5], colors='red', linewidths=2, linestyles='-')
    
    if show_circle:
        cx, cy, r = show_circle
        circle = plt.Circle((cx, cy), r, fc='gray', ec='red', linewidth=2)
        ax.add_patch(circle)
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.set_aspect('equal')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', label='PINN Region'),
        Patch(facecolor='black', edgecolor='black', label='CFD Region'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Add statistics text
    cfd_pct = 100 * np.sum(mask) / mask.size
    pinn_pct = 100 - cfd_pct
    stats_text = f'CFD: {cfd_pct:.1f}%  |  PINN: {pinn_pct:.1f}%'
    ax.text(0.5, -0.08, stats_text, transform=ax.transAxes, 
            ha='center', fontsize=12, style='italic')
    
    plt.tight_layout()
    
    if save_path is None and simulation_type and mode and Re is not None:
        save_path = generate_filename(simulation_type, mode, Re, 'regions')
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.close(fig)