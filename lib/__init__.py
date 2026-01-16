"""
PINN-CFD Hybrid Fluid Dynamics Library

This library provides tools for solving fluid dynamics problems using
hybrid Physics-Informed Neural Networks (PINN) and Computational Fluid
Dynamics (CFD) methods.

Modules:
- base_simulation: Abstract base class for simulations
- cavity_flow: Lid-driven cavity flow simulations
- cylinder_flow: Flow around cylinder simulations
- network: Neural network architectures
- rejector_network: Rejector network for CFD/PINN region selection
- plotting: Visualization utilities
"""

from lib.base_simulation import BaseSimulation
from lib.cavity_flow import (
    CavityFlowSimulation,
    CavityFlowHybridSimulation,
    create_center_pinn_mask,
    create_boundary_pinn_mask,
    create_custom_mask,
)
from lib.cylinder_flow import (
    CylinderFlowSimulation,
    CylinderFlowHybridSimulation,
    CylinderFlowPINNSimulation,
    create_cylinder_boundary_mask,
    create_cylinder_wake_mask,
)
from lib.network import Network
from lib.rejector_network import (
    RejectorNetwork,
    RejectorLoss,
    compute_weighted_loss,
    create_rejector_training_data,
)
from lib.plotting import (
    generate_filename,
    plot_contour,
    plot_solution,
    plot_single_field,
    plot_hybrid_solution,
    plot_comparison,
    plot_streamlines,
)

__all__ = [
    # Base classes
    'BaseSimulation',
    
    # Cavity flow
    'CavityFlowSimulation',
    'CavityFlowHybridSimulation',
    'create_center_pinn_mask',
    'create_boundary_pinn_mask',
    'create_custom_mask',
    
    # Cylinder flow
    'CylinderFlowSimulation',
    'CylinderFlowHybridSimulation',
    'CylinderFlowPINNSimulation',
    'create_cylinder_boundary_mask',
    'create_cylinder_wake_mask',
    
    # Network
    'Network',
    
    # Rejector
    'RejectorNetwork',
    'RejectorLoss',
    'compute_weighted_loss',
    'create_rejector_training_data',
    
    # Plotting
    'generate_filename',
    'plot_contour',
    'plot_solution',
    'plot_single_field',
    'plot_hybrid_solution',
    'plot_comparison',
    'plot_streamlines',
]
