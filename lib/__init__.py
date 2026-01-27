"""
PINN-CFD Hybrid Fluid Dynamics Library

This library provides tools for solving fluid dynamics problems using
hybrid Physics-Informed Neural Networks (PINN) and Computational Fluid
Dynamics (CFD) methods.

Modules:
- base_simulation: Abstract base class for simulations
- cavity_flow: Lid-driven cavity flow simulations
- cylinder_flow: Flow around cylinder simulations
- complexity_scoring: Dynamic segregation based on local complexity scoring
- network: Neural network architectures
- plotting: Visualization utilities
"""

from lib.base_simulation import BaseSimulation
from lib.cavity_flow import (
    CavityFlowSimulation,
    CavityFlowHybridSimulation,
    CavityFlowDynamicHybridSimulation,
    create_center_pinn_mask,
    create_boundary_pinn_mask,
    create_custom_mask,
)
from lib.cylinder_flow import (
    CylinderFlowSimulation,
    CylinderFlowHybridSimulation,
    CylinderFlowDynamicHybridSimulation,
    CylinderFlowPINNSimulation,
    create_cylinder_boundary_mask,
    create_cylinder_wake_mask,
)
from lib.complexity_scoring import (
    ComplexityScorer,
    create_dynamic_mask,
    compute_mask_statistics,
)
from lib.network import Network
from lib.router import (
    RouterCNN,
    RouterTrainer,
    PINNResidualComputer,
    create_router_input,
    create_cylinder_setup,
    plot_router_output,
    plot_training_history,
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
    'CavityFlowDynamicHybridSimulation',
    'create_center_pinn_mask',
    'create_boundary_pinn_mask',
    'create_custom_mask',
    
    # Cylinder flow
    'CylinderFlowSimulation',
    'CylinderFlowHybridSimulation',
    'CylinderFlowDynamicHybridSimulation',
    'CylinderFlowPINNSimulation',
    'create_cylinder_boundary_mask',
    'create_cylinder_wake_mask',
    
    # Complexity scoring
    'ComplexityScorer',
    'create_dynamic_mask',
    'compute_mask_statistics',
    
    # Network
    'Network',
    
    # Router
    'RouterCNN',
    'RouterTrainer',
    'PINNResidualComputer',
    'create_router_input',
    'create_cylinder_setup',
    'plot_router_output',
    'plot_training_history',
    
    # Plotting
    'generate_filename',
    'plot_contour',
    'plot_solution',
    'plot_single_field',
    'plot_hybrid_solution',
    'plot_comparison',
    'plot_streamlines',
]
