"""
Rejector Network for CFD/PINN Region Selection.

This module implements a neural network that learns to predict which regions
of the domain should be solved by CFD vs PINN, balancing computational cost
and solution accuracy.
"""

import tensorflow as tf
import numpy as np


class RejectorNetwork:
    """
    Neural network that predicts probability of using CFD vs PINN at each point.
    
    Input: (x, y, Re) - spatial coordinates and Reynolds number
    Output: p ∈ [0, 1] - probability of using CFD (1 = CFD, 0 = PINN)
    
    The boundary regions are always assigned to CFD regardless of prediction.
    """
    
    def __init__(self, boundary_mask_fn=None):
        """
        Initialize the rejector network.
        
        Parameters:
        -----------
        boundary_mask_fn : callable, optional
            Function that returns boundary mask given grid parameters.
            Boundary points are always assigned to CFD.
        """
        self.boundary_mask_fn = boundary_mask_fn
        self.model = None
        self.history = None
        
    def build(self, num_inputs=3, layers=[64, 64, 64, 32], activation='swish'):
        """
        Build the rejector network.
        
        Parameters:
        -----------
        num_inputs : int
            Number of inputs (default 3 for x, y, Re)
        layers : list
            Hidden layer sizes
        activation : str
            Activation function ('swish', 'tanh', 'relu')
            
        Returns:
        --------
        tf.keras.Model
        """
        # Input layer
        inputs = tf.keras.layers.Input(shape=(num_inputs,), name='input')
        
        # Normalize inputs (especially Re which can have large range)
        x = inputs
        
        # Hidden layers
        for i, units in enumerate(layers):
            x = tf.keras.layers.Dense(
                units, 
                activation=activation,
                kernel_initializer='he_normal',
                name=f'hidden_{i}'
            )(x)
            # Add dropout for regularization
            if i < len(layers) - 1:
                x = tf.keras.layers.Dropout(0.1)(x)
        
        # Output layer with sigmoid for probability
        outputs = tf.keras.layers.Dense(
            1, 
            activation='sigmoid',
            kernel_initializer='glorot_uniform',
            name='output'
        )(x)
        
        self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name='rejector')
        return self.model
    
    def predict_probability(self, xy_re, batch_size=4096):
        """
        Predict CFD probability for given points.
        
        Parameters:
        -----------
        xy_re : ndarray of shape (N, 3)
            Input coordinates (x, y, Re)
            
        Returns:
        --------
        prob : ndarray of shape (N,)
            Probability of using CFD at each point
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        return self.model.predict(xy_re, batch_size=batch_size).flatten()
    
    def predict_mask(self, xy_re, threshold=0.5, batch_size=4096):
        """
        Predict binary mask for CFD/PINN regions.
        
        Parameters:
        -----------
        xy_re : ndarray of shape (N, 3)
            Input coordinates (x, y, Re)
        threshold : float
            Threshold for converting probability to binary mask
            
        Returns:
        --------
        mask : ndarray of shape (N,)
            Binary mask (1 = CFD, 0 = PINN)
        """
        prob = self.predict_probability(xy_re, batch_size=batch_size)
        return (prob >= threshold).astype(np.int32)
    
    def predict_mask_grid(self, X, Y, Re, threshold=0.5, enforce_boundary=True, 
                          boundary_width=5):
        """
        Predict mask for a 2D grid.
        
        Parameters:
        -----------
        X, Y : ndarray
            Meshgrid coordinates
        Re : float
            Reynolds number
        threshold : float
            Threshold for binary mask
        enforce_boundary : bool
            Whether to enforce CFD at domain boundaries
        boundary_width : int
            Width of boundary region to enforce as CFD
            
        Returns:
        --------
        mask : ndarray
            2D binary mask (1 = CFD, 0 = PINN)
        """
        Ny, Nx = X.shape
        
        # Flatten and create input
        xy_re = np.column_stack([
            X.flatten(), 
            Y.flatten(), 
            np.full(X.size, Re)
        ])
        
        # Predict
        prob = self.predict_probability(xy_re)
        mask = (prob >= threshold).astype(np.int32).reshape(Ny, Nx)
        
        # Enforce boundary conditions
        if enforce_boundary:
            mask[:boundary_width, :] = 1   # Bottom
            mask[-boundary_width:, :] = 1  # Top
            mask[:, :boundary_width] = 1   # Left
            mask[:, -boundary_width:] = 1  # Right
        
        return mask
    
    def save(self, filepath):
        """Save model weights."""
        if self.model is None:
            raise ValueError("No model to save.")
        self.model.save_weights(filepath)
        print(f"Saved rejector model to {filepath}")
    
    def load(self, filepath):
        """Load model weights."""
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        self.model.load_weights(filepath)
        print(f"Loaded rejector model from {filepath}")
    
    def summary(self):
        """Print model summary."""
        if self.model is not None:
            self.model.summary()


class RejectorLoss(tf.keras.losses.Loss):
    """
    Custom loss function for rejector training.
    
    Combines:
    1. Solution accuracy: weighted MSE between hybrid and ground truth CFD
    2. Computational cost: penalizes CFD usage (high cost) vs PINN (low cost)
    
    Loss = α * accuracy_loss + β * cost_loss
    
    Where:
    - accuracy_loss = MSE(hybrid, ground_truth)
    - cost_loss = mean(p * c_cfd + (1-p) * c_pinn)
    """
    
    def __init__(self, c_cfd=1.0, c_pinn=0.01, accuracy_weight=1.0, cost_weight=0.1,
                 name='rejector_loss'):
        """
        Initialize loss function.
        
        Parameters:
        -----------
        c_cfd : float
            Cost of using CFD at a point (high)
        c_pinn : float
            Cost of using PINN at a point (low)
        accuracy_weight : float
            Weight for accuracy term (α)
        cost_weight : float
            Weight for cost term (β)
        """
        super().__init__(name=name)
        self.c_cfd = c_cfd
        self.c_pinn = c_pinn
        self.accuracy_weight = accuracy_weight
        self.cost_weight = cost_weight
    
    def call(self, y_true, y_pred):
        """
        Compute loss.
        
        Parameters:
        -----------
        y_true : tensor
            Contains [error_at_point, ground_truth_should_be_cfd] 
            For training, this is computed from the hybrid solution error
        y_pred : tensor
            Predicted probability of using CFD
        """
        # y_true contains the error at each point from hybrid vs ground truth
        error = y_true
        
        # Accuracy loss: encourage CFD where error is high
        # If error is high and p is low (using PINN), loss is high
        accuracy_loss = tf.reduce_mean(error * (1 - y_pred))
        
        # Cost loss: penalize CFD usage
        cost_loss = tf.reduce_mean(y_pred * self.c_cfd + (1 - y_pred) * self.c_pinn)
        
        total_loss = self.accuracy_weight * accuracy_loss + self.cost_weight * cost_loss
        
        return total_loss


def compute_weighted_loss(errors, probs, c_cfd=1.0, c_pinn=0.01, 
                          accuracy_weight=1.0, cost_weight=0.1):
    """
    Compute the rejector loss given errors and predicted probabilities.
    
    This function implements a weighted logistic regression style loss:
    
    L = α * Σ(error_i * (1 - p_i)) + β * Σ(p_i * c_cfd + (1-p_i) * c_pinn)
    
    The first term encourages using CFD (p=1) where errors are high.
    The second term encourages using PINN (p=0) to minimize cost.
    
    Parameters:
    -----------
    errors : ndarray
        Error at each point (|hybrid - ground_truth|)
    probs : ndarray
        Predicted probability of using CFD at each point
    c_cfd : float
        Cost of CFD computation
    c_pinn : float
        Cost of PINN computation
    accuracy_weight : float
        Weight for accuracy term
    cost_weight : float
        Weight for cost term
        
    Returns:
    --------
    loss : float
        Total loss value
    accuracy_loss : float
        Accuracy component
    cost_loss : float
        Cost component
    """
    # Accuracy loss: penalize PINN usage where error is high
    accuracy_loss = np.mean(errors * (1 - probs))
    
    # Cost loss: penalize CFD usage
    cost_loss = np.mean(probs * c_cfd + (1 - probs) * c_pinn)
    
    total_loss = accuracy_weight * accuracy_loss + cost_weight * cost_loss
    
    return total_loss, accuracy_loss, cost_loss


def create_rejector_training_data(xy, Re, errors, normalize=True):
    """
    Create training data for the rejector network.
    
    Parameters:
    -----------
    xy : ndarray of shape (N, 2)
        Spatial coordinates
    Re : float
        Reynolds number for this sample
    errors : ndarray of shape (N,)
        Error at each point from hybrid vs ground truth
    normalize : bool
        Whether to normalize the Reynolds number
        
    Returns:
    --------
    X : ndarray of shape (N, 3)
        Input features (x, y, Re_normalized)
    y : ndarray of shape (N,)
        Target (normalized errors as pseudo-labels)
    """
    N = len(xy)
    
    # Normalize Re to [0, 1] range (assuming Re_max = 1000)
    Re_normalized = Re / 1000.0 if normalize else Re
    
    # Create input features
    X = np.column_stack([
        xy[:, 0],  # x
        xy[:, 1],  # y
        np.full(N, Re_normalized)  # Re
    ])
    
    # Normalize errors to [0, 1] for training stability
    if normalize and np.max(errors) > 0:
        y = errors / np.max(errors)
    else:
        y = errors
    
    return X, y
