"""
PDE Layers for Continuous-Time Attention

This module implements various PDE-guided refinement layers that can be inserted
into Transformer architectures to model continuous-time token interactions.
"""

import torch
import torch.nn as nn


class DiffusionPDELayer(nn.Module):
    """
    Diffusion PDE Layer: du/dt = alpha * d²u/dx²
    
    Models token interactions as a diffusion process where information
    smoothly propagates across the sequence.
    
    Args:
        hidden_size (int): Hidden dimension size (not used in current implementation)
        alpha_init (float): Initial value for diffusion coefficient
    """
    def __init__(self, hidden_size, alpha_init=0.10):
        super(DiffusionPDELayer, self).__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha_init]))
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, hidden_dim, seq_len)
        
        Returns:
            Updated tensor of shape (batch_size, hidden_dim, seq_len)
        """
        # Need at least 3 positions to compute Laplacian
        if x.size(2) < 3:
            return x
        
        # Compute discrete Laplacian: ∇²u ≈ u[i+1] - 2*u[i] + u[i-1]
        laplacian = x[:, :, 2:] - 2 * x[:, :, 1:-1] + x[:, :, :-2]
        
        # Apply PDE update: u_new = u_old + alpha * ∇²u
        dx = self.alpha * laplacian
        
        # Preserve boundary conditions and update interior
        return torch.cat([x[:, :, :1], x[:, :, 1:-1] + dx, x[:, :, -1:]], dim=2)


class WavePDELayer(nn.Module):
    """
    Wave PDE Layer: d²u/dt² = c² * d²u/dx²
    
    Models token interactions as wave propagation, allowing for
    oscillatory information flow patterns.
    
    Args:
        hidden_size (int): Hidden dimension size
        alpha_init (float): Initial value for wave speed parameter
    """
    def __init__(self, hidden_size, alpha_init=0.15):
        super(WavePDELayer, self).__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha_init]))
    
    def forward(self, x):
        if x.size(2) < 3:
            return x
        
        laplacian = x[:, :, 2:] - 2 * x[:, :, 1:-1] + x[:, :, :-2]
        
        # Wave equation: use alpha² as c² (wave speed squared)
        dx = self.alpha ** 2 * laplacian
        
        return torch.cat([x[:, :, :1], x[:, :, 1:-1] + dx, x[:, :, -1:]], dim=2)


class ReactionDiffusionPDELayer(nn.Module):
    """
    Reaction-Diffusion PDE Layer: du/dt = alpha * d²u/dx² + beta * f(u)
    
    Combines diffusion with reaction terms, allowing for more complex
    dynamics including pattern formation and nonlinear interactions.
    
    Args:
        hidden_size (int): Hidden dimension size
        alpha_init (float): Initial diffusion coefficient
        beta_init (float): Initial reaction coefficient
    """
    def __init__(self, hidden_size, alpha_init=0.10, beta_init=0.02):
        super(ReactionDiffusionPDELayer, self).__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha_init]))
        self.beta = nn.Parameter(torch.tensor([beta_init]))
    
    def forward(self, x):
        if x.size(2) < 3:
            return x
        
        laplacian = x[:, :, 2:] - 2 * x[:, :, 1:-1] + x[:, :, :-2]
        
        # Diffusion term
        diffusion = self.alpha * laplacian
        
        # Reaction term: f(u) = u * (1 - u) (Fisher-KPP type)
        u = x[:, :, 1:-1]
        reaction = self.beta * u * (1 - torch.sigmoid(u))
        
        # Combined update
        dx = diffusion + reaction
        
        return torch.cat([x[:, :, :1], x[:, :, 1:-1] + dx, x[:, :, -1:]], dim=2)


class AdvectionDiffusionPDELayer(nn.Module):
    """
    Advection-Diffusion PDE Layer: du/dt = alpha * d²u/dx² + beta * du/dx
    
    Combines diffusion with advection (directional transport), useful for
    modeling information flow with preferred directions.
    
    Args:
        hidden_size (int): Hidden dimension size
        alpha_init (float): Initial diffusion coefficient
        beta_init (float): Initial advection coefficient
    """
    def __init__(self, hidden_size, alpha_init=0.10, beta_init=0.03):
        super(AdvectionDiffusionPDELayer, self).__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha_init]))
        self.beta = nn.Parameter(torch.tensor([beta_init]))
    
    def forward(self, x):
        if x.size(2) < 3:
            return x
        
        # Diffusion: second-order derivative
        laplacian = x[:, :, 2:] - 2 * x[:, :, 1:-1] + x[:, :, :-2]
        diffusion = self.alpha * laplacian
        
        # Advection: first-order derivative (central difference)
        gradient = (x[:, :, 2:] - x[:, :, :-2]) / 2.0
        advection = self.beta * gradient
        
        # Combined update
        dx = diffusion + advection
        
        return torch.cat([x[:, :, :1], x[:, :, 1:-1] + dx, x[:, :, -1:]], dim=2)


def create_pde_layer(pde_type, hidden_size, **kwargs):
    """
    Factory function to create PDE layers.
    
    Args:
        pde_type (str): Type of PDE ('diffusion', 'wave', 'reaction-diffusion', 'advection-diffusion')
        hidden_size (int): Hidden dimension size
        **kwargs: Additional arguments for specific PDE layers
    
    Returns:
        nn.Module: Instantiated PDE layer
    """
    pde_layers = {
        'diffusion': DiffusionPDELayer,
        'wave': WavePDELayer,
        'reaction-diffusion': ReactionDiffusionPDELayer,
        'advection-diffusion': AdvectionDiffusionPDELayer,
    }
    
    if pde_type not in pde_layers:
        raise ValueError(f"Unknown PDE type: {pde_type}. Available types: {list(pde_layers.keys())}")
    
    return pde_layers[pde_type](hidden_size, **kwargs)

