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
        causal (bool): If True, use causal (past-only) stencil. Default: False
    """
    def __init__(self, hidden_size, alpha_init=0.10, layout: str = "bdl", causal: bool = False):
        super(DiffusionPDELayer, self).__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha_init]))
        if layout not in {"bdl", "bld"}:
            raise ValueError("layout must be 'bdl' or 'bld'")
        self.layout = layout
        self.causal = causal
    
    def forward(self, x, attention_mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, hidden_dim, seq_len) if layout='bdl'
               or (batch_size, seq_len, hidden_dim) if layout='bld'
            attention_mask: Optional (batch_size, seq_len), 1 for valid tokens, 0 for padding.
                           Used to mask padding regions and avoid boundary pollution.
        
        Returns:
            Updated tensor with the same layout as input.
        """
        # Need at least 3 positions to compute Laplacian (non-causal) or 2 (causal)
        seq_dim = 2 if self.layout == "bdl" else 1
        min_seq_len = 2 if self.causal else 3
        if x.size(seq_dim) < min_seq_len:
            return x

        # fp32 accumulate for numerical stability; cast back to input dtype.
        x_fp32 = x.to(dtype=torch.float32)
        device = x.device

        if self.causal:
            # Causal stencil: use backward difference for second derivative
            # For position i, we use i, i-1, i-2 (only past information)
            if self.layout == "bdl":
                # Second derivative using backward difference: f''(i) ≈ f(i) - 2f(i-1) + f(i-2)
                # For positions >= 2, we can compute
                if x.size(2) >= 3:
                    # For positions 2:, use backward difference
                    laplacian = x_fp32[:, :, 2:] - 2.0 * x_fp32[:, :, 1:-1] + x_fp32[:, :, :-2]
                    dx = self.alpha.to(dtype=torch.float32) * laplacian
                    # Positions 0 and 1 are unchanged (boundary conditions)
                    y_fp32 = torch.cat(
                        [x_fp32[:, :, :2], x_fp32[:, :, 2:-1] + dx, x_fp32[:, :, -1:]],
                        dim=2,
                    )
                else:
                    # Too short for causal update, return as is
                    return x_fp32.to(dtype=x.dtype)
            else:
                if x.size(1) >= 3:
                    laplacian = x_fp32[:, 2:, :] - 2.0 * x_fp32[:, 1:-1, :] + x_fp32[:, :-2, :]
                    dx = self.alpha.to(dtype=torch.float32) * laplacian
                    y_fp32 = torch.cat(
                        [x_fp32[:, :2, :], x_fp32[:, 2:-1, :] + dx, x_fp32[:, -1:, :]],
                        dim=1,
                    )
                else:
                    return x_fp32.to(dtype=x.dtype)
        else:
            # Non-causal: symmetric stencil (can use future information)
            if self.layout == "bdl":
                laplacian = x_fp32[:, :, 2:] - 2.0 * x_fp32[:, :, 1:-1] + x_fp32[:, :, :-2]
                dx = self.alpha.to(dtype=torch.float32) * laplacian
                y_fp32 = torch.cat(
                    [x_fp32[:, :, :1], x_fp32[:, :, 1:-1] + dx, x_fp32[:, :, -1:]],
                    dim=2,
                )
            else:
                laplacian = x_fp32[:, 2:, :] - 2.0 * x_fp32[:, 1:-1, :] + x_fp32[:, :-2, :]
                dx = self.alpha.to(dtype=torch.float32) * laplacian
                y_fp32 = torch.cat(
                    [x_fp32[:, :1, :], x_fp32[:, 1:-1, :] + dx, x_fp32[:, -1:, :]],
                    dim=1,
                )
        
        # Apply mask-aware update: don't update padding positions
        if attention_mask is not None:
            if self.layout == "bdl":
                # attention_mask is (batch, seq_len), need to expand to (batch, hidden_dim, seq_len)
                mask_expanded = attention_mask.unsqueeze(1).to(dtype=x_fp32.dtype)  # (batch, 1, seq_len)
                y_fp32 = y_fp32 * mask_expanded + x_fp32 * (1 - mask_expanded)
            else:
                # attention_mask is (batch, seq_len), need to expand to (batch, seq_len, hidden_dim)
                mask_expanded = attention_mask.unsqueeze(-1).to(dtype=x_fp32.dtype)  # (batch, seq_len, 1)
                y_fp32 = y_fp32 * mask_expanded + x_fp32 * (1 - mask_expanded)
        
        return y_fp32.to(dtype=x.dtype)


class WavePDELayer(nn.Module):
    """
    Wave PDE Layer: d²u/dt² = c² * d²u/dx²
    
    Models token interactions as wave propagation, allowing for
    oscillatory information flow patterns.
    
    Args:
        hidden_size (int): Hidden dimension size
        alpha_init (float): Initial value for wave speed parameter
        causal (bool): If True, use causal (past-only) stencil. Default: False
    """
    def __init__(self, hidden_size, alpha_init=0.15, layout: str = "bdl", causal: bool = False):
        super(WavePDELayer, self).__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha_init]))
        if layout not in {"bdl", "bld"}:
            raise ValueError("layout must be 'bdl' or 'bld'")
        self.layout = layout
        self.causal = causal
    
    def forward(self, x, attention_mask=None):
        seq_dim = 2 if self.layout == "bdl" else 1
        min_seq_len = 2 if self.causal else 3
        if x.size(seq_dim) < min_seq_len:
            return x

        x_fp32 = x.to(dtype=torch.float32)
        
        if self.causal:
            # Causal: use backward difference
            if self.layout == "bdl":
                if x.size(2) >= 3:
                    laplacian = x_fp32[:, :, 2:] - 2.0 * x_fp32[:, :, 1:-1] + x_fp32[:, :, :-2]
                    alpha_fp32 = self.alpha.to(dtype=torch.float32)
                    dx = alpha_fp32**2 * laplacian
                    y_fp32 = torch.cat(
                        [x_fp32[:, :, :2], x_fp32[:, :, 2:-1] + dx, x_fp32[:, :, -1:]],
                        dim=2,
                    )
                else:
                    return x_fp32.to(dtype=x.dtype)
            else:
                if x.size(1) >= 3:
                    laplacian = x_fp32[:, 2:, :] - 2.0 * x_fp32[:, 1:-1, :] + x_fp32[:, :-2, :]
                    alpha_fp32 = self.alpha.to(dtype=torch.float32)
                    dx = alpha_fp32**2 * laplacian
                    y_fp32 = torch.cat(
                        [x_fp32[:, :2, :], x_fp32[:, 2:-1, :] + dx, x_fp32[:, -1:, :]],
                        dim=1,
                    )
                else:
                    return x_fp32.to(dtype=x.dtype)
        else:
            if self.layout == "bdl":
                laplacian = x_fp32[:, :, 2:] - 2.0 * x_fp32[:, :, 1:-1] + x_fp32[:, :, :-2]
            else:
                laplacian = x_fp32[:, 2:, :] - 2.0 * x_fp32[:, 1:-1, :] + x_fp32[:, :-2, :]

            # Wave equation: use alpha² as c² (wave speed squared)
            alpha_fp32 = self.alpha.to(dtype=torch.float32)
            dx = alpha_fp32**2 * laplacian

            if self.layout == "bdl":
                y_fp32 = torch.cat(
                    [x_fp32[:, :, :1], x_fp32[:, :, 1:-1] + dx, x_fp32[:, :, -1:]],
                    dim=2,
                )
            else:
                y_fp32 = torch.cat(
                    [x_fp32[:, :1, :], x_fp32[:, 1:-1, :] + dx, x_fp32[:, -1:, :]],
                    dim=1,
                )
        
        # Apply mask-aware update
        if attention_mask is not None:
            if self.layout == "bdl":
                mask_expanded = attention_mask.unsqueeze(1).to(dtype=x_fp32.dtype)
                y_fp32 = y_fp32 * mask_expanded + x_fp32 * (1 - mask_expanded)
            else:
                mask_expanded = attention_mask.unsqueeze(-1).to(dtype=x_fp32.dtype)
                y_fp32 = y_fp32 * mask_expanded + x_fp32 * (1 - mask_expanded)
        
        return y_fp32.to(dtype=x.dtype)


class ReactionDiffusionPDELayer(nn.Module):
    """
    Reaction-Diffusion PDE Layer: du/dt = alpha * d²u/dx² + beta * f(u)
    
    Combines diffusion with reaction terms, allowing for more complex
    dynamics including pattern formation and nonlinear interactions.
    
    Args:
        hidden_size (int): Hidden dimension size
        alpha_init (float): Initial diffusion coefficient
        beta_init (float): Initial reaction coefficient
        causal (bool): If True, use causal (past-only) stencil. Default: False
    """
    def __init__(self, hidden_size, alpha_init=0.10, beta_init=0.02, layout: str = "bdl", causal: bool = False):
        super(ReactionDiffusionPDELayer, self).__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha_init]))
        self.beta = nn.Parameter(torch.tensor([beta_init]))
        if layout not in {"bdl", "bld"}:
            raise ValueError("layout must be 'bdl' or 'bld'")
        self.layout = layout
        self.causal = causal
    
    def forward(self, x, attention_mask=None):
        seq_dim = 2 if self.layout == "bdl" else 1
        min_seq_len = 2 if self.causal else 3
        if x.size(seq_dim) < min_seq_len:
            return x

        x_fp32 = x.to(dtype=torch.float32)
        
        if self.causal:
            if self.layout == "bdl":
                if x.size(2) >= 3:
                    laplacian = x_fp32[:, :, 2:] - 2.0 * x_fp32[:, :, 1:-1] + x_fp32[:, :, :-2]
                    u = x_fp32[:, :, 1:-1]
                    diffusion = self.alpha.to(dtype=torch.float32) * laplacian
                    reaction = self.beta.to(dtype=torch.float32) * u * (1.0 - torch.sigmoid(u))
                    dx = diffusion + reaction
                    y_fp32 = torch.cat(
                        [x_fp32[:, :, :2], x_fp32[:, :, 2:-1] + dx, x_fp32[:, :, -1:]],
                        dim=2,
                    )
                else:
                    return x_fp32.to(dtype=x.dtype)
            else:
                if x.size(1) >= 3:
                    laplacian = x_fp32[:, 2:, :] - 2.0 * x_fp32[:, 1:-1, :] + x_fp32[:, :-2, :]
                    u = x_fp32[:, 1:-1, :]
                    diffusion = self.alpha.to(dtype=torch.float32) * laplacian
                    reaction = self.beta.to(dtype=torch.float32) * u * (1.0 - torch.sigmoid(u))
                    dx = diffusion + reaction
                    y_fp32 = torch.cat(
                        [x_fp32[:, :2, :], x_fp32[:, 2:-1, :] + dx, x_fp32[:, -1:, :]],
                        dim=1,
                    )
                else:
                    return x_fp32.to(dtype=x.dtype)
        else:
            if self.layout == "bdl":
                laplacian = x_fp32[:, :, 2:] - 2.0 * x_fp32[:, :, 1:-1] + x_fp32[:, :, :-2]
                u = x_fp32[:, :, 1:-1]
            else:
                laplacian = x_fp32[:, 2:, :] - 2.0 * x_fp32[:, 1:-1, :] + x_fp32[:, :-2, :]
                u = x_fp32[:, 1:-1, :]

            # Diffusion term
            diffusion = self.alpha.to(dtype=torch.float32) * laplacian

            # Reaction term: f(u) = u * (1 - u) (Fisher-KPP type)
            reaction = self.beta.to(dtype=torch.float32) * u * (1.0 - torch.sigmoid(u))

            # Combined update
            dx = diffusion + reaction

            if self.layout == "bdl":
                y_fp32 = torch.cat(
                    [x_fp32[:, :, :1], x_fp32[:, :, 1:-1] + dx, x_fp32[:, :, -1:]],
                    dim=2,
                )
            else:
                y_fp32 = torch.cat(
                    [x_fp32[:, :1, :], x_fp32[:, 1:-1, :] + dx, x_fp32[:, -1:, :]],
                    dim=1,
                )
        
        # Apply mask-aware update
        if attention_mask is not None:
            if self.layout == "bdl":
                mask_expanded = attention_mask.unsqueeze(1).to(dtype=x_fp32.dtype)
                y_fp32 = y_fp32 * mask_expanded + x_fp32 * (1 - mask_expanded)
            else:
                mask_expanded = attention_mask.unsqueeze(-1).to(dtype=x_fp32.dtype)
                y_fp32 = y_fp32 * mask_expanded + x_fp32 * (1 - mask_expanded)
        
        return y_fp32.to(dtype=x.dtype)


class AdvectionDiffusionPDELayer(nn.Module):
    """
    Advection-Diffusion PDE Layer: du/dt = alpha * d²u/dx² + beta * du/dx
    
    Combines diffusion with advection (directional transport), useful for
    modeling information flow with preferred directions.
    
    Args:
        hidden_size (int): Hidden dimension size
        alpha_init (float): Initial diffusion coefficient
        beta_init (float): Initial advection coefficient
        causal (bool): If True, use causal (past-only) stencil. Default: False
    """
    def __init__(self, hidden_size, alpha_init=0.10, beta_init=0.03, layout: str = "bdl", causal: bool = False):
        super(AdvectionDiffusionPDELayer, self).__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha_init]))
        self.beta = nn.Parameter(torch.tensor([beta_init]))
        if layout not in {"bdl", "bld"}:
            raise ValueError("layout must be 'bdl' or 'bld'")
        self.layout = layout
        self.causal = causal
    
    def forward(self, x, attention_mask=None):
        seq_dim = 2 if self.layout == "bdl" else 1
        min_seq_len = 2 if self.causal else 3
        if x.size(seq_dim) < min_seq_len:
            return x

        x_fp32 = x.to(dtype=torch.float32)

        if self.causal:
            # Causal: use backward differences
            if self.layout == "bdl":
                if x.size(2) >= 3:
                    # Diffusion: backward difference for second derivative
                    laplacian = x_fp32[:, :, 2:] - 2.0 * x_fp32[:, :, 1:-1] + x_fp32[:, :, :-2]
                    # Advection: backward difference for first derivative
                    gradient = x_fp32[:, :, 1:-1] - x_fp32[:, :, :-2]
                    diffusion = self.alpha.to(dtype=torch.float32) * laplacian
                    advection = self.beta.to(dtype=torch.float32) * gradient
                    dx = diffusion + advection
                    y_fp32 = torch.cat(
                        [x_fp32[:, :, :2], x_fp32[:, :, 2:-1] + dx, x_fp32[:, :, -1:]],
                        dim=2,
                    )
                else:
                    return x_fp32.to(dtype=x.dtype)
            else:
                if x.size(1) >= 3:
                    laplacian = x_fp32[:, 2:, :] - 2.0 * x_fp32[:, 1:-1, :] + x_fp32[:, :-2, :]
                    gradient = x_fp32[:, 1:-1, :] - x_fp32[:, :-2, :]
                    diffusion = self.alpha.to(dtype=torch.float32) * laplacian
                    advection = self.beta.to(dtype=torch.float32) * gradient
                    dx = diffusion + advection
                    y_fp32 = torch.cat(
                        [x_fp32[:, :2, :], x_fp32[:, 2:-1, :] + dx, x_fp32[:, -1:, :]],
                        dim=1,
                    )
                else:
                    return x_fp32.to(dtype=x.dtype)
        else:
            # Non-causal: use central differences
            if self.layout == "bdl":
                laplacian = x_fp32[:, :, 2:] - 2.0 * x_fp32[:, :, 1:-1] + x_fp32[:, :, :-2]
                gradient = (x_fp32[:, :, 2:] - x_fp32[:, :, :-2]) / 2.0
            else:
                laplacian = x_fp32[:, 2:, :] - 2.0 * x_fp32[:, 1:-1, :] + x_fp32[:, :-2, :]
                gradient = (x_fp32[:, 2:, :] - x_fp32[:, :-2, :]) / 2.0
            diffusion = self.alpha.to(dtype=torch.float32) * laplacian

            # Advection: first-order derivative (central difference)
            advection = self.beta.to(dtype=torch.float32) * gradient

            # Combined update
            dx = diffusion + advection

            if self.layout == "bdl":
                y_fp32 = torch.cat(
                    [x_fp32[:, :, :1], x_fp32[:, :, 1:-1] + dx, x_fp32[:, :, -1:]],
                    dim=2,
                )
            else:
                y_fp32 = torch.cat(
                    [x_fp32[:, :1, :], x_fp32[:, 1:-1, :] + dx, x_fp32[:, -1:, :]],
                    dim=1,
                )
        
        # Apply mask-aware update
        if attention_mask is not None:
            if self.layout == "bdl":
                mask_expanded = attention_mask.unsqueeze(1).to(dtype=x_fp32.dtype)
                y_fp32 = y_fp32 * mask_expanded + x_fp32 * (1 - mask_expanded)
            else:
                mask_expanded = attention_mask.unsqueeze(-1).to(dtype=x_fp32.dtype)
                y_fp32 = y_fp32 * mask_expanded + x_fp32 * (1 - mask_expanded)
        
        return y_fp32.to(dtype=x.dtype)


def create_pde_layer(pde_type, hidden_size, causal: bool = False, **kwargs):
    """
    Factory function to create PDE layers.
    
    Args:
        pde_type (str): Type of PDE ('diffusion', 'wave', 'reaction-diffusion', 'advection-diffusion')
        hidden_size (int): Hidden dimension size
        causal (bool): If True, use causal (past-only) stencil for autoregressive LM. Default: False
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
    
    # Pass causal parameter to all PDE layers
    kwargs['causal'] = causal
    return pde_layers[pde_type](hidden_size, **kwargs)

