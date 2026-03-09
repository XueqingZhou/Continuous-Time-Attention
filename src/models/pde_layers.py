"""
PDE Layers for Continuous-Time Attention

This module implements various PDE-guided refinement layers that can be inserted
into Transformer architectures to model continuous-time token interactions.

Supported PDE types:
  - diffusion:            du/dt = alpha * d^2u/dx^2
  - wave:                 d^2u/dt^2 = c^2 * d^2u/dx^2  (c = alpha)
  - reaction-diffusion:   du/dt = alpha * d^2u/dx^2 + beta * u * (1 - sigmoid(u))
  - advection-diffusion:  du/dt = alpha * d^2u/dx^2 + beta * du/dx
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def _expand_attention_mask(
    attention_mask: torch.Tensor,
    *,
    layout: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Expand ``(B, L)`` mask to broadcast with ``(B, D, L)`` or ``(B, L, D)``."""
    mask = attention_mask.bool().to(dtype=dtype)
    if layout == "bdl":
        return mask.unsqueeze(1)       # (B, 1, L)
    return mask.unsqueeze(-1)          # (B, L, 1)


def _valid_stencil_mask(
    attention_mask: Optional[torch.Tensor],
    *,
    seq_len_out: int,
    layout: str,
) -> Optional[torch.Tensor]:
    """Return a bool mask ``(B, 1, L')`` or ``(B, L', 1)`` indicating positions
    where *all three* stencil neighbours (left, centre, right) are valid.

    ``seq_len_out`` is the length of the updated interior slice so that the
    caller does not need to know whether causal or non-causal slicing was used.
    """
    if attention_mask is None:
        return None
    L = attention_mask.size(1)
    # The stencil always touches three consecutive positions.  We compute the
    # valid mask over all possible (L-2)-length windows, then slice to match
    # the output length (which may differ between causal and non-causal).
    valid_full = (
        attention_mask[:, :-2] * attention_mask[:, 1:-1] * attention_mask[:, 2:]
    )
    # causal keeps the *last* seq_len_out positions, non-causal keeps all L-2.
    valid = valid_full[:, -seq_len_out:]
    if layout == "bdl":
        return valid.unsqueeze(1).bool()   # (B, 1, L')
    return valid.unsqueeze(-1).bool()      # (B, L', 1)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _BasePDELayer(nn.Module):
    """Common infrastructure shared by all PDE layers.

    Subclasses only need to implement :meth:`_compute_update` which returns
    the delta ``dx`` to be added to the interior slice.
    """

    def __init__(
        self,
        hidden_size: int,
        layout: str = "bdl",
        causal: bool = False,
    ):
        super().__init__()
        if layout not in {"bdl", "bld"}:
            raise ValueError("layout must be 'bdl' or 'bld'")
        self.layout = layout
        self.causal = causal

    # -- subclass hook -------------------------------------------------------

    def _compute_update(
        self,
        x_fp32: torch.Tensor,
        laplacian: torch.Tensor,
        interior: torch.Tensor,
    ) -> torch.Tensor:
        """Return the *delta* to add to ``interior``.

        Args:
            x_fp32: Full input in fp32, shape ``(B, D, L)`` (always bdl here).
            laplacian: ``x[i+1] - 2*x[i] + x[i-1]`` for the relevant slice.
            interior: The slice of ``x_fp32`` that will be updated (centre for
                non-causal, trailing for causal).

        Returns:
            Tensor with the same shape as ``interior``.
        """
        raise NotImplementedError

    # -- forward -------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply one PDE refinement step.

        Args:
            x: Input tensor ``(B, D, L)`` if layout='bdl' or ``(B, L, D)``
               if layout='bld'.
            attention_mask: Optional ``(B, L)``, 1 = valid, 0 = padding.

        Returns:
            Updated tensor with the same shape and dtype as *x*.
        """
        seq_dim = 2 if self.layout == "bdl" else 1
        # All stencils touch 3 positions, so we need at least length 3.
        if x.size(seq_dim) < 3:
            return x

        # fp32 accumulate for numerical stability.
        x_fp32 = x.to(dtype=torch.float32)

        # Normalise to BDL internally for cleaner logic; transpose back later.
        if self.layout == "bld":
            x_fp32 = x_fp32.transpose(1, 2)   # -> (B, D, L)

        # Laplacian is always the same three-point stencil.
        lap = x_fp32[:, :, 2:] - 2.0 * x_fp32[:, :, 1:-1] + x_fp32[:, :, :-2]

        if self.causal:
            # Causal: update positions [2:], keep positions [0:2] unchanged.
            interior = x_fp32[:, :, 2:]
            dx = self._compute_update(x_fp32, lap, interior)
            updated = interior + dx

            valid = _valid_stencil_mask(
                attention_mask,
                seq_len_out=updated.size(2),
                layout="bdl",
            )
            if valid is not None:
                updated = torch.where(valid, updated, interior)

            y_fp32 = torch.cat([x_fp32[:, :, :2], updated], dim=2)
        else:
            # Non-causal: update interior positions [1:-1].
            interior = x_fp32[:, :, 1:-1]
            dx = self._compute_update(x_fp32, lap, interior)
            updated = interior + dx

            valid = _valid_stencil_mask(
                attention_mask,
                seq_len_out=updated.size(2),
                layout="bdl",
            )
            if valid is not None:
                updated = torch.where(valid, updated, interior)

            y_fp32 = torch.cat(
                [x_fp32[:, :, :1], updated, x_fp32[:, :, -1:]],
                dim=2,
            )

        # Mask-aware update: don't modify padding positions.
        if attention_mask is not None:
            mask_expanded = _expand_attention_mask(
                attention_mask, layout="bdl", dtype=y_fp32.dtype,
            )
            y_fp32 = y_fp32 * mask_expanded + x_fp32 * (1 - mask_expanded)

        # Transpose back if needed.
        if self.layout == "bld":
            y_fp32 = y_fp32.transpose(1, 2)

        return y_fp32.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# Concrete PDE layers
# ---------------------------------------------------------------------------

class DiffusionPDELayer(_BasePDELayer):
    """Diffusion PDE: ``du/dt = alpha * d^2u/dx^2``

    Models token interactions as a diffusion process where information
    smoothly propagates across the sequence.

    Args:
        hidden_size: Hidden dimension size.
        alpha_init: Initial value for diffusion coefficient.
        layout: Tensor layout, ``'bdl'`` or ``'bld'``.
        causal: If True, use causal (past-only) stencil.
    """

    def __init__(
        self,
        hidden_size: int,
        alpha_init: float = 0.10,
        layout: str = "bdl",
        causal: bool = False,
    ):
        super().__init__(hidden_size, layout=layout, causal=causal)
        self.alpha = nn.Parameter(torch.tensor([alpha_init]))

    def _compute_update(self, x_fp32, laplacian, interior):
        return self.alpha.to(dtype=torch.float32) * laplacian


class WavePDELayer(_BasePDELayer):
    """Wave PDE: ``d^2u/dt^2 = c^2 * d^2u/dx^2``  (c = alpha)

    Models oscillatory information flow patterns.

    Args:
        hidden_size: Hidden dimension size.
        alpha_init: Initial wave speed parameter.
        layout: Tensor layout, ``'bdl'`` or ``'bld'``.
        causal: If True, use causal (past-only) stencil.
    """

    def __init__(
        self,
        hidden_size: int,
        alpha_init: float = 0.15,
        layout: str = "bdl",
        causal: bool = False,
    ):
        super().__init__(hidden_size, layout=layout, causal=causal)
        self.alpha = nn.Parameter(torch.tensor([alpha_init]))

    def _compute_update(self, x_fp32, laplacian, interior):
        alpha_fp32 = self.alpha.to(dtype=torch.float32)
        return alpha_fp32 ** 2 * laplacian


class ReactionDiffusionPDELayer(_BasePDELayer):
    """Reaction-Diffusion PDE: ``du/dt = alpha * d^2u/dx^2 + beta * f(u)``

    Combines diffusion with a Fisher-KPP reaction term
    ``f(u) = u * (1 - sigmoid(u))``, enabling pattern formation and
    nonlinear interactions.

    Args:
        hidden_size: Hidden dimension size.
        alpha_init: Initial diffusion coefficient.
        beta_init: Initial reaction coefficient.
        layout: Tensor layout, ``'bdl'`` or ``'bld'``.
        causal: If True, use causal (past-only) stencil.
    """

    def __init__(
        self,
        hidden_size: int,
        alpha_init: float = 0.10,
        beta_init: float = 0.02,
        layout: str = "bdl",
        causal: bool = False,
    ):
        super().__init__(hidden_size, layout=layout, causal=causal)
        self.alpha = nn.Parameter(torch.tensor([alpha_init]))
        self.beta = nn.Parameter(torch.tensor([beta_init]))

    def _compute_update(self, x_fp32, laplacian, interior):
        diffusion = self.alpha.to(dtype=torch.float32) * laplacian
        reaction = self.beta.to(dtype=torch.float32) * interior * (
            1.0 - torch.sigmoid(interior)
        )
        return diffusion + reaction


class AdvectionDiffusionPDELayer(_BasePDELayer):
    """Advection-Diffusion PDE: ``du/dt = alpha * d^2u/dx^2 + beta * du/dx``

    Combines diffusion with advection (directional transport).

    Args:
        hidden_size: Hidden dimension size.
        alpha_init: Initial diffusion coefficient.
        beta_init: Initial advection coefficient.
        layout: Tensor layout, ``'bdl'`` or ``'bld'``.
        causal: If True, use causal (past-only) stencil.
    """

    def __init__(
        self,
        hidden_size: int,
        alpha_init: float = 0.10,
        beta_init: float = 0.03,
        layout: str = "bdl",
        causal: bool = False,
    ):
        super().__init__(hidden_size, layout=layout, causal=causal)
        self.alpha = nn.Parameter(torch.tensor([alpha_init]))
        self.beta = nn.Parameter(torch.tensor([beta_init]))

    def _compute_update(self, x_fp32, laplacian, interior):
        diffusion = self.alpha.to(dtype=torch.float32) * laplacian
        if self.causal:
            # Backward difference: du/dx ~ x[i] - x[i-1]
            gradient = x_fp32[:, :, 2:] - x_fp32[:, :, 1:-1]
        else:
            # Central difference: du/dx ~ (x[i+1] - x[i-1]) / 2
            gradient = (x_fp32[:, :, 2:] - x_fp32[:, :, :-2]) / 2.0
        advection = self.beta.to(dtype=torch.float32) * gradient
        return diffusion + advection


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PDE_REGISTRY = {
    "diffusion": DiffusionPDELayer,
    "wave": WavePDELayer,
    "reaction-diffusion": ReactionDiffusionPDELayer,
    "advection-diffusion": AdvectionDiffusionPDELayer,
}


def create_pde_layer(
    pde_type: str,
    hidden_size: int,
    causal: bool = False,
    **kwargs,
) -> nn.Module:
    """Factory function to create PDE layers.

    Args:
        pde_type: One of ``'diffusion'``, ``'wave'``,
            ``'reaction-diffusion'``, ``'advection-diffusion'``.
        hidden_size: Hidden dimension size.
        causal: If True, use causal (past-only) stencil for autoregressive LM.
        **kwargs: Additional arguments forwarded to the layer constructor.

    Returns:
        Instantiated PDE layer.
    """
    if pde_type not in _PDE_REGISTRY:
        raise ValueError(
            f"Unknown PDE type: {pde_type}. "
            f"Available types: {list(_PDE_REGISTRY.keys())}"
        )
    kwargs["causal"] = causal
    return _PDE_REGISTRY[pde_type](hidden_size, **kwargs)
