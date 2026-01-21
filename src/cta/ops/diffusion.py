"""
Diffusion stencil ops (reference + utilities).

We intentionally implement a small, well-tested reference that matches the
boundary behavior used in `models.pde_layers.DiffusionPDELayer`:
- copy boundary: y[..., 0] = x[..., 0], y[..., -1] = x[..., -1]
- update interior with discrete Laplacian.
"""

from __future__ import annotations

from typing import Optional

import torch


def diffusion_step_reference(
    x_bdl: torch.Tensor,
    *,
    alpha: float,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Single diffusion step on a `[B, D, L]` tensor (copy boundary).

    Implements:
        y[i] = x[i] + alpha * (x[i+1] - 2*x[i] + x[i-1])

    Args:
        x_bdl: Input of shape `[B, D, L]`.
        alpha: Diffusion coefficient.
        out_dtype: Output dtype. Defaults to `x_bdl.dtype`.

    Returns:
        Tensor of shape `[B, D, L]`.
    """
    if x_bdl.ndim != 3:
        raise ValueError(f"expected x_bdl.ndim == 3, got {x_bdl.ndim}")

    out_dtype = x_bdl.dtype if out_dtype is None else out_dtype

    b, d, l = x_bdl.shape
    if l < 3:
        return x_bdl.to(dtype=out_dtype)

    # fp32 accumulate for numerical stability (bf16/fp16 friendly).
    x = x_bdl.to(dtype=torch.float32)

    left = x[:, :, :-2]
    mid = x[:, :, 1:-1]
    right = x[:, :, 2:]

    lap = right - 2.0 * mid + left
    y_mid = mid + float(alpha) * lap

    y = torch.cat([x[:, :, :1], y_mid, x[:, :, -1:]], dim=2)
    return y.to(dtype=out_dtype)


def diffusion_steps_reference(
    x_bdl: torch.Tensor,
    *,
    alpha: float,
    steps: int,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Multiple diffusion steps by looping the single-step reference."""
    if steps < 0:
        raise ValueError(f"expected steps >= 0, got {steps}")
    if steps == 0:
        out_dtype = x_bdl.dtype if out_dtype is None else out_dtype
        return x_bdl.to(dtype=out_dtype)

    y = x_bdl
    for _ in range(steps):
        y = diffusion_step_reference(y, alpha=alpha, out_dtype=torch.float32)
    out_dtype = x_bdl.dtype if out_dtype is None else out_dtype
    return y.to(dtype=out_dtype)

