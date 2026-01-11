"""
Triton diffusion stencil (forward) - minimal stub.

Implements one diffusion update step with discrete Laplacian along the last dim:
  y[..., i] = x[..., i] + alpha * (x[..., i+1] - 2*x[..., i] + x[..., i-1])
with boundary positions copied (i=0 and i=L-1).

This is intentionally minimal and optional. The main repo does not depend on Triton.
"""

from __future__ import annotations

import argparse
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except Exception as e:  # noqa: BLE001
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = e
else:
    _TRITON_IMPORT_ERROR = None


def _require_triton() -> None:
    if triton is None or tl is None:
        raise RuntimeError(
            "Triton is not available. Install triton to use this kernel. "
            f"Original import error: {_TRITON_IMPORT_ERROR}"
        )


@triton.jit
def _diffusion_step_kernel(
    x_ptr,
    y_ptr,
    alpha,
    L: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    # Map linear index -> position in last dimension
    pos = offsets % L

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Boundary: copy
    is_left = pos == 0
    is_right = pos == (L - 1)
    is_boundary = is_left | is_right

    x_prev = tl.load(x_ptr + offsets - 1, mask=mask & (pos > 0), other=0.0)
    x_next = tl.load(x_ptr + offsets + 1, mask=mask & (pos < (L - 1)), other=0.0)

    lap = x_next - 2.0 * x + x_prev
    y = tl.where(is_boundary, x, x + alpha * lap)

    tl.store(y_ptr + offsets, y, mask=mask)


def diffusion_step_triton(x: torch.Tensor, alpha: float = 0.10) -> torch.Tensor:
    """
    Args:
        x: (B, D, L) contiguous tensor on CUDA
        alpha: diffusion coefficient
    """
    _require_triton()
    if x.device.type != "cuda":
        raise ValueError("x must be on CUDA")
    if x.ndim != 3:
        raise ValueError("x must have shape (B, D, L)")
    if not x.is_contiguous():
        x = x.contiguous()

    B, D, L = x.shape
    N = B * D * L
    y = torch.empty_like(x)

    grid = (triton.cdiv(N, 1024),)
    _diffusion_step_kernel[grid](
        x,
        y,
        alpha,
        L=L,
        N=N,
        BLOCK=1024,
        num_warps=4,
    )
    return y


def diffusion_triton(x: torch.Tensor, alpha: float = 0.10, steps: int = 1) -> torch.Tensor:
    """
    Multi-step diffusion using the Triton kernel in a Python loop.

    Note: this is *not* fused yet. Fusion is a planned optimization.
    """
    y = x
    for _ in range(steps):
        y = diffusion_step_triton(y, alpha=alpha)
    return y


def _diffusion_torch(x: torch.Tensor, alpha: float = 0.10) -> torch.Tensor:
    if x.size(-1) < 3:
        return x
    lap = x[..., 2:] - 2 * x[..., 1:-1] + x[..., :-2]
    interior = x[..., 1:-1] + alpha * lap
    return torch.cat([x[..., :1], interior, x[..., -1:]], dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--D", type=int, default=256)
    parser.add_argument("--L", type=int, default=4096)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()

    device = torch.device(args.device)
    x = torch.randn(args.B, args.D, args.L, device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32)

    y_ref = x
    for _ in range(args.steps):
        y_ref = _diffusion_torch(y_ref, alpha=args.alpha)

    if device.type != "cuda":
        print("[diffusion_triton] CPU selected; Triton path is CUDA-only. Torch reference OK.")
        return

    y = diffusion_triton(x, alpha=args.alpha, steps=args.steps)
    max_err = (y.float() - y_ref.float()).abs().max().item()
    print(f"[diffusion_triton] max_abs_error={max_err:.6e}")


if __name__ == "__main__":
    main()

