"""
Triton diffusion stencil (forward) - minimal stub.

Implements one diffusion update step with discrete Laplacian along the last dim:
  y[..., i] = x[..., i] + alpha * (x[..., i+1] - 2*x[..., i] + x[..., i-1])
with boundary positions copied (i=0 and i=L-1).

This module also contains an **exact** multi-step fused variant (forward-only)
that performs `steps` diffusion updates in a *single* Triton launch using a
time-tiling stencil with halo.

Notes:
- Triton is an **optional** dependency; the main repo does not require it.
- The fused kernel focuses on reducing kernel-launch overhead and intermediate
  global-memory roundtrips (prefill-friendly).
"""

from __future__ import annotations

import argparse

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


if triton is not None and tl is not None:
    _DIFFUSION_STEP_CONFIGS = [
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=3),
    ]

    _DIFFUSION_FUSED_CONFIGS = [
        # Fused time-tiling uses more registers; prefer smaller BLOCKs.
        triton.Config({"BLOCK": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
    ]

    @triton.autotune(configs=_DIFFUSION_STEP_CONFIGS, key=["L"])
    @triton.jit
    def _diffusion_step_kernel(
        x_ptr,
        y_ptr,
        alpha,
        L,
        N,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N

        # Map linear index -> position in last dimension
        pos = offsets % L

        alpha_f = tl.full((), alpha, tl.float32)

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        # Boundary: copy
        is_left = pos == 0
        is_right = pos == (L - 1)
        is_boundary = is_left | is_right

        x_prev = tl.load(x_ptr + offsets - 1, mask=mask & (pos > 0), other=0.0).to(
            tl.float32
        )
        x_next = tl.load(
            x_ptr + offsets + 1, mask=mask & (pos < (L - 1)), other=0.0
        ).to(tl.float32)

        lap = x_next - 2.0 * x + x_prev
        y = tl.where(is_boundary, x, x + alpha_f * lap)

        tl.store(y_ptr + offsets, y, mask=mask)

    @triton.autotune(configs=_DIFFUSION_STEP_CONFIGS, key=["L"])
    @triton.jit
    def _diffusion_step_backward_kernel(
        dy_ptr,
        dx_ptr,
        alpha,
        L,
        N,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N

        # Map linear index -> position in last dimension
        pos = offsets % L

        alpha_f = tl.full((), alpha, tl.float32)
        dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        # For A^T, exclude boundary rows from contributing to adjacent dx.
        dy_prev = tl.load(
            dy_ptr + offsets - 1, mask=mask & (pos > 1), other=0.0
        ).to(tl.float32)
        dy_next = tl.load(
            dy_ptr + offsets + 1, mask=mask & (pos < (L - 2)), other=0.0
        ).to(tl.float32)

        dx = (1.0 - 2.0 * alpha_f) * dy + alpha_f * (dy_prev + dy_next)

        is_left = pos == 0
        is_right = pos == (L - 1)
        dx_left = dy + alpha_f * dy_next
        dx_right = dy + alpha_f * dy_prev
        dx = tl.where(is_left, dx_left, dx)
        dx = tl.where(is_right, dx_right, dx)

        tl.store(dx_ptr + offsets, dx, mask=mask)

    @triton.autotune(configs=_DIFFUSION_FUSED_CONFIGS, key=["L"])
    @triton.jit
    def _diffusion_fused_s2_kernel(x_ptr, y_ptr, alpha, L, BLOCK: tl.constexpr):
        pid_row = tl.program_id(axis=0)
        pid_blk = tl.program_id(axis=1)
        p = pid_blk * BLOCK + tl.arange(0, BLOCK)
        row_base = pid_row * L
        alpha_f = tl.full((), alpha, tl.float32)

        pos_u_m2 = (p - 2)
        m_u_m2 = (pos_u_m2 >= 0) & (pos_u_m2 < L)
        u_m2 = tl.load(x_ptr + row_base + pos_u_m2, mask=m_u_m2, other=0.0).to(tl.float32)
        upd_u_m2 = m_u_m2 & (~((pos_u_m2 == 0) | (pos_u_m2 == (L - 1))))

        pos_u_m1 = (p - 1)
        m_u_m1 = (pos_u_m1 >= 0) & (pos_u_m1 < L)
        u_m1 = tl.load(x_ptr + row_base + pos_u_m1, mask=m_u_m1, other=0.0).to(tl.float32)
        upd_u_m1 = m_u_m1 & (~((pos_u_m1 == 0) | (pos_u_m1 == (L - 1))))

        pos_u_0 = p
        m_u_0 = (pos_u_0 >= 0) & (pos_u_0 < L)
        u_0 = tl.load(x_ptr + row_base + pos_u_0, mask=m_u_0, other=0.0).to(tl.float32)
        upd_u_0 = m_u_0 & (~((pos_u_0 == 0) | (pos_u_0 == (L - 1))))

        pos_u_p1 = (p + 1)
        m_u_p1 = (pos_u_p1 >= 0) & (pos_u_p1 < L)
        u_p1 = tl.load(x_ptr + row_base + pos_u_p1, mask=m_u_p1, other=0.0).to(tl.float32)
        upd_u_p1 = m_u_p1 & (~((pos_u_p1 == 0) | (pos_u_p1 == (L - 1))))

        pos_u_p2 = (p + 2)
        m_u_p2 = (pos_u_p2 >= 0) & (pos_u_p2 < L)
        u_p2 = tl.load(x_ptr + row_base + pos_u_p2, mask=m_u_p2, other=0.0).to(tl.float32)
        upd_u_p2 = m_u_p2 & (~((pos_u_p2 == 0) | (pos_u_p2 == (L - 1))))

        # step 1/2
        nu_m2 = u_m2
        nu_m1 = tl.where(upd_u_m1, u_m1 + alpha_f * (u_0 - 2.0 * u_m1 + u_m2), u_m1)
        nu_0 = tl.where(upd_u_0, u_0 + alpha_f * (u_p1 - 2.0 * u_0 + u_m1), u_0)
        nu_p1 = tl.where(upd_u_p1, u_p1 + alpha_f * (u_p2 - 2.0 * u_p1 + u_0), u_p1)
        nu_p2 = u_p2
        u_m2, u_m1, u_0, u_p1, u_p2 = nu_m2, nu_m1, nu_0, nu_p1, nu_p2

        # step 2/2
        nu_m2 = u_m2
        nu_m1 = tl.where(upd_u_m1, u_m1 + alpha_f * (u_0 - 2.0 * u_m1 + u_m2), u_m1)
        nu_0 = tl.where(upd_u_0, u_0 + alpha_f * (u_p1 - 2.0 * u_0 + u_m1), u_0)
        nu_p1 = tl.where(upd_u_p1, u_p1 + alpha_f * (u_p2 - 2.0 * u_p1 + u_0), u_p1)
        nu_p2 = u_p2
        u_m2, u_m1, u_0, u_p1, u_p2 = nu_m2, nu_m1, nu_0, nu_p1, nu_p2

        tl.store(y_ptr + row_base + p, u_0, mask=(p < L))

    @triton.autotune(configs=_DIFFUSION_FUSED_CONFIGS, key=["L"])
    @triton.jit
    def _diffusion_fused_s4_kernel(x_ptr, y_ptr, alpha, L, BLOCK: tl.constexpr):
        pid_row = tl.program_id(axis=0)
        pid_blk = tl.program_id(axis=1)
        p = pid_blk * BLOCK + tl.arange(0, BLOCK)
        row_base = pid_row * L
        alpha_f = tl.full((), alpha, tl.float32)

        pos_u_m4 = (p - 4)
        m_u_m4 = (pos_u_m4 >= 0) & (pos_u_m4 < L)
        u_m4 = tl.load(x_ptr + row_base + pos_u_m4, mask=m_u_m4, other=0.0).to(tl.float32)
        upd_u_m4 = m_u_m4 & (~((pos_u_m4 == 0) | (pos_u_m4 == (L - 1))))

        pos_u_m3 = (p - 3)
        m_u_m3 = (pos_u_m3 >= 0) & (pos_u_m3 < L)
        u_m3 = tl.load(x_ptr + row_base + pos_u_m3, mask=m_u_m3, other=0.0).to(tl.float32)
        upd_u_m3 = m_u_m3 & (~((pos_u_m3 == 0) | (pos_u_m3 == (L - 1))))

        pos_u_m2 = (p - 2)
        m_u_m2 = (pos_u_m2 >= 0) & (pos_u_m2 < L)
        u_m2 = tl.load(x_ptr + row_base + pos_u_m2, mask=m_u_m2, other=0.0).to(tl.float32)
        upd_u_m2 = m_u_m2 & (~((pos_u_m2 == 0) | (pos_u_m2 == (L - 1))))

        pos_u_m1 = (p - 1)
        m_u_m1 = (pos_u_m1 >= 0) & (pos_u_m1 < L)
        u_m1 = tl.load(x_ptr + row_base + pos_u_m1, mask=m_u_m1, other=0.0).to(tl.float32)
        upd_u_m1 = m_u_m1 & (~((pos_u_m1 == 0) | (pos_u_m1 == (L - 1))))

        pos_u_0 = p
        m_u_0 = (pos_u_0 >= 0) & (pos_u_0 < L)
        u_0 = tl.load(x_ptr + row_base + pos_u_0, mask=m_u_0, other=0.0).to(tl.float32)
        upd_u_0 = m_u_0 & (~((pos_u_0 == 0) | (pos_u_0 == (L - 1))))

        pos_u_p1 = (p + 1)
        m_u_p1 = (pos_u_p1 >= 0) & (pos_u_p1 < L)
        u_p1 = tl.load(x_ptr + row_base + pos_u_p1, mask=m_u_p1, other=0.0).to(tl.float32)
        upd_u_p1 = m_u_p1 & (~((pos_u_p1 == 0) | (pos_u_p1 == (L - 1))))

        pos_u_p2 = (p + 2)
        m_u_p2 = (pos_u_p2 >= 0) & (pos_u_p2 < L)
        u_p2 = tl.load(x_ptr + row_base + pos_u_p2, mask=m_u_p2, other=0.0).to(tl.float32)
        upd_u_p2 = m_u_p2 & (~((pos_u_p2 == 0) | (pos_u_p2 == (L - 1))))

        pos_u_p3 = (p + 3)
        m_u_p3 = (pos_u_p3 >= 0) & (pos_u_p3 < L)
        u_p3 = tl.load(x_ptr + row_base + pos_u_p3, mask=m_u_p3, other=0.0).to(tl.float32)
        upd_u_p3 = m_u_p3 & (~((pos_u_p3 == 0) | (pos_u_p3 == (L - 1))))

        pos_u_p4 = (p + 4)
        m_u_p4 = (pos_u_p4 >= 0) & (pos_u_p4 < L)
        u_p4 = tl.load(x_ptr + row_base + pos_u_p4, mask=m_u_p4, other=0.0).to(tl.float32)
        upd_u_p4 = m_u_p4 & (~((pos_u_p4 == 0) | (pos_u_p4 == (L - 1))))

        # step 1/4
        nu_m4 = u_m4
        nu_m3 = tl.where(upd_u_m3, u_m3 + alpha_f * (u_m2 - 2.0 * u_m3 + u_m4), u_m3)
        nu_m2 = tl.where(upd_u_m2, u_m2 + alpha_f * (u_m1 - 2.0 * u_m2 + u_m3), u_m2)
        nu_m1 = tl.where(upd_u_m1, u_m1 + alpha_f * (u_0 - 2.0 * u_m1 + u_m2), u_m1)
        nu_0 = tl.where(upd_u_0, u_0 + alpha_f * (u_p1 - 2.0 * u_0 + u_m1), u_0)
        nu_p1 = tl.where(upd_u_p1, u_p1 + alpha_f * (u_p2 - 2.0 * u_p1 + u_0), u_p1)
        nu_p2 = tl.where(upd_u_p2, u_p2 + alpha_f * (u_p3 - 2.0 * u_p2 + u_p1), u_p2)
        nu_p3 = tl.where(upd_u_p3, u_p3 + alpha_f * (u_p4 - 2.0 * u_p3 + u_p2), u_p3)
        nu_p4 = u_p4
        u_m4, u_m3, u_m2, u_m1, u_0, u_p1, u_p2, u_p3, u_p4 = (
            nu_m4,
            nu_m3,
            nu_m2,
            nu_m1,
            nu_0,
            nu_p1,
            nu_p2,
            nu_p3,
            nu_p4,
        )

        # step 2/4
        nu_m4 = u_m4
        nu_m3 = tl.where(upd_u_m3, u_m3 + alpha_f * (u_m2 - 2.0 * u_m3 + u_m4), u_m3)
        nu_m2 = tl.where(upd_u_m2, u_m2 + alpha_f * (u_m1 - 2.0 * u_m2 + u_m3), u_m2)
        nu_m1 = tl.where(upd_u_m1, u_m1 + alpha_f * (u_0 - 2.0 * u_m1 + u_m2), u_m1)
        nu_0 = tl.where(upd_u_0, u_0 + alpha_f * (u_p1 - 2.0 * u_0 + u_m1), u_0)
        nu_p1 = tl.where(upd_u_p1, u_p1 + alpha_f * (u_p2 - 2.0 * u_p1 + u_0), u_p1)
        nu_p2 = tl.where(upd_u_p2, u_p2 + alpha_f * (u_p3 - 2.0 * u_p2 + u_p1), u_p2)
        nu_p3 = tl.where(upd_u_p3, u_p3 + alpha_f * (u_p4 - 2.0 * u_p3 + u_p2), u_p3)
        nu_p4 = u_p4
        u_m4, u_m3, u_m2, u_m1, u_0, u_p1, u_p2, u_p3, u_p4 = (
            nu_m4,
            nu_m3,
            nu_m2,
            nu_m1,
            nu_0,
            nu_p1,
            nu_p2,
            nu_p3,
            nu_p4,
        )

        # step 3/4
        nu_m4 = u_m4
        nu_m3 = tl.where(upd_u_m3, u_m3 + alpha_f * (u_m2 - 2.0 * u_m3 + u_m4), u_m3)
        nu_m2 = tl.where(upd_u_m2, u_m2 + alpha_f * (u_m1 - 2.0 * u_m2 + u_m3), u_m2)
        nu_m1 = tl.where(upd_u_m1, u_m1 + alpha_f * (u_0 - 2.0 * u_m1 + u_m2), u_m1)
        nu_0 = tl.where(upd_u_0, u_0 + alpha_f * (u_p1 - 2.0 * u_0 + u_m1), u_0)
        nu_p1 = tl.where(upd_u_p1, u_p1 + alpha_f * (u_p2 - 2.0 * u_p1 + u_0), u_p1)
        nu_p2 = tl.where(upd_u_p2, u_p2 + alpha_f * (u_p3 - 2.0 * u_p2 + u_p1), u_p2)
        nu_p3 = tl.where(upd_u_p3, u_p3 + alpha_f * (u_p4 - 2.0 * u_p3 + u_p2), u_p3)
        nu_p4 = u_p4
        u_m4, u_m3, u_m2, u_m1, u_0, u_p1, u_p2, u_p3, u_p4 = (
            nu_m4,
            nu_m3,
            nu_m2,
            nu_m1,
            nu_0,
            nu_p1,
            nu_p2,
            nu_p3,
            nu_p4,
        )

        # step 4/4
        nu_m4 = u_m4
        nu_m3 = tl.where(upd_u_m3, u_m3 + alpha_f * (u_m2 - 2.0 * u_m3 + u_m4), u_m3)
        nu_m2 = tl.where(upd_u_m2, u_m2 + alpha_f * (u_m1 - 2.0 * u_m2 + u_m3), u_m2)
        nu_m1 = tl.where(upd_u_m1, u_m1 + alpha_f * (u_0 - 2.0 * u_m1 + u_m2), u_m1)
        nu_0 = tl.where(upd_u_0, u_0 + alpha_f * (u_p1 - 2.0 * u_0 + u_m1), u_0)
        nu_p1 = tl.where(upd_u_p1, u_p1 + alpha_f * (u_p2 - 2.0 * u_p1 + u_0), u_p1)
        nu_p2 = tl.where(upd_u_p2, u_p2 + alpha_f * (u_p3 - 2.0 * u_p2 + u_p1), u_p2)
        nu_p3 = tl.where(upd_u_p3, u_p3 + alpha_f * (u_p4 - 2.0 * u_p3 + u_p2), u_p3)
        nu_p4 = u_p4
        u_m4, u_m3, u_m2, u_m1, u_0, u_p1, u_p2, u_p3, u_p4 = (
            nu_m4,
            nu_m3,
            nu_m2,
            nu_m1,
            nu_0,
            nu_p1,
            nu_p2,
            nu_p3,
            nu_p4,
        )

        tl.store(y_ptr + row_base + p, u_0, mask=(p < L))

    @triton.autotune(configs=_DIFFUSION_FUSED_CONFIGS, key=["L"])
    @triton.jit
    def _diffusion_fused_s8_kernel(x_ptr, y_ptr, alpha, L, BLOCK: tl.constexpr):
        pid_row = tl.program_id(axis=0)
        pid_blk = tl.program_id(axis=1)
        p = pid_blk * BLOCK + tl.arange(0, BLOCK)
        row_base = pid_row * L
        alpha_f = tl.full((), alpha, tl.float32)

        pos_u_m8 = (p - 8)
        m_u_m8 = (pos_u_m8 >= 0) & (pos_u_m8 < L)
        u_m8 = tl.load(x_ptr + row_base + pos_u_m8, mask=m_u_m8, other=0.0).to(tl.float32)
        upd_u_m8 = m_u_m8 & (~((pos_u_m8 == 0) | (pos_u_m8 == (L - 1))))

        pos_u_m7 = (p - 7)
        m_u_m7 = (pos_u_m7 >= 0) & (pos_u_m7 < L)
        u_m7 = tl.load(x_ptr + row_base + pos_u_m7, mask=m_u_m7, other=0.0).to(tl.float32)
        upd_u_m7 = m_u_m7 & (~((pos_u_m7 == 0) | (pos_u_m7 == (L - 1))))

        pos_u_m6 = (p - 6)
        m_u_m6 = (pos_u_m6 >= 0) & (pos_u_m6 < L)
        u_m6 = tl.load(x_ptr + row_base + pos_u_m6, mask=m_u_m6, other=0.0).to(tl.float32)
        upd_u_m6 = m_u_m6 & (~((pos_u_m6 == 0) | (pos_u_m6 == (L - 1))))

        pos_u_m5 = (p - 5)
        m_u_m5 = (pos_u_m5 >= 0) & (pos_u_m5 < L)
        u_m5 = tl.load(x_ptr + row_base + pos_u_m5, mask=m_u_m5, other=0.0).to(tl.float32)
        upd_u_m5 = m_u_m5 & (~((pos_u_m5 == 0) | (pos_u_m5 == (L - 1))))

        pos_u_m4 = (p - 4)
        m_u_m4 = (pos_u_m4 >= 0) & (pos_u_m4 < L)
        u_m4 = tl.load(x_ptr + row_base + pos_u_m4, mask=m_u_m4, other=0.0).to(tl.float32)
        upd_u_m4 = m_u_m4 & (~((pos_u_m4 == 0) | (pos_u_m4 == (L - 1))))

        pos_u_m3 = (p - 3)
        m_u_m3 = (pos_u_m3 >= 0) & (pos_u_m3 < L)
        u_m3 = tl.load(x_ptr + row_base + pos_u_m3, mask=m_u_m3, other=0.0).to(tl.float32)
        upd_u_m3 = m_u_m3 & (~((pos_u_m3 == 0) | (pos_u_m3 == (L - 1))))

        pos_u_m2 = (p - 2)
        m_u_m2 = (pos_u_m2 >= 0) & (pos_u_m2 < L)
        u_m2 = tl.load(x_ptr + row_base + pos_u_m2, mask=m_u_m2, other=0.0).to(tl.float32)
        upd_u_m2 = m_u_m2 & (~((pos_u_m2 == 0) | (pos_u_m2 == (L - 1))))

        pos_u_m1 = (p - 1)
        m_u_m1 = (pos_u_m1 >= 0) & (pos_u_m1 < L)
        u_m1 = tl.load(x_ptr + row_base + pos_u_m1, mask=m_u_m1, other=0.0).to(tl.float32)
        upd_u_m1 = m_u_m1 & (~((pos_u_m1 == 0) | (pos_u_m1 == (L - 1))))

        pos_u_0 = p
        m_u_0 = (pos_u_0 >= 0) & (pos_u_0 < L)
        u_0 = tl.load(x_ptr + row_base + pos_u_0, mask=m_u_0, other=0.0).to(tl.float32)
        upd_u_0 = m_u_0 & (~((pos_u_0 == 0) | (pos_u_0 == (L - 1))))

        pos_u_p1 = (p + 1)
        m_u_p1 = (pos_u_p1 >= 0) & (pos_u_p1 < L)
        u_p1 = tl.load(x_ptr + row_base + pos_u_p1, mask=m_u_p1, other=0.0).to(tl.float32)
        upd_u_p1 = m_u_p1 & (~((pos_u_p1 == 0) | (pos_u_p1 == (L - 1))))

        pos_u_p2 = (p + 2)
        m_u_p2 = (pos_u_p2 >= 0) & (pos_u_p2 < L)
        u_p2 = tl.load(x_ptr + row_base + pos_u_p2, mask=m_u_p2, other=0.0).to(tl.float32)
        upd_u_p2 = m_u_p2 & (~((pos_u_p2 == 0) | (pos_u_p2 == (L - 1))))

        pos_u_p3 = (p + 3)
        m_u_p3 = (pos_u_p3 >= 0) & (pos_u_p3 < L)
        u_p3 = tl.load(x_ptr + row_base + pos_u_p3, mask=m_u_p3, other=0.0).to(tl.float32)
        upd_u_p3 = m_u_p3 & (~((pos_u_p3 == 0) | (pos_u_p3 == (L - 1))))

        pos_u_p4 = (p + 4)
        m_u_p4 = (pos_u_p4 >= 0) & (pos_u_p4 < L)
        u_p4 = tl.load(x_ptr + row_base + pos_u_p4, mask=m_u_p4, other=0.0).to(tl.float32)
        upd_u_p4 = m_u_p4 & (~((pos_u_p4 == 0) | (pos_u_p4 == (L - 1))))

        pos_u_p5 = (p + 5)
        m_u_p5 = (pos_u_p5 >= 0) & (pos_u_p5 < L)
        u_p5 = tl.load(x_ptr + row_base + pos_u_p5, mask=m_u_p5, other=0.0).to(tl.float32)
        upd_u_p5 = m_u_p5 & (~((pos_u_p5 == 0) | (pos_u_p5 == (L - 1))))

        pos_u_p6 = (p + 6)
        m_u_p6 = (pos_u_p6 >= 0) & (pos_u_p6 < L)
        u_p6 = tl.load(x_ptr + row_base + pos_u_p6, mask=m_u_p6, other=0.0).to(tl.float32)
        upd_u_p6 = m_u_p6 & (~((pos_u_p6 == 0) | (pos_u_p6 == (L - 1))))

        pos_u_p7 = (p + 7)
        m_u_p7 = (pos_u_p7 >= 0) & (pos_u_p7 < L)
        u_p7 = tl.load(x_ptr + row_base + pos_u_p7, mask=m_u_p7, other=0.0).to(tl.float32)
        upd_u_p7 = m_u_p7 & (~((pos_u_p7 == 0) | (pos_u_p7 == (L - 1))))

        pos_u_p8 = (p + 8)
        m_u_p8 = (pos_u_p8 >= 0) & (pos_u_p8 < L)
        u_p8 = tl.load(x_ptr + row_base + pos_u_p8, mask=m_u_p8, other=0.0).to(tl.float32)
        upd_u_p8 = m_u_p8 & (~((pos_u_p8 == 0) | (pos_u_p8 == (L - 1))))

        # step 1/8
        nu_m8 = u_m8
        nu_m7 = tl.where(upd_u_m7, u_m7 + alpha_f * (u_m6 - 2.0 * u_m7 + u_m8), u_m7)
        nu_m6 = tl.where(upd_u_m6, u_m6 + alpha_f * (u_m5 - 2.0 * u_m6 + u_m7), u_m6)
        nu_m5 = tl.where(upd_u_m5, u_m5 + alpha_f * (u_m4 - 2.0 * u_m5 + u_m6), u_m5)
        nu_m4 = tl.where(upd_u_m4, u_m4 + alpha_f * (u_m3 - 2.0 * u_m4 + u_m5), u_m4)
        nu_m3 = tl.where(upd_u_m3, u_m3 + alpha_f * (u_m2 - 2.0 * u_m3 + u_m4), u_m3)
        nu_m2 = tl.where(upd_u_m2, u_m2 + alpha_f * (u_m1 - 2.0 * u_m2 + u_m3), u_m2)
        nu_m1 = tl.where(upd_u_m1, u_m1 + alpha_f * (u_0 - 2.0 * u_m1 + u_m2), u_m1)
        nu_0 = tl.where(upd_u_0, u_0 + alpha_f * (u_p1 - 2.0 * u_0 + u_m1), u_0)
        nu_p1 = tl.where(upd_u_p1, u_p1 + alpha_f * (u_p2 - 2.0 * u_p1 + u_0), u_p1)
        nu_p2 = tl.where(upd_u_p2, u_p2 + alpha_f * (u_p3 - 2.0 * u_p2 + u_p1), u_p2)
        nu_p3 = tl.where(upd_u_p3, u_p3 + alpha_f * (u_p4 - 2.0 * u_p3 + u_p2), u_p3)
        nu_p4 = tl.where(upd_u_p4, u_p4 + alpha_f * (u_p5 - 2.0 * u_p4 + u_p3), u_p4)
        nu_p5 = tl.where(upd_u_p5, u_p5 + alpha_f * (u_p6 - 2.0 * u_p5 + u_p4), u_p5)
        nu_p6 = tl.where(upd_u_p6, u_p6 + alpha_f * (u_p7 - 2.0 * u_p6 + u_p5), u_p6)
        nu_p7 = tl.where(upd_u_p7, u_p7 + alpha_f * (u_p8 - 2.0 * u_p7 + u_p6), u_p7)
        nu_p8 = u_p8
        u_m8, u_m7, u_m6, u_m5, u_m4, u_m3, u_m2, u_m1, u_0, u_p1, u_p2, u_p3, u_p4, u_p5, u_p6, u_p7, u_p8 = (
            nu_m8,
            nu_m7,
            nu_m6,
            nu_m5,
            nu_m4,
            nu_m3,
            nu_m2,
            nu_m1,
            nu_0,
            nu_p1,
            nu_p2,
            nu_p3,
            nu_p4,
            nu_p5,
            nu_p6,
            nu_p7,
            nu_p8,
        )

        # step 2/8
        nu_m8 = u_m8
        nu_m7 = tl.where(upd_u_m7, u_m7 + alpha_f * (u_m6 - 2.0 * u_m7 + u_m8), u_m7)
        nu_m6 = tl.where(upd_u_m6, u_m6 + alpha_f * (u_m5 - 2.0 * u_m6 + u_m7), u_m6)
        nu_m5 = tl.where(upd_u_m5, u_m5 + alpha_f * (u_m4 - 2.0 * u_m5 + u_m6), u_m5)
        nu_m4 = tl.where(upd_u_m4, u_m4 + alpha_f * (u_m3 - 2.0 * u_m4 + u_m5), u_m4)
        nu_m3 = tl.where(upd_u_m3, u_m3 + alpha_f * (u_m2 - 2.0 * u_m3 + u_m4), u_m3)
        nu_m2 = tl.where(upd_u_m2, u_m2 + alpha_f * (u_m1 - 2.0 * u_m2 + u_m3), u_m2)
        nu_m1 = tl.where(upd_u_m1, u_m1 + alpha_f * (u_0 - 2.0 * u_m1 + u_m2), u_m1)
        nu_0 = tl.where(upd_u_0, u_0 + alpha_f * (u_p1 - 2.0 * u_0 + u_m1), u_0)
        nu_p1 = tl.where(upd_u_p1, u_p1 + alpha_f * (u_p2 - 2.0 * u_p1 + u_0), u_p1)
        nu_p2 = tl.where(upd_u_p2, u_p2 + alpha_f * (u_p3 - 2.0 * u_p2 + u_p1), u_p2)
        nu_p3 = tl.where(upd_u_p3, u_p3 + alpha_f * (u_p4 - 2.0 * u_p3 + u_p2), u_p3)
        nu_p4 = tl.where(upd_u_p4, u_p4 + alpha_f * (u_p5 - 2.0 * u_p4 + u_p3), u_p4)
        nu_p5 = tl.where(upd_u_p5, u_p5 + alpha_f * (u_p6 - 2.0 * u_p5 + u_p4), u_p5)
        nu_p6 = tl.where(upd_u_p6, u_p6 + alpha_f * (u_p7 - 2.0 * u_p6 + u_p5), u_p6)
        nu_p7 = tl.where(upd_u_p7, u_p7 + alpha_f * (u_p8 - 2.0 * u_p7 + u_p6), u_p7)
        nu_p8 = u_p8
        u_m8, u_m7, u_m6, u_m5, u_m4, u_m3, u_m2, u_m1, u_0, u_p1, u_p2, u_p3, u_p4, u_p5, u_p6, u_p7, u_p8 = (
            nu_m8,
            nu_m7,
            nu_m6,
            nu_m5,
            nu_m4,
            nu_m3,
            nu_m2,
            nu_m1,
            nu_0,
            nu_p1,
            nu_p2,
            nu_p3,
            nu_p4,
            nu_p5,
            nu_p6,
            nu_p7,
            nu_p8,
        )

        # step 3/8
        nu_m8 = u_m8
        nu_m7 = tl.where(upd_u_m7, u_m7 + alpha_f * (u_m6 - 2.0 * u_m7 + u_m8), u_m7)
        nu_m6 = tl.where(upd_u_m6, u_m6 + alpha_f * (u_m5 - 2.0 * u_m6 + u_m7), u_m6)
        nu_m5 = tl.where(upd_u_m5, u_m5 + alpha_f * (u_m4 - 2.0 * u_m5 + u_m6), u_m5)
        nu_m4 = tl.where(upd_u_m4, u_m4 + alpha_f * (u_m3 - 2.0 * u_m4 + u_m5), u_m4)
        nu_m3 = tl.where(upd_u_m3, u_m3 + alpha_f * (u_m2 - 2.0 * u_m3 + u_m4), u_m3)
        nu_m2 = tl.where(upd_u_m2, u_m2 + alpha_f * (u_m1 - 2.0 * u_m2 + u_m3), u_m2)
        nu_m1 = tl.where(upd_u_m1, u_m1 + alpha_f * (u_0 - 2.0 * u_m1 + u_m2), u_m1)
        nu_0 = tl.where(upd_u_0, u_0 + alpha_f * (u_p1 - 2.0 * u_0 + u_m1), u_0)
        nu_p1 = tl.where(upd_u_p1, u_p1 + alpha_f * (u_p2 - 2.0 * u_p1 + u_0), u_p1)
        nu_p2 = tl.where(upd_u_p2, u_p2 + alpha_f * (u_p3 - 2.0 * u_p2 + u_p1), u_p2)
        nu_p3 = tl.where(upd_u_p3, u_p3 + alpha_f * (u_p4 - 2.0 * u_p3 + u_p2), u_p3)
        nu_p4 = tl.where(upd_u_p4, u_p4 + alpha_f * (u_p5 - 2.0 * u_p4 + u_p3), u_p4)
        nu_p5 = tl.where(upd_u_p5, u_p5 + alpha_f * (u_p6 - 2.0 * u_p5 + u_p4), u_p5)
        nu_p6 = tl.where(upd_u_p6, u_p6 + alpha_f * (u_p7 - 2.0 * u_p6 + u_p5), u_p6)
        nu_p7 = tl.where(upd_u_p7, u_p7 + alpha_f * (u_p8 - 2.0 * u_p7 + u_p6), u_p7)
        nu_p8 = u_p8
        u_m8, u_m7, u_m6, u_m5, u_m4, u_m3, u_m2, u_m1, u_0, u_p1, u_p2, u_p3, u_p4, u_p5, u_p6, u_p7, u_p8 = (
            nu_m8,
            nu_m7,
            nu_m6,
            nu_m5,
            nu_m4,
            nu_m3,
            nu_m2,
            nu_m1,
            nu_0,
            nu_p1,
            nu_p2,
            nu_p3,
            nu_p4,
            nu_p5,
            nu_p6,
            nu_p7,
            nu_p8,
        )

        # step 4/8
        nu_m8 = u_m8
        nu_m7 = tl.where(upd_u_m7, u_m7 + alpha_f * (u_m6 - 2.0 * u_m7 + u_m8), u_m7)
        nu_m6 = tl.where(upd_u_m6, u_m6 + alpha_f * (u_m5 - 2.0 * u_m6 + u_m7), u_m6)
        nu_m5 = tl.where(upd_u_m5, u_m5 + alpha_f * (u_m4 - 2.0 * u_m5 + u_m6), u_m5)
        nu_m4 = tl.where(upd_u_m4, u_m4 + alpha_f * (u_m3 - 2.0 * u_m4 + u_m5), u_m4)
        nu_m3 = tl.where(upd_u_m3, u_m3 + alpha_f * (u_m2 - 2.0 * u_m3 + u_m4), u_m3)
        nu_m2 = tl.where(upd_u_m2, u_m2 + alpha_f * (u_m1 - 2.0 * u_m2 + u_m3), u_m2)
        nu_m1 = tl.where(upd_u_m1, u_m1 + alpha_f * (u_0 - 2.0 * u_m1 + u_m2), u_m1)
        nu_0 = tl.where(upd_u_0, u_0 + alpha_f * (u_p1 - 2.0 * u_0 + u_m1), u_0)
        nu_p1 = tl.where(upd_u_p1, u_p1 + alpha_f * (u_p2 - 2.0 * u_p1 + u_0), u_p1)
        nu_p2 = tl.where(upd_u_p2, u_p2 + alpha_f * (u_p3 - 2.0 * u_p2 + u_p1), u_p2)
        nu_p3 = tl.where(upd_u_p3, u_p3 + alpha_f * (u_p4 - 2.0 * u_p3 + u_p2), u_p3)
        nu_p4 = tl.where(upd_u_p4, u_p4 + alpha_f * (u_p5 - 2.0 * u_p4 + u_p3), u_p4)
        nu_p5 = tl.where(upd_u_p5, u_p5 + alpha_f * (u_p6 - 2.0 * u_p5 + u_p4), u_p5)
        nu_p6 = tl.where(upd_u_p6, u_p6 + alpha_f * (u_p7 - 2.0 * u_p6 + u_p5), u_p6)
        nu_p7 = tl.where(upd_u_p7, u_p7 + alpha_f * (u_p8 - 2.0 * u_p7 + u_p6), u_p7)
        nu_p8 = u_p8
        u_m8, u_m7, u_m6, u_m5, u_m4, u_m3, u_m2, u_m1, u_0, u_p1, u_p2, u_p3, u_p4, u_p5, u_p6, u_p7, u_p8 = (
            nu_m8,
            nu_m7,
            nu_m6,
            nu_m5,
            nu_m4,
            nu_m3,
            nu_m2,
            nu_m1,
            nu_0,
            nu_p1,
            nu_p2,
            nu_p3,
            nu_p4,
            nu_p5,
            nu_p6,
            nu_p7,
            nu_p8,
        )

        # step 5/8
        nu_m8 = u_m8
        nu_m7 = tl.where(upd_u_m7, u_m7 + alpha_f * (u_m6 - 2.0 * u_m7 + u_m8), u_m7)
        nu_m6 = tl.where(upd_u_m6, u_m6 + alpha_f * (u_m5 - 2.0 * u_m6 + u_m7), u_m6)
        nu_m5 = tl.where(upd_u_m5, u_m5 + alpha_f * (u_m4 - 2.0 * u_m5 + u_m6), u_m5)
        nu_m4 = tl.where(upd_u_m4, u_m4 + alpha_f * (u_m3 - 2.0 * u_m4 + u_m5), u_m4)
        nu_m3 = tl.where(upd_u_m3, u_m3 + alpha_f * (u_m2 - 2.0 * u_m3 + u_m4), u_m3)
        nu_m2 = tl.where(upd_u_m2, u_m2 + alpha_f * (u_m1 - 2.0 * u_m2 + u_m3), u_m2)
        nu_m1 = tl.where(upd_u_m1, u_m1 + alpha_f * (u_0 - 2.0 * u_m1 + u_m2), u_m1)
        nu_0 = tl.where(upd_u_0, u_0 + alpha_f * (u_p1 - 2.0 * u_0 + u_m1), u_0)
        nu_p1 = tl.where(upd_u_p1, u_p1 + alpha_f * (u_p2 - 2.0 * u_p1 + u_0), u_p1)
        nu_p2 = tl.where(upd_u_p2, u_p2 + alpha_f * (u_p3 - 2.0 * u_p2 + u_p1), u_p2)
        nu_p3 = tl.where(upd_u_p3, u_p3 + alpha_f * (u_p4 - 2.0 * u_p3 + u_p2), u_p3)
        nu_p4 = tl.where(upd_u_p4, u_p4 + alpha_f * (u_p5 - 2.0 * u_p4 + u_p3), u_p4)
        nu_p5 = tl.where(upd_u_p5, u_p5 + alpha_f * (u_p6 - 2.0 * u_p5 + u_p4), u_p5)
        nu_p6 = tl.where(upd_u_p6, u_p6 + alpha_f * (u_p7 - 2.0 * u_p6 + u_p5), u_p6)
        nu_p7 = tl.where(upd_u_p7, u_p7 + alpha_f * (u_p8 - 2.0 * u_p7 + u_p6), u_p7)
        nu_p8 = u_p8
        u_m8, u_m7, u_m6, u_m5, u_m4, u_m3, u_m2, u_m1, u_0, u_p1, u_p2, u_p3, u_p4, u_p5, u_p6, u_p7, u_p8 = (
            nu_m8,
            nu_m7,
            nu_m6,
            nu_m5,
            nu_m4,
            nu_m3,
            nu_m2,
            nu_m1,
            nu_0,
            nu_p1,
            nu_p2,
            nu_p3,
            nu_p4,
            nu_p5,
            nu_p6,
            nu_p7,
            nu_p8,
        )

        # step 6/8
        nu_m8 = u_m8
        nu_m7 = tl.where(upd_u_m7, u_m7 + alpha_f * (u_m6 - 2.0 * u_m7 + u_m8), u_m7)
        nu_m6 = tl.where(upd_u_m6, u_m6 + alpha_f * (u_m5 - 2.0 * u_m6 + u_m7), u_m6)
        nu_m5 = tl.where(upd_u_m5, u_m5 + alpha_f * (u_m4 - 2.0 * u_m5 + u_m6), u_m5)
        nu_m4 = tl.where(upd_u_m4, u_m4 + alpha_f * (u_m3 - 2.0 * u_m4 + u_m5), u_m4)
        nu_m3 = tl.where(upd_u_m3, u_m3 + alpha_f * (u_m2 - 2.0 * u_m3 + u_m4), u_m3)
        nu_m2 = tl.where(upd_u_m2, u_m2 + alpha_f * (u_m1 - 2.0 * u_m2 + u_m3), u_m2)
        nu_m1 = tl.where(upd_u_m1, u_m1 + alpha_f * (u_0 - 2.0 * u_m1 + u_m2), u_m1)
        nu_0 = tl.where(upd_u_0, u_0 + alpha_f * (u_p1 - 2.0 * u_0 + u_m1), u_0)
        nu_p1 = tl.where(upd_u_p1, u_p1 + alpha_f * (u_p2 - 2.0 * u_p1 + u_0), u_p1)
        nu_p2 = tl.where(upd_u_p2, u_p2 + alpha_f * (u_p3 - 2.0 * u_p2 + u_p1), u_p2)
        nu_p3 = tl.where(upd_u_p3, u_p3 + alpha_f * (u_p4 - 2.0 * u_p3 + u_p2), u_p3)
        nu_p4 = tl.where(upd_u_p4, u_p4 + alpha_f * (u_p5 - 2.0 * u_p4 + u_p3), u_p4)
        nu_p5 = tl.where(upd_u_p5, u_p5 + alpha_f * (u_p6 - 2.0 * u_p5 + u_p4), u_p5)
        nu_p6 = tl.where(upd_u_p6, u_p6 + alpha_f * (u_p7 - 2.0 * u_p6 + u_p5), u_p6)
        nu_p7 = tl.where(upd_u_p7, u_p7 + alpha_f * (u_p8 - 2.0 * u_p7 + u_p6), u_p7)
        nu_p8 = u_p8
        u_m8, u_m7, u_m6, u_m5, u_m4, u_m3, u_m2, u_m1, u_0, u_p1, u_p2, u_p3, u_p4, u_p5, u_p6, u_p7, u_p8 = (
            nu_m8,
            nu_m7,
            nu_m6,
            nu_m5,
            nu_m4,
            nu_m3,
            nu_m2,
            nu_m1,
            nu_0,
            nu_p1,
            nu_p2,
            nu_p3,
            nu_p4,
            nu_p5,
            nu_p6,
            nu_p7,
            nu_p8,
        )

        # step 7/8
        nu_m8 = u_m8
        nu_m7 = tl.where(upd_u_m7, u_m7 + alpha_f * (u_m6 - 2.0 * u_m7 + u_m8), u_m7)
        nu_m6 = tl.where(upd_u_m6, u_m6 + alpha_f * (u_m5 - 2.0 * u_m6 + u_m7), u_m6)
        nu_m5 = tl.where(upd_u_m5, u_m5 + alpha_f * (u_m4 - 2.0 * u_m5 + u_m6), u_m5)
        nu_m4 = tl.where(upd_u_m4, u_m4 + alpha_f * (u_m3 - 2.0 * u_m4 + u_m5), u_m4)
        nu_m3 = tl.where(upd_u_m3, u_m3 + alpha_f * (u_m2 - 2.0 * u_m3 + u_m4), u_m3)
        nu_m2 = tl.where(upd_u_m2, u_m2 + alpha_f * (u_m1 - 2.0 * u_m2 + u_m3), u_m2)
        nu_m1 = tl.where(upd_u_m1, u_m1 + alpha_f * (u_0 - 2.0 * u_m1 + u_m2), u_m1)
        nu_0 = tl.where(upd_u_0, u_0 + alpha_f * (u_p1 - 2.0 * u_0 + u_m1), u_0)
        nu_p1 = tl.where(upd_u_p1, u_p1 + alpha_f * (u_p2 - 2.0 * u_p1 + u_0), u_p1)
        nu_p2 = tl.where(upd_u_p2, u_p2 + alpha_f * (u_p3 - 2.0 * u_p2 + u_p1), u_p2)
        nu_p3 = tl.where(upd_u_p3, u_p3 + alpha_f * (u_p4 - 2.0 * u_p3 + u_p2), u_p3)
        nu_p4 = tl.where(upd_u_p4, u_p4 + alpha_f * (u_p5 - 2.0 * u_p4 + u_p3), u_p4)
        nu_p5 = tl.where(upd_u_p5, u_p5 + alpha_f * (u_p6 - 2.0 * u_p5 + u_p4), u_p5)
        nu_p6 = tl.where(upd_u_p6, u_p6 + alpha_f * (u_p7 - 2.0 * u_p6 + u_p5), u_p6)
        nu_p7 = tl.where(upd_u_p7, u_p7 + alpha_f * (u_p8 - 2.0 * u_p7 + u_p6), u_p7)
        nu_p8 = u_p8
        u_m8, u_m7, u_m6, u_m5, u_m4, u_m3, u_m2, u_m1, u_0, u_p1, u_p2, u_p3, u_p4, u_p5, u_p6, u_p7, u_p8 = (
            nu_m8,
            nu_m7,
            nu_m6,
            nu_m5,
            nu_m4,
            nu_m3,
            nu_m2,
            nu_m1,
            nu_0,
            nu_p1,
            nu_p2,
            nu_p3,
            nu_p4,
            nu_p5,
            nu_p6,
            nu_p7,
            nu_p8,
        )

        # step 8/8
        nu_m8 = u_m8
        nu_m7 = tl.where(upd_u_m7, u_m7 + alpha_f * (u_m6 - 2.0 * u_m7 + u_m8), u_m7)
        nu_m6 = tl.where(upd_u_m6, u_m6 + alpha_f * (u_m5 - 2.0 * u_m6 + u_m7), u_m6)
        nu_m5 = tl.where(upd_u_m5, u_m5 + alpha_f * (u_m4 - 2.0 * u_m5 + u_m6), u_m5)
        nu_m4 = tl.where(upd_u_m4, u_m4 + alpha_f * (u_m3 - 2.0 * u_m4 + u_m5), u_m4)
        nu_m3 = tl.where(upd_u_m3, u_m3 + alpha_f * (u_m2 - 2.0 * u_m3 + u_m4), u_m3)
        nu_m2 = tl.where(upd_u_m2, u_m2 + alpha_f * (u_m1 - 2.0 * u_m2 + u_m3), u_m2)
        nu_m1 = tl.where(upd_u_m1, u_m1 + alpha_f * (u_0 - 2.0 * u_m1 + u_m2), u_m1)
        nu_0 = tl.where(upd_u_0, u_0 + alpha_f * (u_p1 - 2.0 * u_0 + u_m1), u_0)
        nu_p1 = tl.where(upd_u_p1, u_p1 + alpha_f * (u_p2 - 2.0 * u_p1 + u_0), u_p1)
        nu_p2 = tl.where(upd_u_p2, u_p2 + alpha_f * (u_p3 - 2.0 * u_p2 + u_p1), u_p2)
        nu_p3 = tl.where(upd_u_p3, u_p3 + alpha_f * (u_p4 - 2.0 * u_p3 + u_p2), u_p3)
        nu_p4 = tl.where(upd_u_p4, u_p4 + alpha_f * (u_p5 - 2.0 * u_p4 + u_p3), u_p4)
        nu_p5 = tl.where(upd_u_p5, u_p5 + alpha_f * (u_p6 - 2.0 * u_p5 + u_p4), u_p5)
        nu_p6 = tl.where(upd_u_p6, u_p6 + alpha_f * (u_p7 - 2.0 * u_p6 + u_p5), u_p6)
        nu_p7 = tl.where(upd_u_p7, u_p7 + alpha_f * (u_p8 - 2.0 * u_p7 + u_p6), u_p7)
        nu_p8 = u_p8
        u_m8, u_m7, u_m6, u_m5, u_m4, u_m3, u_m2, u_m1, u_0, u_p1, u_p2, u_p3, u_p4, u_p5, u_p6, u_p7, u_p8 = (
            nu_m8,
            nu_m7,
            nu_m6,
            nu_m5,
            nu_m4,
            nu_m3,
            nu_m2,
            nu_m1,
            nu_0,
            nu_p1,
            nu_p2,
            nu_p3,
            nu_p4,
            nu_p5,
            nu_p6,
            nu_p7,
            nu_p8,
        )

        tl.store(y_ptr + row_base + p, u_0, mask=(p < L))


def _diffusion_step_triton_forward(
    x: torch.Tensor, alpha: float = 0.10
) -> torch.Tensor:
    """Run a single diffusion step (Triton forward).

    Args:
        x: (B, D, L) contiguous tensor on CUDA.
        alpha: Diffusion coefficient.

    Returns:
        Tensor of shape (B, D, L).
    """
    _require_triton()
    if x.device.type != "cuda":
        raise ValueError("x must be on CUDA")
    if x.ndim != 3:
        raise ValueError("x must have shape (B, D, L)")
    if not x.is_contiguous():
        x = x.contiguous()

    b, d, l = x.shape
    if l < 3:
        return x.clone()
    n = b * d * l
    y = torch.empty_like(x)

    x_1d = x.view(-1)
    y_1d = y.view(-1)
    grid = lambda META: (triton.cdiv(n, META["BLOCK"]),)
    _diffusion_step_kernel[grid](x_1d, y_1d, float(alpha), l, n)
    return y


def diffusion_step_triton_backward(
    grad_out: torch.Tensor, alpha: float = 0.10
) -> torch.Tensor:
    """Backward pass for the diffusion step (Triton).

    Args:
        grad_out: Gradient of output with shape (B, D, L).
        alpha: Diffusion coefficient.

    Returns:
        Gradient of input with shape (B, D, L).
    """
    _require_triton()
    if grad_out.device.type != "cuda":
        raise ValueError("grad_out must be on CUDA")
    if grad_out.ndim != 3:
        raise ValueError("grad_out must have shape (B, D, L)")
    b, d, l = grad_out.shape
    if l < 3:
        return grad_out
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    n = b * d * l
    grad_in = torch.empty_like(grad_out)
    dy_1d = grad_out.view(-1)
    dx_1d = grad_in.view(-1)
    grid = lambda META: (triton.cdiv(n, META["BLOCK"]),)
    _diffusion_step_backward_kernel[grid](dy_1d, dx_1d, float(alpha), l, n)
    return grad_in


class _DiffusionStepTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = float(alpha)
        ctx.l = x.shape[-1]
        return _diffusion_step_triton_forward(x, alpha=alpha)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor, None]:
        if ctx.l < 3:
            return grad_out, None
        grad_in = diffusion_step_triton_backward(grad_out, alpha=ctx.alpha)
        return grad_in, None


def diffusion_step_triton(x: torch.Tensor, alpha: float = 0.10) -> torch.Tensor:
    """Autograd-enabled diffusion step (Triton)."""
    return _DiffusionStepTriton.apply(x, float(alpha))


def diffusion_triton(x: torch.Tensor, alpha: float = 0.10, steps: int = 1) -> torch.Tensor:
    """
    Multi-step diffusion using the Triton kernel in a Python loop.

    Note: this is *not* fused yet. Fusion is a planned optimization.
    """
    y = x
    for _ in range(steps):
        y = diffusion_step_triton(y, alpha=alpha)
    return y


def _diffusion_fused_triton_forward(
    x: torch.Tensor, alpha: float = 0.10, steps: int = 1
) -> torch.Tensor:
    """Multi-step diffusion using a single fused Triton kernel launch."""
    _require_triton()
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}")
    if steps == 0:
        return x
    if x.device.type != "cuda":
        raise ValueError("x must be on CUDA")
    if x.ndim != 3:
        raise ValueError("x must have shape (B, D, L)")
    if not x.is_contiguous():
        x = x.contiguous()

    b, d, l = x.shape
    if l < 3:
        return x

    if steps == 1:
        return _diffusion_step_triton_forward(x, alpha=alpha)

    n_rows = b * d
    y = torch.empty_like(x)

    x_1d = x.view(-1)
    y_1d = y.view(-1)

    # 2D grid: rows x blocks_along_L
    grid = lambda META: (n_rows, triton.cdiv(l, META["BLOCK"]))
    if steps == 2:
        _diffusion_fused_s2_kernel[grid](x_1d, y_1d, float(alpha), l)
    elif steps == 4:
        _diffusion_fused_s4_kernel[grid](x_1d, y_1d, float(alpha), l)
    elif steps == 8:
        _diffusion_fused_s8_kernel[grid](x_1d, y_1d, float(alpha), l)
    else:
        raise ValueError("fused kernel only supports steps in {1, 2, 4, 8}")
    return y


class _DiffusionFusedTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float, steps: int) -> torch.Tensor:
        ctx.alpha = float(alpha)
        ctx.steps = int(steps)
        ctx.l = x.shape[-1]
        return _diffusion_fused_triton_forward(x, alpha=alpha, steps=steps)

    @staticmethod
    def backward(
        ctx, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        if ctx.steps == 0 or ctx.l < 3:
            return grad_out, None, None
        grad_in = grad_out
        for _ in range(ctx.steps):
            grad_in = diffusion_step_triton_backward(grad_in, alpha=ctx.alpha)
        return grad_in, None, None


def diffusion_triton_fused(
    x: torch.Tensor, alpha: float = 0.10, steps: int = 1
) -> torch.Tensor:
    """Multi-step diffusion using a single fused Triton kernel launch.

    Requirements:
    - x is CUDA tensor of shape (B, D, L)
    - contiguous (row-major) so each [B,D] row has stride(L)=1.

    Notes:
    - Fused specializations are provided for `steps in {2, 4, 8}` (the most
      relevant serving settings). For other `steps`, use `diffusion_triton`
      (Python loop) or extend the fused kernels.
    - Backward uses repeated K3 steps (A^T) and is not fused.
    """
    if steps == 0:
        return x
    if x.requires_grad:
        return _DiffusionFusedTriton.apply(x, float(alpha), int(steps))
    return _diffusion_fused_triton_forward(x, alpha=alpha, steps=steps)


def _diffusion_torch(x: torch.Tensor, alpha: float = 0.10) -> torch.Tensor:
    if x.size(-1) < 3:
        return x
    x_fp32 = x.to(dtype=torch.float32)
    lap = x_fp32[..., 2:] - 2.0 * x_fp32[..., 1:-1] + x_fp32[..., :-2]
    interior = x_fp32[..., 1:-1] + float(alpha) * lap
    y = torch.cat([x_fp32[..., :1], interior, x_fp32[..., -1:]], dim=-1)
    return y.to(dtype=x.dtype)


def _diffusion_torch_fp32_step(x_fp32: torch.Tensor, alpha: float = 0.10) -> torch.Tensor:
    """Single diffusion step that keeps fp32 (no cast-back)."""
    if x_fp32.size(-1) < 3:
        return x_fp32
    if x_fp32.dtype != torch.float32:
        raise ValueError("expected fp32 input")
    lap = x_fp32[..., 2:] - 2.0 * x_fp32[..., 1:-1] + x_fp32[..., :-2]
    interior = x_fp32[..., 1:-1] + float(alpha) * lap
    return torch.cat([x_fp32[..., :1], interior, x_fp32[..., -1:]], dim=-1)


@torch.no_grad()
def _bench_cuda(
    fn,
    *,
    iters: int,
    warmup: int,
) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms: list[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(float(start.elapsed_time(end)))
    return times_ms


def _format_gbps(*, bytes_total: int, time_ms: float) -> float:
    if time_ms <= 0:
        return 0.0
    return (bytes_total / 1e9) / (time_ms / 1e3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--D", type=int, default=256)
    parser.add_argument("--L", type=int, default=4096)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--fused", action="store_true", help="Use fused multi-step kernel.")
    parser.add_argument("--bench", action="store_true", help="Run a small microbench.")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    else:
        dtype = torch.float32
    x = torch.randn(args.B, args.D, args.L, device=device, dtype=dtype)

    # Reference: fp32 accumulate across steps, cast back at the end.
    y_ref_fp32 = x.to(dtype=torch.float32)
    for _ in range(args.steps):
        y_ref_fp32 = _diffusion_torch_fp32_step(y_ref_fp32, alpha=args.alpha)
    y_ref = y_ref_fp32.to(dtype=x.dtype)

    if device.type != "cuda":
        print("[diffusion_triton] CPU selected; Triton path is CUDA-only. Torch reference OK.")
        return

    if args.fused:
        y = diffusion_triton_fused(x, alpha=args.alpha, steps=args.steps)
    else:
        y = diffusion_triton(x, alpha=args.alpha, steps=args.steps)
    max_err = (y.float() - y_ref.float()).abs().max().item()
    print(f"[diffusion_triton] max_abs_error={max_err:.6e}")

    if args.bench:
        # Approx traffic per step: 3 reads + 1 write (boundary negligible).
        n = args.B * args.D * args.L
        bytes_per_elem = x.element_size()
        bytes_per_step = n * bytes_per_elem * 4

        def _fn_triton() -> torch.Tensor:
            return diffusion_step_triton(x, alpha=args.alpha)

        def _fn_triton_fused() -> torch.Tensor:
            return diffusion_triton_fused(x, alpha=args.alpha, steps=args.steps)

        def _fn_torch() -> torch.Tensor:
            return _diffusion_torch(x, alpha=args.alpha)

        t_triton = _bench_cuda(_fn_triton, iters=args.iters, warmup=args.warmup)
        t_triton_fused = _bench_cuda(_fn_triton_fused, iters=args.iters, warmup=args.warmup)
        t_torch = _bench_cuda(_fn_torch, iters=args.iters, warmup=args.warmup)

        t_triton_p50 = float(sorted(t_triton)[len(t_triton) // 2])
        t_triton_fused_p50 = float(sorted(t_triton_fused)[len(t_triton_fused) // 2])
        t_torch_p50 = float(sorted(t_torch)[len(t_torch) // 2])

        gbps_triton = _format_gbps(bytes_total=bytes_per_step, time_ms=t_triton_p50)
        gbps_triton_fused = _format_gbps(bytes_total=bytes_per_step * args.steps, time_ms=t_triton_fused_p50)
        gbps_torch = _format_gbps(bytes_total=bytes_per_step, time_ms=t_torch_p50)

        print(
            "[diffusion_triton][bench] "
            f"B={args.B} D={args.D} L={args.L} dtype={args.dtype} "
            f"torch_step_p50={t_torch_p50:.3f}ms ({gbps_torch:.1f} GB/s) "
            f"triton_step_p50={t_triton_p50:.3f}ms ({gbps_triton:.1f} GB/s) "
            f"triton_fused_steps_p50={t_triton_fused_p50:.3f}ms ({gbps_triton_fused:.1f} GB/s)"
        )


if __name__ == "__main__":
    main()

