"""
CTA post-attention mixer for vLLM integration (prefill-oriented).

This module is intentionally self-contained to make it easy to import from a
patched vLLM codebase. It supports:
- Torch reference path (layout aware, fp32 accumulate)
- Triton fused path (steps in {2, 4, 8}) with optional BLD<->BDL transpose
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

try:
    from torch.profiler import record_function
except Exception:  # noqa: BLE001
    def record_function(_name: str):  # type: ignore
        return nullcontext()

from cta.ops.diffusion import diffusion_steps_reference


def _diffusion_steps_bld(
    x_bld: torch.Tensor, *, alpha: float, steps: int
) -> torch.Tensor:
    if steps <= 0:
        return x_bld
    if x_bld.ndim != 3:
        raise ValueError(f"expected x_bld.ndim == 3, got {x_bld.ndim}")
    if x_bld.size(1) < 3:
        return x_bld

    y = x_bld.to(dtype=torch.float32)
    for _ in range(steps):
        lap = y[:, 2:, :] - 2.0 * y[:, 1:-1, :] + y[:, :-2, :]
        y = torch.cat(
            [y[:, :1, :], y[:, 1:-1, :] + float(alpha) * lap, y[:, -1:, :]],
            dim=1,
        )
    return y.to(dtype=x_bld.dtype)


def _infer_prefill(attn_metadata: Optional[object]) -> bool:
    if attn_metadata is None:
        return True
    for attr in ("is_prefill", "prefill"):
        val = getattr(attn_metadata, attr, None)
        if isinstance(val, bool):
            return val
    num_prefill = getattr(attn_metadata, "num_prefill_tokens", None)
    if isinstance(num_prefill, int):
        return num_prefill > 0
    if getattr(attn_metadata, "prefill_metadata", None) is not None:
        return True
    return True


def _to_int(value: object) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return int(value.item())
    return None


def _pack_bld_from_query_start_loc(
    x_flat: torch.Tensor,
    query_start_loc: object,
    max_query_len: int,
) -> tuple[torch.Tensor, list[int]]:
    if isinstance(query_start_loc, torch.Tensor):
        qsl = query_start_loc.detach().cpu().tolist()
    else:
        qsl = list(query_start_loc)
    num_reqs = len(qsl) - 1
    bld = torch.zeros(
        (num_reqs, max_query_len, x_flat.shape[-1]),
        device=x_flat.device,
        dtype=x_flat.dtype,
    )
    for i in range(num_reqs):
        start = int(qsl[i])
        end = int(qsl[i + 1])
        if end > start:
            bld[i, : end - start, :] = x_flat[start:end]
    return bld, qsl


def _unpack_bld_to_flat(
    x_bld: torch.Tensor,
    query_start_loc: list[int],
    total_tokens: int,
) -> torch.Tensor:
    out = torch.empty(
        (total_tokens, x_bld.shape[-1]),
        device=x_bld.device,
        dtype=x_bld.dtype,
    )
    num_reqs = len(query_start_loc) - 1
    for i in range(num_reqs):
        start = int(query_start_loc[i])
        end = int(query_start_loc[i + 1])
        if end > start:
            out[start:end] = x_bld[i, : end - start, :]
    return out


class _BufferPool:
    def __init__(self) -> None:
        self._buf: Optional[torch.Tensor] = None

    def get(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if (
            self._buf is None
            or self._buf.shape != shape
            or self._buf.dtype != dtype
            or self._buf.device != device
        ):
            self._buf = torch.empty(shape, dtype=dtype, device=device)
        return self._buf


@dataclass
class CtaConfig:
    enabled: bool = True
    prefill_only: bool = True
    steps: int = 4
    alpha: float = 0.10
    layout: str = "bld"
    use_triton: bool = True
    fused: bool = True
    allow_transpose: bool = True
    min_seq_len: int = 0
    steps_policy: str = "fixed"
    steps_scale: int = 2048
    max_steps: int = 8
    use_buffer_pool: bool = False
    trace: bool = True


class CtaPostAttnMixer(nn.Module):
    """Apply CTA diffusion after attention output (prefill-only by default)."""

    def __init__(self, config: CtaConfig):
        super().__init__()
        if config.layout not in {"bld", "bdl"}:
            raise ValueError("layout must be 'bld' or 'bdl'")
        self.config = config
        self._triton_available = False
        if config.use_triton:
            try:
                from kernels.diffusion_triton import (  # noqa: WPS433
                    diffusion_triton,
                    diffusion_triton_fused,
                )

                self._triton_available = True
                self._diffusion_triton = diffusion_triton
                self._diffusion_triton_fused = diffusion_triton_fused
            except Exception:
                self._triton_available = False
        self._buffer_pool_bdl = _BufferPool() if config.use_buffer_pool else None
        self._buffer_pool_bld = _BufferPool() if config.use_buffer_pool else None

    def _resolve_steps(self, seq_len: int) -> int:
        if seq_len < self.config.min_seq_len:
            return 0
        policy = self.config.steps_policy
        if policy == "fixed":
            steps = self.config.steps
        elif policy == "linear":
            steps = max(1, seq_len // max(1, self.config.steps_scale))
        elif policy == "log2":
            steps = max(1, int(math.log2(max(2, seq_len))))
        else:
            steps = self.config.steps
        return min(self.config.max_steps, int(steps))

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        *,
        attn_metadata: Optional[object] = None,
        is_prefill: Optional[bool] = None,
    ) -> torch.Tensor:
        if x.ndim == 2:
            return self._forward_packed(
                x, attn_metadata=attn_metadata, is_prefill=is_prefill
            )
        if not self.config.enabled:
            return x
        if self.config.prefill_only:
            if is_prefill is None:
                is_prefill = _infer_prefill(attn_metadata)
            if not is_prefill:
                return x

        seq_dim = 2 if self.config.layout == "bdl" else 1
        steps = self._resolve_steps(x.size(seq_dim))
        if steps <= 0:
            return x

        use_triton = (
            self.config.use_triton
            and self._triton_available
            and x.device.type == "cuda"
        )
        use_fused = self.config.fused and steps in {1, 2, 4, 8}

        ctx = record_function("cta_post_attn_mixer") if self.config.trace else nullcontext()
        with ctx:
            if use_triton:
                if self.config.layout == "bdl":
                    if use_fused:
                        return self._diffusion_triton_fused(
                            x, alpha=self.config.alpha, steps=steps
                        )
                    return self._diffusion_triton(
                        x, alpha=self.config.alpha, steps=steps
                    )

                # layout=bld: optionally transpose to BDL for Triton
                if not self.config.allow_transpose:
                    return _diffusion_steps_bld(
                        x, alpha=self.config.alpha, steps=steps
                    )
                if self._buffer_pool_bdl is not None:
                    x_bdl = self._buffer_pool_bdl.get(
                        x.transpose(1, 2).shape,
                        dtype=x.dtype,
                        device=x.device,
                    )
                    x_bdl.copy_(x.transpose(1, 2))
                else:
                    x_bdl = x.transpose(1, 2).contiguous()

                if use_fused:
                    y_bdl = self._diffusion_triton_fused(
                        x_bdl, alpha=self.config.alpha, steps=steps
                    )
                else:
                    y_bdl = self._diffusion_triton(
                        x_bdl, alpha=self.config.alpha, steps=steps
                    )

                if self._buffer_pool_bld is not None:
                    y_bld = self._buffer_pool_bld.get(
                        y_bdl.transpose(1, 2).shape,
                        dtype=y_bdl.dtype,
                        device=y_bdl.device,
                    )
                    y_bld.copy_(y_bdl.transpose(1, 2))
                    return y_bld
                return y_bdl.transpose(1, 2).contiguous()

            # Torch fallback
            if self.config.layout == "bdl":
                return diffusion_steps_reference(
                    x, alpha=self.config.alpha, steps=steps
                )
            return _diffusion_steps_bld(x, alpha=self.config.alpha, steps=steps)

    def _forward_packed(
        self,
        x: torch.Tensor,
        *,
        attn_metadata: Optional[object],
        is_prefill: Optional[bool],
    ) -> torch.Tensor:
        if not self.config.enabled:
            return x
        if attn_metadata is None:
            return x
        if self.config.prefill_only:
            if is_prefill is None:
                is_prefill = _infer_prefill(attn_metadata)
            if not is_prefill:
                return x

        max_query_len = _to_int(getattr(attn_metadata, "max_query_len", None))
        query_start_loc = getattr(attn_metadata, "query_start_loc", None)
        if max_query_len is None or query_start_loc is None:
            return x
        if max_query_len < 3:
            return x

        x_bld, qsl = _pack_bld_from_query_start_loc(
            x, query_start_loc, max_query_len
        )
        y_bld = self.forward(x_bld, attn_metadata=None, is_prefill=is_prefill)
        return _unpack_bld_to_flat(y_bld, qsl, x.shape[0])

