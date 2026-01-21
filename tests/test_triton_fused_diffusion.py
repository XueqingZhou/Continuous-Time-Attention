from __future__ import annotations

import pytest
import torch

from cta.ops.diffusion import diffusion_steps_reference

try:
    import triton  # noqa: F401

    _TRITON_AVAILABLE = True
except Exception:  # noqa: BLE001
    _TRITON_AVAILABLE = False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(not _TRITON_AVAILABLE, reason="requires Triton")
@pytest.mark.parametrize("steps", [2, 4, 8])
def test_triton_fused_matches_reference(steps: int) -> None:
    from kernels.diffusion_triton import diffusion_triton_fused  # noqa: WPS433

    torch.manual_seed(0)
    b, d, l = 1, 8, 257
    alpha = 0.1

    x = torch.randn(b, d, l, device="cuda", dtype=torch.bfloat16)
    y = diffusion_triton_fused(x, alpha=alpha, steps=steps)
    y_ref = diffusion_steps_reference(x, alpha=alpha, steps=steps)

    torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)

