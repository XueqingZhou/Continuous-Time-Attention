from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cta.ops.diffusion import diffusion_steps_reference


def _maybe_add_repo_root_to_syspath() -> None:
    # Tests already add `src/` to sys.path. For `kernels/`, also add repo root.
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _assert_close(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.dtype in (torch.float16, torch.bfloat16) or b.dtype in (torch.float16, torch.bfloat16):
        torch.testing.assert_close(a, b, rtol=1e-2, atol=1e-2)
    else:
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("steps", [1, 2, 4, 8])
def test_triton_fused_matches_reference(dtype: torch.dtype, steps: int) -> None:
    _maybe_add_repo_root_to_syspath()
    from kernels.diffusion_triton import diffusion_triton_fused  # noqa: E402

    torch.manual_seed(0)
    b, d, l = 2, 32, 1024
    x = torch.randn(b, d, l, device="cuda", dtype=dtype)
    alpha = 0.1

    y = diffusion_triton_fused(x, alpha=alpha, steps=steps)
    y_ref = diffusion_steps_reference(x, alpha=alpha, steps=steps)
    _assert_close(y, y_ref)

