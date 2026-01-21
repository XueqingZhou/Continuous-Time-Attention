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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("l", [1, 2, 3, 4, 17])
def test_triton_step_backward_matches_reference(l: int) -> None:
    _maybe_add_repo_root_to_syspath()
    from kernels.diffusion_triton import diffusion_step_triton  # noqa: E402

    torch.manual_seed(0)
    b, d = 2, 4
    alpha = 0.1
    x = torch.randn(
        b, d, l, device="cuda", dtype=torch.float32, requires_grad=True
    )
    grad_out = torch.randn_like(x)

    y = diffusion_step_triton(x, alpha=alpha)
    (y * grad_out).sum().backward()
    grad_triton = x.grad.detach()

    x_ref = x.detach().clone().requires_grad_(True)
    y_ref = diffusion_steps_reference(x_ref, alpha=alpha, steps=1)
    (y_ref * grad_out).sum().backward()
    grad_ref = x_ref.grad.detach()

    torch.testing.assert_close(grad_triton, grad_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_triton_fused_backward_matches_reference() -> None:
    _maybe_add_repo_root_to_syspath()
    from kernels.diffusion_triton import diffusion_triton_fused  # noqa: E402

    torch.manual_seed(0)
    b, d, l = 2, 8, 33
    alpha = 0.1
    steps = 2
    x = torch.randn(
        b, d, l, device="cuda", dtype=torch.float32, requires_grad=True
    )
    grad_out = torch.randn_like(x)

    y = diffusion_triton_fused(x, alpha=alpha, steps=steps)
    (y * grad_out).sum().backward()
    grad_triton = x.grad.detach()

    x_ref = x.detach().clone().requires_grad_(True)
    y_ref = diffusion_steps_reference(x_ref, alpha=alpha, steps=steps)
    (y_ref * grad_out).sum().backward()
    grad_ref = x_ref.grad.detach()

    torch.testing.assert_close(grad_triton, grad_ref, rtol=1e-4, atol=1e-4)
