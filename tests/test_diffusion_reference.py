from __future__ import annotations

import pytest
import torch

from cta.ops.diffusion import diffusion_step_reference, diffusion_steps_reference
from models.pde_layers import DiffusionPDELayer


def _assert_close(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.dtype in (torch.float16, torch.bfloat16) or b.dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        torch.testing.assert_close(a, b, rtol=1e-2, atol=1e-2)
    else:
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def _causal_diffusion_step_reference(
    x: torch.Tensor, *, alpha: float, layout: str
) -> torch.Tensor:
    """Hand-rolled causal diffusion reference (backward-difference stencil)."""
    x_fp32 = x.to(dtype=torch.float32)
    if layout == "bdl":
        if x_fp32.size(2) < 3:
            return x
        lap = x_fp32[:, :, 2:] - 2.0 * x_fp32[:, :, 1:-1] + x_fp32[:, :, :-2]
        y = torch.cat([x_fp32[:, :, :2], x_fp32[:, :, 2:] + alpha * lap], dim=2)
    else:
        if x_fp32.size(1) < 3:
            return x
        lap = x_fp32[:, 2:, :] - 2.0 * x_fp32[:, 1:-1, :] + x_fp32[:, :-2, :]
        y = torch.cat([x_fp32[:, :2, :], x_fp32[:, 2:, :] + alpha * lap], dim=1)
    return y.to(dtype=x.dtype)


# ---- boundary / shape tests ------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("l", [1, 2, 3, 4, 5, 17])
def test_diffusion_step_reference_boundary_copy(dtype: torch.dtype, l: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, l, dtype=dtype)
    alpha = 0.1

    y = diffusion_step_reference(x, alpha=alpha)
    assert y.shape == x.shape
    assert y.dtype == x.dtype

    if l >= 1:
        _assert_close(y[:, :, 0], x[:, :, 0])
    if l >= 2:
        _assert_close(y[:, :, -1], x[:, :, -1])

    if l < 3:
        _assert_close(y, x)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("steps", [0, 1, 2, 4, 8])
def test_diffusion_steps_reference_shape_dtype(dtype: torch.dtype, steps: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 11, dtype=dtype)
    y = diffusion_steps_reference(x, alpha=0.1, steps=steps)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


# ---- layer vs. reference correctness ---------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("steps", [1, 2, 4, 8])
@pytest.mark.parametrize("alpha", [0.0, 0.03, 0.1])
def test_diffusion_layer_matches_reference(
    dtype: torch.dtype, steps: int, alpha: float
) -> None:
    torch.manual_seed(0)
    b, d, l = 2, 4, 23
    x = torch.randn(b, d, l, dtype=dtype)

    layer = DiffusionPDELayer(hidden_size=d, alpha_init=float(alpha))
    layer.alpha.data.fill_(float(alpha))

    y = x
    for _ in range(steps):
        y = layer(y)

    y_ref = diffusion_steps_reference(x, alpha=float(alpha), steps=steps)
    _assert_close(y, y_ref)


@pytest.mark.parametrize("layout", ["bdl", "bld"])
def test_causal_diffusion_layer_matches_reference(layout: str) -> None:
    torch.manual_seed(0)
    alpha = 0.1

    if layout == "bdl":
        x = torch.randn(2, 4, 23, dtype=torch.float32)
    else:
        x = torch.randn(2, 23, 4, dtype=torch.float32)

    layer = DiffusionPDELayer(hidden_size=4, alpha_init=alpha, layout=layout, causal=True)
    layer.alpha.data.fill_(alpha)

    y = layer(x)
    y_ref = _causal_diffusion_step_reference(x, alpha=alpha, layout=layout)
    _assert_close(y, y_ref)


# ---- error handling ---------------------------------------------------------

def test_diffusion_reference_invalid_ndim() -> None:
    x = torch.randn(2, 3)
    with pytest.raises(ValueError):
        _ = diffusion_step_reference(x, alpha=0.1)


def test_diffusion_reference_invalid_steps() -> None:
    x = torch.randn(2, 3, 4)
    with pytest.raises(ValueError):
        _ = diffusion_steps_reference(x, alpha=0.1, steps=-1)


# ---- BLD layout vs. BDL reference ------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("steps", [1, 2, 4, 8])
def test_diffusion_layer_layout_bld_matches_reference(steps: int) -> None:
    torch.manual_seed(0)
    b, d, l = 2, 16, 257
    x_bld = torch.randn(b, l, d, device="cuda", dtype=torch.bfloat16)
    alpha = 0.1

    # Reference uses BDL; transpose to compare
    x_bdl = x_bld.transpose(1, 2)
    y_ref = diffusion_steps_reference(x_bdl, alpha=alpha, steps=steps)
    y_ref_bld = y_ref.transpose(1, 2)

    layer = DiffusionPDELayer(hidden_size=d, alpha_init=alpha, layout="bld").to(x_bld.device)
    layer.alpha.data.fill_(alpha)

    y = x_bld
    for _ in range(steps):
        y = layer(y)

    _assert_close(y, y_ref_bld)
