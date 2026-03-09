"""Tests for all PDE layer types (shape, dtype, mask, layout, causal)."""

from __future__ import annotations

import pytest
import torch

from models.pde_layers import (
    AdvectionDiffusionPDELayer,
    DiffusionPDELayer,
    ReactionDiffusionPDELayer,
    WavePDELayer,
    create_pde_layer,
)

PDE_CLASSES = [
    DiffusionPDELayer,
    WavePDELayer,
    ReactionDiffusionPDELayer,
    AdvectionDiffusionPDELayer,
]

PDE_TYPE_NAMES = ["diffusion", "wave", "reaction-diffusion", "advection-diffusion"]


# ---------------------------------------------------------------------------
# Shape / dtype preservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", PDE_CLASSES)
@pytest.mark.parametrize("layout", ["bdl", "bld"])
@pytest.mark.parametrize("causal", [False, True])
def test_shape_preserved(cls, layout: str, causal: bool) -> None:
    torch.manual_seed(0)
    b, d, l = 2, 8, 17
    if layout == "bdl":
        x = torch.randn(b, d, l)
    else:
        x = torch.randn(b, l, d)

    layer = cls(hidden_size=d, layout=layout, causal=causal)
    y = layer(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


@pytest.mark.parametrize("cls", PDE_CLASSES)
@pytest.mark.parametrize("l", [1, 2])
def test_short_sequence_passthrough(cls, l: int) -> None:
    """Sequences shorter than 3 should be returned unchanged."""
    torch.manual_seed(0)
    x = torch.randn(1, 4, l)
    layer = cls(hidden_size=4, layout="bdl")
    y = layer(x)
    torch.testing.assert_close(y, x)


# ---------------------------------------------------------------------------
# Mask-aware: padding tokens should not change
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", PDE_CLASSES)
@pytest.mark.parametrize("layout", ["bdl", "bld"])
def test_padding_tokens_unchanged(cls, layout: str) -> None:
    """Padded positions (mask=0) must remain identical after the PDE step."""
    torch.manual_seed(42)
    b, d, l = 1, 4, 10
    if layout == "bdl":
        x = torch.randn(b, d, l)
    else:
        x = torch.randn(b, l, d)

    attention_mask = torch.ones(b, l, dtype=torch.long)
    attention_mask[:, 7:] = 0  # last 3 positions are padding

    layer = cls(hidden_size=d, layout=layout)
    y = layer(x, attention_mask=attention_mask)

    if layout == "bdl":
        torch.testing.assert_close(y[:, :, 7:], x[:, :, 7:])
    else:
        torch.testing.assert_close(y[:, 7:, :], x[:, 7:, :])


# ---------------------------------------------------------------------------
# Padding perturbation invariance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", PDE_CLASSES)
@pytest.mark.parametrize("layout", ["bdl", "bld"])
def test_padding_perturbation_invariance(cls, layout: str) -> None:
    """Changing values in masked positions must not affect valid outputs."""
    torch.manual_seed(7)
    b, d, l = 1, 4, 12
    if layout == "bdl":
        x = torch.randn(b, d, l)
    else:
        x = torch.randn(b, l, d)

    attention_mask = torch.ones(b, l, dtype=torch.long)
    attention_mask[:, 8:] = 0

    layer = cls(hidden_size=d, layout=layout)

    y1 = layer(x, attention_mask=attention_mask)

    x2 = x.clone()
    if layout == "bdl":
        x2[:, :, 8:] = torch.randn(b, d, 4)
    else:
        x2[:, 8:, :] = torch.randn(b, 4, d)

    y2 = layer(x2, attention_mask=attention_mask)

    if layout == "bdl":
        torch.testing.assert_close(y1[:, :, :8], y2[:, :, :8], rtol=1e-5, atol=1e-5)
    else:
        torch.testing.assert_close(y1[:, :8, :], y2[:, :8, :], rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# BDL vs BLD layout equivalence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", PDE_CLASSES)
@pytest.mark.parametrize("causal", [False, True])
def test_bdl_bld_equivalence(cls, causal: bool) -> None:
    """BDL and BLD layouts should produce equivalent results."""
    torch.manual_seed(99)
    b, d, l = 2, 8, 20
    x_bdl = torch.randn(b, d, l)
    x_bld = x_bdl.transpose(1, 2).contiguous()

    layer_bdl = cls(hidden_size=d, layout="bdl", causal=causal)
    layer_bld = cls(hidden_size=d, layout="bld", causal=causal)

    # Copy parameters
    for p_bld, p_bdl in zip(layer_bld.parameters(), layer_bdl.parameters()):
        p_bld.data.copy_(p_bdl.data)

    y_bdl = layer_bdl(x_bdl)
    y_bld = layer_bld(x_bld)

    torch.testing.assert_close(
        y_bdl, y_bld.transpose(1, 2), rtol=1e-5, atol=1e-5,
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pde_type", PDE_TYPE_NAMES)
def test_create_pde_layer(pde_type: str) -> None:
    layer = create_pde_layer(pde_type, hidden_size=16)
    x = torch.randn(1, 16, 10)
    y = layer(x)
    assert y.shape == x.shape


def test_create_pde_layer_invalid() -> None:
    with pytest.raises(ValueError, match="Unknown PDE type"):
        create_pde_layer("nonexistent", hidden_size=16)


def test_create_pde_layer_invalid_layout() -> None:
    with pytest.raises(ValueError, match="layout"):
        DiffusionPDELayer(hidden_size=16, layout="xyz")


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", PDE_CLASSES)
def test_gradient_flows(cls) -> None:
    """All PDE layers must be differentiable."""
    torch.manual_seed(0)
    x = torch.randn(1, 8, 16, requires_grad=True)
    layer = cls(hidden_size=8, layout="bdl")
    y = layer(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
