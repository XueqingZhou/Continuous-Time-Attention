from __future__ import annotations

import torch

from models import PDETransformerClassifier


@torch.no_grad()
def test_pde_classifier_ignores_padded_tokens() -> None:
    torch.manual_seed(0)

    model = PDETransformerClassifier(
        vocab_size=256,
        embed_dim=32,
        num_heads=4,
        hidden_dim=64,
        num_layers=2,
        num_classes=3,
        pde_type="diffusion",
        pde_steps=2,
        pde_layout="bld",
        max_length=16,
        dropout=0.0,
    )
    model.eval()

    input_ids = torch.tensor([[5, 7, 11, 13, 17, 0, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]])

    perturbed_input_ids = input_ids.clone()
    perturbed_input_ids[:, 5:] = torch.tensor([101, 102, 103])

    logits_a = model(input_ids, attention_mask)
    logits_b = model(perturbed_input_ids, attention_mask)

    torch.testing.assert_close(logits_a, logits_b, rtol=1e-5, atol=1e-5)
