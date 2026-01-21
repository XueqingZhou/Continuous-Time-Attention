"""Quick sanity checks for LM pipeline and causality."""

from __future__ import annotations

from typing import Tuple

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.language_modeling import prepare_lm_data
from models import PDETransformerLM, StandardTransformerLM


def _seed_all(seed: int) -> None:
    """Set all relevant RNG seeds."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_causal_invariance() -> Tuple[bool, str]:
    """Check that future tokens do not affect past outputs."""
    _seed_all(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = StandardTransformerLM(
        vocab_size=1000,
        embed_dim=64,
        num_heads=4,
        hidden_dim=128,
        num_layers=2,
        max_length=32,
    ).to(device)
    model.eval()

    input_ids = torch.randint(0, 1000, (2, 16), device=device)
    attention_mask = torch.ones_like(input_ids, device=device)

    with torch.no_grad():
        outputs_1 = model(input_ids, attention_mask)

    input_ids_shuffled = input_ids.clone()
    input_ids_shuffled[:, 8:] = torch.randint(0, 1000, (2, 8), device=device)

    with torch.no_grad():
        outputs_2 = model(input_ids_shuffled, attention_mask)

    max_diff = torch.abs(outputs_1[:, :8, :] - outputs_2[:, :8, :]).max().item()
    ok = max_diff < 1e-5
    msg = f"causal_invariance max_diff={max_diff:.2e}"
    return ok, msg


def check_block_dataset(tokenizer_path: str) -> Tuple[bool, str]:
    """Check LM block dataset shape and padding-free attention mask."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    train_ds, _, _ = prepare_lm_data(
        tokenizer_path=tokenizer_path,
        max_length=32,
        block_size=32,
        train_sample_size=2,
        val_sample_size=2,
        add_eos=True,
    )
    sample = train_ds[0]
    input_ids = sample["input_ids"]
    attn = sample["attention_mask"]
    ok = input_ids.shape[0] == 32 and attn.sum().item() == 32
    msg = f"block_dataset len={input_ids.shape[0]} attn_sum={int(attn.sum().item())}"
    return ok, msg


def check_seed_stability() -> Tuple[bool, str]:
    """Check deterministic forward outputs with fixed seed."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _seed_all(42)
    model_1 = PDETransformerLM(
        vocab_size=1000,
        embed_dim=64,
        num_heads=4,
        hidden_dim=128,
        num_layers=2,
        pde_type="diffusion",
        pde_steps=2,
        max_length=32,
    ).to(device)
    model_1.eval()

    _seed_all(42)
    model_2 = PDETransformerLM(
        vocab_size=1000,
        embed_dim=64,
        num_heads=4,
        hidden_dim=128,
        num_layers=2,
        pde_type="diffusion",
        pde_steps=2,
        max_length=32,
    ).to(device)
    model_2.eval()

    input_ids = torch.randint(0, 1000, (1, 16), device=device)
    attention_mask = torch.ones_like(input_ids, device=device)

    with torch.no_grad():
        out_1 = model_1(input_ids, attention_mask)
        out_2 = model_2(input_ids, attention_mask)

    max_diff = torch.abs(out_1 - out_2).max().item()
    ok = max_diff < 1e-6
    msg = f"seed_stability max_diff={max_diff:.2e}"
    return ok, msg


def main() -> None:
    tokenizer_path = "./local_models/tinyllama"
    checks = [
        ("causal_invariance", check_causal_invariance),
        ("block_dataset", lambda: check_block_dataset(tokenizer_path)),
        ("seed_stability", check_seed_stability),
    ]

    all_ok = True
    for name, fn in checks:
        ok, msg = fn()
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {name}: {msg}")
        all_ok = all_ok and ok

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
