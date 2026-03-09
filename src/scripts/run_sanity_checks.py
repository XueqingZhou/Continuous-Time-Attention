"""Quick sanity checks for LM pipeline and causality."""

from __future__ import annotations

import argparse
from typing import Tuple

import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.language_modeling import _ensure_pad_token, _group_texts, _tokenize_texts
from models import PDETransformerLM, StandardTransformerLM


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOKENIZER_PATH = REPO_ROOT / "src" / "tokenizer" / "bert-base-uncased"
HUB_FALLBACK_TOKENIZER = "bert-base-uncased"


def _resolve_tokenizer_path(path: str) -> str:
    """Return *path* if it exists locally, otherwise fall back to the HF Hub."""
    if Path(path).is_dir():
        return path
    print(
        f"[sanity] Local tokenizer not found at {path}; "
        f"falling back to '{HUB_FALLBACK_TOKENIZER}' from HuggingFace Hub."
    )
    return HUB_FALLBACK_TOKENIZER


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
    """Check tokenizer + block grouping logic without remote dataset access."""
    tokenizer_path = _resolve_tokenizer_path(tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    _ensure_pad_token(tokenizer)

    examples = {
        "text": [
            "continuous time attention " * 24,
            "diffusion refinement improves long-context modeling " * 16,
        ]
    }
    tokenized = _tokenize_texts(examples, tokenizer=tokenizer, add_eos=True)
    grouped = _group_texts(tokenized, block_size=32)
    if not grouped["input_ids"]:
        return False, "block_dataset produced no blocks"

    input_ids = grouped["input_ids"][0]
    attn = grouped["attention_mask"][0]
    ok = len(input_ids) == 32 and sum(attn) == 32
    msg = f"block_dataset len={len(input_ids)} attn_sum={int(sum(attn))}"
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
    parser = argparse.ArgumentParser(description="Run lightweight CTA sanity checks")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=str(DEFAULT_TOKENIZER_PATH),
        help="Path to a local tokenizer directory. Falls back to HuggingFace Hub.",
    )
    args = parser.parse_args()

    tokenizer_path = args.tokenizer_path
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
