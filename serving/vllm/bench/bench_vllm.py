"""
vLLM baseline harness: prefill latency, decode throughput, peak memory.

This script is intentionally minimal and self-contained. It:
- loads a vLLM model
- runs synthetic prompts of configurable length
- reports p50/p95 latency, tokens/s (decode), and peak CUDA memory
- writes CSV/JSON under --out-dir

Example:
python serving/vllm/bench/bench_vllm.py \\
  --model mistralai/Mistral-7B-Instruct-v0.1 \\
  --dtype float16 \\
  --prompt-lens 512 2048 4096 \\
  --max-new-tokens 64 \\
  --iters 5 --warmup 2 \\
  --out-dir results/serving/vllm
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

try:
    from vllm import LLM, SamplingParams
except Exception as e:  # noqa: BLE001
    raise RuntimeError(
        "vLLM is required for this benchmark. "
        "Install with `pip install vllm` (ensure CUDA build)."
    ) from e


@dataclass
class BenchRow:
    benchmark: str
    model: str
    dtype: str
    tp_size: int
    prompt_len: int
    max_new_tokens: int
    prefill_tokens: int
    iters: int
    warmup: int
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    prefill_latency_ms_mean: float
    prefill_latency_ms_p50: float
    prefill_latency_ms_p95: float
    decode_tokens_per_s: Optional[float]
    peak_mem_bytes_mean: Optional[float]
    peak_mem_bytes_p95: Optional[float]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _percentiles(values: List[float]) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return (
        float(np.mean(arr)),
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 95)),
    )


def _maybe_reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _max_allocated() -> Optional[int]:
    if torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    return None


def _nvidia_smi_used_bytes() -> Optional[int]:
    """Return current GPU memory.used in bytes (global, cross-process)."""
    if not torch.cuda.is_available():
        return None
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        # If multiple GPUs exist, this returns multiple lines. This benchmark is
        # typically run with CUDA_VISIBLE_DEVICES set, so take the first line.
        first = out.strip().splitlines()[0].strip()
        mib = int(first)
        return mib * 1024 * 1024
    except Exception:  # noqa: BLE001
        return None


def _generate_prompt_with_token_count(
    tokenizer, target_token_count: int, seed_text: str = " hello"
) -> str:
    """
    Generate a prompt string that tokenizes to exactly target_token_count tokens.
    
    Args:
        tokenizer: The tokenizer to use for encoding/decoding
        target_token_count: Desired number of tokens in the prompt
        seed_text: Base text to repeat (should tokenize cleanly)
    
    Returns:
        A prompt string that tokenizes to target_token_count tokens
    """
    # Helper to ensure we always get a list of token IDs
    def _encode_to_ids(text: str) -> List[int]:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if hasattr(encoded, 'ids'):
            return list(encoded.ids)
        if isinstance(encoded, list):
            return encoded
        return list(encoded)
    
    # First, encode the seed text to see how many tokens it produces
    seed_tokens = _encode_to_ids(seed_text)
    tokens_per_repetition = len(seed_tokens)
    
    if tokens_per_repetition == 0:
        # Fallback: use a single space or a common token
        seed_text = " hello"
        seed_tokens = _encode_to_ids(seed_text)
        tokens_per_repetition = len(seed_tokens)
        if tokens_per_repetition == 0:
            raise ValueError("seed_text must tokenize to at least 1 token")
    
    # Calculate how many repetitions we need
    num_repetitions = max(1, target_token_count // tokens_per_repetition)
    
    # Generate repeated text
    repeated_text = seed_text * num_repetitions
    
    # Encode to get actual token count
    tokens = _encode_to_ids(repeated_text)
    
    # Adjust if we're over or under
    if len(tokens) > target_token_count:
        # Truncate tokens and decode
        tokens = tokens[:target_token_count]
        prompt = tokenizer.decode(tokens, skip_special_tokens=True)
    elif len(tokens) < target_token_count:
        # Try adding more repetitions gradually
        current_count = len(tokens)
        while current_count < target_token_count:
            tokens.extend(seed_tokens[: min(len(seed_tokens), target_token_count - current_count)])
            current_count = len(tokens)
        tokens = tokens[:target_token_count]
        prompt = tokenizer.decode(tokens, skip_special_tokens=True)
    else:
        prompt = repeated_text
    
    # Verify the token count (with some tolerance for tokenizer quirks)
    final_tokens = _encode_to_ids(prompt)
    actual_count = len(final_tokens)
    
    if abs(actual_count - target_token_count) > 2:
        # If still off, just use the tokens directly
        tokens = seed_tokens * num_repetitions
        tokens = tokens[:target_token_count]
        prompt = tokenizer.decode(tokens, skip_special_tokens=True)
        final_tokens = _encode_to_ids(prompt)
        actual_count = len(final_tokens)
    
    return prompt


def _run_one(
    llm: LLM,
    prompt: str,
    sampling_params: SamplingParams,
    *,
    warmup: int,
    iters: int,
) -> Tuple[List[float], List[int]]:
    for _ in range(warmup):
        llm.generate(prompt, sampling_params)
    latencies_ms: List[float] = []
    peaks: List[int] = []
    for _ in range(iters):
        _maybe_reset_peak()
        mem0 = _nvidia_smi_used_bytes() or 0
        t0 = time.perf_counter()
        outputs = llm.generate(prompt, sampling_params)
        t1 = time.perf_counter()
        mem1 = _nvidia_smi_used_bytes() or 0
        latencies_ms.append((t1 - t0) * 1000.0)
        peak = max(mem0, mem1, int(_max_allocated() or 0))
        peaks.append(peak)
        # Drop outputs to free memory ASAP
        del outputs
    return latencies_ms, peaks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="HF model id or local path.")
    parser.add_argument("--tokenizer", type=str, default=None, help="Optional tokenizer override.")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--prompt-lens", type=int, nargs="+", default=[512, 2048, 4096, 8192])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--prefill-tokens",
        type=int,
        default=1,
        help="Use a tiny decode length to approximate prefill latency.",
    )
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out-dir", type=str, default="results/serving/vllm")
    parser.add_argument("--out-name", type=str, default="bench_vllm")
    args = parser.parse_args()

    _ensure_dir(args.out_dir)

    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        dtype=args.dtype,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
    )

    # Access the tokenizer from the LLM instance
    # vLLM's tokenizer is a wrapper; use _tokenizer for the underlying transformers tokenizer
    tokenizer = getattr(llm.llm_engine.tokenizer, "_tokenizer", llm.llm_engine.tokenizer)

    rows: List[BenchRow] = []

    for prompt_len in args.prompt_lens:
        # Generate prompt with exact token count (prompt_len now means token count)
        prompt = _generate_prompt_with_token_count(tokenizer, prompt_len)
        
        # Verify token count for logging
        actual_token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
        if abs(actual_token_count - prompt_len) > 2:
            print(
                f"[bench_vllm] WARNING: prompt_len={prompt_len} but actual tokens={actual_token_count}"
            )
        prefill_params = SamplingParams(
            max_tokens=args.prefill_tokens,
            temperature=0.0,
            top_p=1.0,
        )
        full_params = SamplingParams(
            max_tokens=args.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
        )

        prefill_lat_ms, _ = _run_one(
            llm,
            prompt,
            prefill_params,
            warmup=args.warmup,
            iters=args.iters,
        )
        lat_ms, peaks = _run_one(
            llm,
            prompt,
            full_params,
            warmup=args.warmup,
            iters=args.iters,
        )
        t_mean, t_p50, t_p95 = _percentiles(lat_ms)
        pre_t_mean, pre_t_p50, pre_t_p95 = _percentiles(prefill_lat_ms)
        peaks = [p for p in peaks if p is not None]
        mem_mean = float(np.mean(np.asarray(peaks, dtype=np.float64))) if peaks else None
        mem_p95 = float(np.percentile(np.asarray(peaks, dtype=np.float64), 95)) if peaks else None

        # Decode throughput: tokens/s averaged over runs (prefill+decode end-to-end)
        tokens_generated = args.max_new_tokens
        decode_time_ms = t_mean - pre_t_mean
        if decode_time_ms <= 0:
            decode_time_ms = t_mean
        toks_per_s = (tokens_generated / (decode_time_ms / 1000.0)) if decode_time_ms > 0 else None

        rows.append(
            BenchRow(
                benchmark="vllm_baseline",
                model=args.model,
                dtype=args.dtype,
                tp_size=args.tp_size,
                prompt_len=prompt_len,
                max_new_tokens=args.max_new_tokens,
                prefill_tokens=args.prefill_tokens,
                iters=args.iters,
                warmup=args.warmup,
                latency_ms_mean=t_mean,
                latency_ms_p50=t_p50,
                latency_ms_p95=t_p95,
                prefill_latency_ms_mean=pre_t_mean,
                prefill_latency_ms_p50=pre_t_p50,
                prefill_latency_ms_p95=pre_t_p95,
                decode_tokens_per_s=toks_per_s,
                peak_mem_bytes_mean=mem_mean,
                peak_mem_bytes_p95=mem_p95,
            )
        )

        toks_per_s_str = f"{toks_per_s:.2f}" if toks_per_s is not None else "n/a"
        print(
            f"[bench_vllm] prompt_len={prompt_len} "
            f"prefill_p50={pre_t_p50:.2f}ms "
            f"e2e_p50={t_p50:.2f}ms e2e_p95={t_p95:.2f}ms "
            f"decode_toks/s={toks_per_s_str}"
        )

    out_csv = os.path.join(args.out_dir, f"{args.out_name}.csv")
    out_json = os.path.join(args.out_dir, f"{args.out_name}.json")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    with open(out_json, "w") as f:
        json.dump([asdict(r) for r in rows], f, indent=2)

    print(f"[bench_vllm] Wrote {out_csv}")
    print(f"[bench_vllm] Wrote {out_json}")


if __name__ == "__main__":
    main()

