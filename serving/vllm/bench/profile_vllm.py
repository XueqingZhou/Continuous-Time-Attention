"""
Profile vLLM generate path with torch.profiler.

Outputs:
- trace_<tag>.json
- summary_<tag>.txt
"""

from __future__ import annotations

import argparse
import os
from typing import List

import torch
from torch.profiler import ProfilerActivity, profile

try:
    from vllm import LLM, SamplingParams
except Exception as e:  # noqa: BLE001
    raise RuntimeError(
        "vLLM is required for this profiler. "
        "Install with `pip install vllm` (ensure CUDA build)."
    ) from e


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="results/serving/vllm")
    parser.add_argument("--tag", type=str, default="vllm_profile")
    args = parser.parse_args()

    _ensure_dir(args.out_dir)
    trace_path = os.path.join(args.out_dir, f"trace_{args.tag}.json")
    summary_path = os.path.join(args.out_dir, f"summary_{args.tag}.txt")

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
    
    # Generate prompt with exact token count (prompt_len now means token count)
    prompt = _generate_prompt_with_token_count(tokenizer, args.prompt_len)
    
    # Verify token count for logging
    actual_token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
    print(f"[profile_vllm] prompt_len={args.prompt_len} tokens, actual={actual_token_count} tokens")
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )

    for _ in range(args.warmup):
        llm.generate(prompt, sampling_params)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(args.iters):
            llm.generate(prompt, sampling_params)

    prof.export_chrome_trace(trace_path)
    table = prof.key_averages().table(
        sort_by="self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total",
        row_limit=80,
    )
    with open(summary_path, "w") as f:
        f.write(table)

    print(f"[profile_vllm] Wrote {trace_path}")
    print(f"[profile_vllm] Wrote {summary_path}")


if __name__ == "__main__":
    main()
