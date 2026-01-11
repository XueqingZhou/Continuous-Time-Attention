"""
Profiler entrypoint for Continuous-Time Attention (systems view).

This script produces:
  - Chrome trace: trace.json
  - Operator summary: summary.txt

It is intended for prefill analysis (forward-only) on synthetic inputs.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function

# Allow `from models import ...` from repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from models import create_pde_layer  # noqa: E402


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {dtype}")


@torch.no_grad()
def run_profile(
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_heads: int,
    pde_type: str,
    pde_steps: int,
    mode: str,
    out_dir: str,
    tag: str,
    iters: int,
    warmup: int,
) -> None:
    if mode not in {"pde_only", "block"}:
        raise ValueError("mode must be one of: pde_only, block")

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    _ensure_dir(out_dir)
    trace_path = os.path.join(out_dir, f"trace_{tag}.json")
    summary_path = os.path.join(out_dir, f"summary_{tag}.txt")

    # Build workload
    if mode == "pde_only":
        x = torch.randn(batch_size, hidden_size, seq_len, device=device, dtype=dtype)
        pde_layers = [create_pde_layer(pde_type, hidden_size).to(device) for _ in range(pde_steps)]
        for l in pde_layers:
            l.eval()

        def step() -> torch.Tensor:
            y = x
            with record_function("cta_pde_loop"):
                for l in pde_layers:
                    y = l(y)
            return y

    else:
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        transformer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        ).to(device=device, dtype=dtype)
        transformer.eval()

        x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
        pde_layers = [create_pde_layer(pde_type, hidden_size).to(device) for _ in range(pde_steps)]
        for l in pde_layers:
            l.eval()

        def step() -> torch.Tensor:
            with record_function("cta_transformer_layer"):
                y = transformer(x)
            with record_function("cta_transpose_to_bdl"):
                y_t = y.transpose(1, 2).contiguous()
            with record_function("cta_pde_loop"):
                for l in pde_layers:
                    y_t = l(y_t)
            with record_function("cta_transpose_to_bld"):
                return y_t.transpose(1, 2).contiguous()

    # Warmup (not profiled)
    for _ in range(warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(iters):
            step()
            if device.type == "cuda":
                torch.cuda.synchronize()

    prof.export_chrome_trace(trace_path)
    table = prof.key_averages().table(
        sort_by="self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total",
        row_limit=50,
    )
    with open(summary_path, "w") as f:
        f.write(table)

    print(f"[profile_pde] Wrote {trace_path}")
    print(f"[profile_pde] Wrote {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--mode", type=str, default="block", choices=["pde_only", "block"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument(
        "--pde_type",
        type=str,
        default="diffusion",
        choices=["diffusion", "wave", "reaction-diffusion", "advection-diffusion"],
    )
    parser.add_argument("--pde_steps", type=int, default=4)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="profiling/out")
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.device == "cpu" and args.dtype != "fp32":
        print("[profile_pde] CPU only supports fp32 reliably; forcing fp32.")
        args.dtype = "fp32"
    dtype = _torch_dtype(args.dtype)

    tag = args.tag
    if tag is None:
        tag = f"{args.mode}_B{args.batch_size}_L{args.seq_len}_D{args.hidden_size}_h{args.num_heads}_{args.pde_type}_s{args.pde_steps}_{args.dtype}_{args.device}"

    run_profile(
        device=device,
        dtype=dtype,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        pde_type=args.pde_type,
        pde_steps=args.pde_steps,
        mode=args.mode,
        out_dir=args.out_dir,
        tag=tag,
        iters=args.iters,
        warmup=args.warmup,
    )


if __name__ == "__main__":
    main()

