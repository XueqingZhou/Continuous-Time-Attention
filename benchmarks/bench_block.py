"""
Microbenchmark: Transformer block vs Transformer block + PDE refinement (forward-only).

This benchmark is designed for serving/prefill analysis:
- measures latency scaling with sequence length
- measures peak CUDA memory allocation (approx proxy for activation footprint)

It intentionally includes the transpose/layout overhead incurred by the current
PDE integration pattern in `src/models/transformers.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Allow `from models import ...` from repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from models import create_pde_layer  # noqa: E402


@dataclass
class BenchRow:
    benchmark: str
    variant: str
    device: str
    dtype: str
    pde_type: str
    pde_steps: int
    batch_size: int
    hidden_size: int
    num_heads: int
    seq_len: int
    iters: int
    warmup: int
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    peak_mem_bytes_mean: Optional[float]
    peak_mem_bytes_p95: Optional[float]


def _torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {dtype}")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _percentiles(values: List[float]) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return (
        float(np.mean(arr)),
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 95)),
    )


@torch.no_grad()
def _bench_cuda(fn, iters: int, warmup: int) -> Tuple[List[float], List[int]]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms: List[float] = []
    peak_mems: List[int] = []

    for _ in range(iters):
        torch.cuda.reset_peak_memory_stats()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(float(start.elapsed_time(end)))
        peak_mems.append(int(torch.cuda.max_memory_allocated()))

    return times_ms, peak_mems


@torch.no_grad()
def _bench_cpu(fn, iters: int, warmup: int) -> Tuple[List[float], List[int]]:
    for _ in range(warmup):
        fn()

    times_ms: List[float] = []
    peak_mems: List[int] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
        peak_mems.append(0)
    return times_ms, peak_mems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument(
        "--pde_type",
        type=str,
        default="diffusion",
        choices=["diffusion", "wave", "reaction-diffusion", "advection-diffusion"],
    )
    parser.add_argument("--pde_steps", type=int, default=4, help="Set 0 to benchmark baseline only.")
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="results/bench")
    parser.add_argument("--out_name", type=str, default="bench_block")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.device == "cpu" and args.dtype != "fp32":
        print("[bench_block] CPU only supports fp32 reliably; forcing fp32.")
        args.dtype = "fp32"

    if args.hidden_size % args.num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_heads")

    dtype = _torch_dtype(args.dtype)
    _ensure_dir(args.out_dir)

    rows: List[BenchRow] = []

    # Keep dim_feedforward fixed relative to hidden_size
    transformer = nn.TransformerEncoderLayer(
        d_model=args.hidden_size,
        nhead=args.num_heads,
        dim_feedforward=args.hidden_size * 4,
        dropout=0.0,
        batch_first=True,
        activation="gelu",
        norm_first=True,
    ).to(device=device, dtype=dtype)
    transformer.eval()

    for seq_len in args.seq_lens:
        x = torch.randn(args.batch_size, seq_len, args.hidden_size, device=device, dtype=dtype)

        @torch.no_grad()
        def baseline_fn() -> torch.Tensor:
            return transformer(x)

        if args.device == "cuda":
            times_ms, mems = _bench_cuda(baseline_fn, iters=args.iters, warmup=args.warmup)
            mem_mean = float(np.mean(np.asarray(mems, dtype=np.float64)))
            mem_p95 = float(np.percentile(np.asarray(mems, dtype=np.float64), 95))
        else:
            times_ms, _ = _bench_cpu(baseline_fn, iters=args.iters, warmup=args.warmup)
            mem_mean, mem_p95 = None, None

        t_mean, t_p50, t_p95 = _percentiles(times_ms)
        rows.append(
            BenchRow(
                benchmark="block",
                variant="baseline",
                device=args.device,
                dtype=args.dtype,
                pde_type=args.pde_type,
                pde_steps=0,
                batch_size=args.batch_size,
                hidden_size=args.hidden_size,
                num_heads=args.num_heads,
                seq_len=seq_len,
                iters=args.iters,
                warmup=args.warmup,
                latency_ms_mean=t_mean,
                latency_ms_p50=t_p50,
                latency_ms_p95=t_p95,
                peak_mem_bytes_mean=mem_mean,
                peak_mem_bytes_p95=mem_p95,
            )
        )
        print(f"[bench_block] baseline B={args.batch_size} D={args.hidden_size} L={seq_len} p50={t_p50:.3f}ms p95={t_p95:.3f}ms")

        if args.pde_steps <= 0:
            continue

        pde_layers = [
            create_pde_layer(args.pde_type, args.hidden_size).to(device)
            for _ in range(args.pde_steps)
        ]
        for layer in pde_layers:
            layer.eval()

        @torch.no_grad()
        def pde_fn() -> torch.Tensor:
            y = transformer(x)
            y_t = y.transpose(1, 2).contiguous()  # (B, L, D) -> (B, D, L)
            for layer in pde_layers:
                y_t = layer(y_t)
            return y_t.transpose(1, 2).contiguous()

        if args.device == "cuda":
            times_ms, mems = _bench_cuda(pde_fn, iters=args.iters, warmup=args.warmup)
            mem_mean = float(np.mean(np.asarray(mems, dtype=np.float64)))
            mem_p95 = float(np.percentile(np.asarray(mems, dtype=np.float64), 95))
        else:
            times_ms, _ = _bench_cpu(pde_fn, iters=args.iters, warmup=args.warmup)
            mem_mean, mem_p95 = None, None

        t_mean, t_p50, t_p95 = _percentiles(times_ms)
        rows.append(
            BenchRow(
                benchmark="block",
                variant="transformer+pde",
                device=args.device,
                dtype=args.dtype,
                pde_type=args.pde_type,
                pde_steps=args.pde_steps,
                batch_size=args.batch_size,
                hidden_size=args.hidden_size,
                num_heads=args.num_heads,
                seq_len=seq_len,
                iters=args.iters,
                warmup=args.warmup,
                latency_ms_mean=t_mean,
                latency_ms_p50=t_p50,
                latency_ms_p95=t_p95,
                peak_mem_bytes_mean=mem_mean,
                peak_mem_bytes_p95=mem_p95,
            )
        )
        print(
            f"[bench_block] transformer+pde type={args.pde_type} steps={args.pde_steps} "
            f"B={args.batch_size} D={args.hidden_size} L={seq_len} "
            f"p50={t_p50:.3f}ms p95={t_p95:.3f}ms"
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

    print(f"[bench_block] Wrote {out_csv}")
    print(f"[bench_block] Wrote {out_json}")


if __name__ == "__main__":
    main()

