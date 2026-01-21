"""
Microbenchmark: Triton diffusion multi-step fusion (forward-only).

This benchmark compares:
- non-fused: Python loop over single-step Triton kernel
- fused: one Triton launch for multiple steps (time-tiling + halo)

Outputs:
- CSV/JSON under `results/bench/`
- (optional) profiler traces & summaries for fused vs non-fused
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

# Allow importing both `src/` and `kernels/` from repo root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for p in (REPO_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from kernels.diffusion_triton import (  # noqa: E402
    diffusion_triton,
    diffusion_triton_fused,
)


@dataclass
class BenchRow:
    benchmark: str
    variant: str
    device: str
    dtype: str
    steps: int
    batch_size: int
    hidden_size: int
    seq_len: int
    iters: int
    warmup: int
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    peak_mem_bytes_mean: Optional[float]
    peak_mem_bytes_p95: Optional[float]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype}")


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


def _profile_one(
    *,
    fn,
    out_dir: str,
    tag: str,
    iters: int,
    warmup: int,
) -> None:
    _ensure_dir(out_dir)
    trace_path = os.path.join(out_dir, f"trace_{tag}.json")
    summary_path = os.path.join(out_dir, f"summary_{tag}.txt")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(iters):
            fn()
            torch.cuda.synchronize()

    prof.export_chrome_trace(trace_path)
    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=80)
    with open(summary_path, "w") as f:
        f.write(table)

    print(f"[bench_triton_fused_diffusion][profile] Wrote {trace_path}")
    print(f"[bench_triton_fused_diffusion][profile] Wrote {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda"])
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--steps_list", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="results/bench")
    parser.add_argument("--out_name", type=str, default="bench_triton_fused_diffusion")
    parser.add_argument("--profile", action="store_true", help="Also emit profiler traces.")
    parser.add_argument("--profile_seq_len", type=int, default=8192)
    parser.add_argument("--profile_hidden_size", type=int, default=256)
    parser.add_argument("--profile_steps", type=int, default=8)
    parser.add_argument("--profile_iters", type=int, default=10)
    parser.add_argument("--profile_warmup", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    dtype = _torch_dtype(args.dtype)
    _ensure_dir(args.out_dir)

    rows: List[BenchRow] = []

    for hidden_size in args.hidden_sizes:
        for seq_len in args.seq_lens:
            x = torch.randn(args.batch_size, hidden_size, seq_len, device="cuda", dtype=dtype)

            for steps in args.steps_list:
                if steps < 0:
                    continue

                def _fn_nonfused() -> torch.Tensor:
                    with record_function("cta_diffusion_nonfused"):
                        return diffusion_triton(x, alpha=args.alpha, steps=steps)

                def _fn_fused() -> torch.Tensor:
                    with record_function("cta_diffusion_fused"):
                        return diffusion_triton_fused(x, alpha=args.alpha, steps=steps)

                # Non-fused
                t_ms, mems = _bench_cuda(_fn_nonfused, iters=args.iters, warmup=args.warmup)
                t_mean, t_p50, t_p95 = _percentiles(t_ms)
                mem_mean = float(np.mean(np.asarray(mems, dtype=np.float64)))
                mem_p95 = float(np.percentile(np.asarray(mems, dtype=np.float64), 95))
                rows.append(
                    BenchRow(
                        benchmark="triton_fused_diffusion",
                        variant="non_fused",
                        device="cuda",
                        dtype=args.dtype,
                        steps=steps,
                        batch_size=args.batch_size,
                        hidden_size=hidden_size,
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

                # Fused (only supports {1,2,4,8} by design)
                try:
                    t_ms, mems = _bench_cuda(_fn_fused, iters=args.iters, warmup=args.warmup)
                except ValueError as e:
                    print(f"[bench_triton_fused_diffusion] skip fused steps={steps}: {e}")
                    continue
                t_mean, t_p50, t_p95 = _percentiles(t_ms)
                mem_mean = float(np.mean(np.asarray(mems, dtype=np.float64)))
                mem_p95 = float(np.percentile(np.asarray(mems, dtype=np.float64), 95))
                rows.append(
                    BenchRow(
                        benchmark="triton_fused_diffusion",
                        variant="fused",
                        device="cuda",
                        dtype=args.dtype,
                        steps=steps,
                        batch_size=args.batch_size,
                        hidden_size=hidden_size,
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
                    f"[bench_triton_fused_diffusion] steps={steps} "
                    f"B={args.batch_size} D={hidden_size} L={seq_len} dtype={args.dtype} "
                    f"non_fused_p50={rows[-2].latency_ms_p50:.3f}ms "
                    f"fused_p50={rows[-1].latency_ms_p50:.3f}ms"
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

    print(f"[bench_triton_fused_diffusion] Wrote {out_csv}")
    print(f"[bench_triton_fused_diffusion] Wrote {out_json}")

    if args.profile:
        x = torch.randn(args.batch_size, args.profile_hidden_size, args.profile_seq_len, device="cuda", dtype=dtype)
        steps = int(args.profile_steps)

        def _fn_nonfused_profile() -> torch.Tensor:
            with record_function("cta_diffusion_nonfused"):
                return diffusion_triton(x, alpha=args.alpha, steps=steps)

        def _fn_fused_profile() -> torch.Tensor:
            with record_function("cta_diffusion_fused"):
                return diffusion_triton_fused(x, alpha=args.alpha, steps=steps)

        tag_base = f"B{args.batch_size}_D{args.profile_hidden_size}_L{args.profile_seq_len}_s{steps}_{args.dtype}"
        prof_dir = os.path.join(args.out_dir, "profiling_triton_fused_diffusion")
        _profile_one(
            fn=_fn_nonfused_profile,
            out_dir=prof_dir,
            tag=f"non_fused_{tag_base}",
            iters=args.profile_iters,
            warmup=args.profile_warmup,
        )
        _profile_one(
            fn=_fn_fused_profile,
            out_dir=prof_dir,
            tag=f"fused_{tag_base}",
            iters=args.profile_iters,
            warmup=args.profile_warmup,
        )


if __name__ == "__main__":
    main()

