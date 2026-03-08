"""
Plot vLLM baseline vs CTA curves from CSVs.

Outputs:
- <prefix>_prefill_latency.png
- <prefix>_e2e_latency.png
- <prefix>_decode_toks.png
- <prefix>_peak_mem.png
- <prefix>_speedup.png
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _read_csv(path: str) -> Dict[int, Dict[str, str]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out: Dict[int, Dict[str, str]] = {}
    for r in rows:
        out[int(r["prompt_len"])] = r
    return out


def _xy(
    rows: Dict[int, Dict[str, str]], key: str
) -> Tuple[List[int], List[float]]:
    xs = sorted(rows.keys())
    ys = [float(rows[x].get(key, "0") or 0) for x in xs]
    return xs, ys


def _plot_metric(
    *,
    base: Dict[int, Dict[str, str]],
    cta: Dict[int, Dict[str, str]],
    key: str,
    ylabel: str,
    out_path: str,
    transform=None,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    xs, ys = _xy(base, key)
    if transform is not None:
        ys = [transform(y) for y in ys]
    ax.plot(xs, ys, marker="o", label="baseline")
    xs, ys = _xy(cta, key)
    if transform is not None:
        ys = [transform(y) for y in ys]
    ax.plot(xs, ys, marker="o", label="cta")
    ax.set_xlabel("Prompt length")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_speedup(
    *,
    base: Dict[int, Dict[str, str]],
    cta: Dict[int, Dict[str, str]],
    out_path: str,
) -> None:
    xs = sorted(set(base.keys()) & set(cta.keys()))
    prefill = [
        float(base[x]["prefill_latency_ms_p50"]) / float(cta[x]["prefill_latency_ms_p50"])
        for x in xs
    ]
    e2e = [
        float(base[x]["latency_ms_p50"]) / float(cta[x]["latency_ms_p50"])
        for x in xs
    ]
    decode = [
        float(cta[x]["decode_tokens_per_s"]) / float(base[x]["decode_tokens_per_s"])
        for x in xs
    ]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(xs, prefill, marker="o", label="prefill speedup")
    ax.plot(xs, e2e, marker="o", label="e2e speedup")
    ax.plot(xs, decode, marker="o", label="decode speedup")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Prompt length")
    ax.set_ylabel("Speedup vs baseline (x)")
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--cta", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="assets/images")
    parser.add_argument("--out-prefix", type=str, default="vllm")
    args = parser.parse_args()

    base = _read_csv(args.baseline)
    cta = _read_csv(args.cta)

    os.makedirs(args.out_dir, exist_ok=True)

    out_latency = os.path.join(args.out_dir, f"{args.out_prefix}_prefill_latency.png")
    _plot_metric(
        base=base,
        cta=cta,
        key="prefill_latency_ms_p50",
        ylabel="Prefill latency p50 (ms)",
        out_path=out_latency,
    )

    out_e2e = os.path.join(args.out_dir, f"{args.out_prefix}_e2e_latency.png")
    _plot_metric(
        base=base,
        cta=cta,
        key="latency_ms_p50",
        ylabel="End-to-end latency p50 (ms)",
        out_path=out_e2e,
    )

    out_decode = os.path.join(args.out_dir, f"{args.out_prefix}_decode_toks.png")
    _plot_metric(
        base=base,
        cta=cta,
        key="decode_tokens_per_s",
        ylabel="Decode throughput (tokens/s)",
        out_path=out_decode,
    )

    out_mem = os.path.join(args.out_dir, f"{args.out_prefix}_peak_mem.png")
    _plot_metric(
        base=base,
        cta=cta,
        key="peak_mem_bytes_p95",
        ylabel="Peak mem p95 (GiB)",
        out_path=out_mem,
        transform=lambda y: y / (1024**3),
    )

    out_speedup = os.path.join(args.out_dir, f"{args.out_prefix}_speedup.png")
    _plot_speedup(base=base, cta=cta, out_path=out_speedup)

    print(f"[plot_vllm_bench] Wrote {out_latency}")
    print(f"[plot_vllm_bench] Wrote {out_e2e}")
    print(f"[plot_vllm_bench] Wrote {out_decode}")
    print(f"[plot_vllm_bench] Wrote {out_mem}")
    print(f"[plot_vllm_bench] Wrote {out_speedup}")


if __name__ == "__main__":
    main()
