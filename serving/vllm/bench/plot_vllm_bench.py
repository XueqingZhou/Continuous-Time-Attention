"""
Plot vLLM baseline vs CTA curves from CSVs.

Outputs:
- vllm_prefill_latency_vs_prompt.png
- vllm_peak_mem_vs_prompt.png
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

    # Prefill latency p50
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    xs, ys = _xy(base, "prefill_latency_ms_p50")
    ax.plot(xs, ys, marker="o", label="baseline")
    xs, ys = _xy(cta, "prefill_latency_ms_p50")
    ax.plot(xs, ys, marker="o", label="cta")
    ax.set_xlabel("Prompt length")
    ax.set_ylabel("Prefill latency p50 (ms)")
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    out_latency = os.path.join(args.out_dir, f"{args.out_prefix}_prefill_latency.png")
    plt.tight_layout()
    plt.savefig(out_latency, dpi=200)
    plt.close(fig)

    # Peak mem p95 (GiB)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    xs, ys = _xy(base, "peak_mem_bytes_p95")
    ax.plot(xs, [y / (1024**3) for y in ys], marker="o", label="baseline")
    xs, ys = _xy(cta, "peak_mem_bytes_p95")
    ax.plot(xs, [y / (1024**3) for y in ys], marker="o", label="cta")
    ax.set_xlabel("Prompt length")
    ax.set_ylabel("Peak mem p95 (GiB)")
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    out_mem = os.path.join(args.out_dir, f"{args.out_prefix}_peak_mem.png")
    plt.tight_layout()
    plt.savefig(out_mem, dpi=200)
    plt.close(fig)

    print(f"[plot_vllm_bench] Wrote {out_latency}")
    print(f"[plot_vllm_bench] Wrote {out_mem}")


if __name__ == "__main__":
    main()
