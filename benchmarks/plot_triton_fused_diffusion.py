"""
Plot Triton fused diffusion curves from CSV.

Input: CSV produced by `benchmarks/bench_triton_fused_diffusion.py`
Outputs (default under assets/images/):
  - triton_fused_diffusion_latency.png
  - triton_fused_diffusion_peak_mem.png
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _group(rows: List[Dict[str, str]]) -> Dict[Tuple[int, str, int], List[Dict[str, str]]]:
    # key: (steps, variant, hidden_size)
    out: Dict[Tuple[int, str, int], List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        out[(int(r["steps"]), r["variant"], int(r["hidden_size"]))].append(r)
    for k in out:
        out[k].sort(key=lambda x: int(x["seq_len"]))
    return out


def _extract_xy(rows: List[Dict[str, str]], key: str) -> Tuple[List[int], List[float]]:
    xs = [int(r["seq_len"]) for r in rows]
    ys = [float(r[key]) for r in rows]
    return xs, ys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="results/bench")
    parser.add_argument("--in_name", type=str, default="bench_triton_fused_diffusion.csv")
    parser.add_argument("--out_dir", type=str, default="assets/images")
    parser.add_argument("--out_prefix", type=str, default="triton_fused_diffusion")
    parser.add_argument("--hidden_size", type=int, default=256, help="Select one hidden size to plot.")
    args = parser.parse_args()

    in_csv = os.path.join(args.in_dir, args.in_name)
    rows = _read_rows(in_csv)
    grouped = _group(rows)
    _ensure_dir(args.out_dir)

    # Determine which steps exist
    steps_list = sorted({int(r["steps"]) for r in rows})

    # Latency plot (p50)
    fig, axes = plt.subplots(1, len(steps_list), figsize=(5 * len(steps_list), 4), sharey=False)
    if len(steps_list) == 1:
        axes = [axes]

    for ax, steps in zip(axes, steps_list):
        for variant in ("non_fused", "fused"):
            key = (steps, variant, args.hidden_size)
            if key not in grouped:
                continue
            xs, ys = _extract_xy(grouped[key], "latency_ms_p50")
            ax.plot(xs, ys, marker="o", label=variant)
        ax.set_title(f"steps={steps}, D={args.hidden_size}")
        ax.set_xlabel("Sequence Length (L)")
        ax.set_ylabel("Latency p50 (ms)")
        ax.set_xscale("log", base=2)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.legend()

    out_latency = os.path.join(args.out_dir, f"{args.out_prefix}_latency.png")
    plt.tight_layout()
    plt.savefig(out_latency, dpi=200)
    plt.close(fig)

    # Peak memory plot (p95, GiB)
    fig, axes = plt.subplots(1, len(steps_list), figsize=(5 * len(steps_list), 4), sharey=False)
    if len(steps_list) == 1:
        axes = [axes]

    for ax, steps in zip(axes, steps_list):
        for variant in ("non_fused", "fused"):
            key = (steps, variant, args.hidden_size)
            if key not in grouped:
                continue
            xs, ys = _extract_xy(grouped[key], "peak_mem_bytes_p95")
            ys_gib = [y / (1024**3) for y in ys]
            ax.plot(xs, ys_gib, marker="o", label=variant)
        ax.set_title(f"steps={steps}, D={args.hidden_size}")
        ax.set_xlabel("Sequence Length (L)")
        ax.set_ylabel("Peak mem p95 (GiB)")
        ax.set_xscale("log", base=2)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.legend()

    out_mem = os.path.join(args.out_dir, f"{args.out_prefix}_peak_mem.png")
    plt.tight_layout()
    plt.savefig(out_mem, dpi=200)
    plt.close(fig)

    print(f"[plot_triton_fused_diffusion] Wrote {out_latency}")
    print(f"[plot_triton_fused_diffusion] Wrote {out_mem}")


if __name__ == "__main__":
    main()

