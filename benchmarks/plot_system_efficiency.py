"""
Plot System Efficiency curves from benchmark CSVs.

Expected input: CSV produced by `benchmarks/bench_block.py`
Outputs:
  - system_latency_vs_seqlen.png
  - system_mem_vs_seqlen.png
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _group_by_variant(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        out.setdefault(r["variant"], []).append(r)
    for v in out:
        out[v].sort(key=lambda x: int(x["seq_len"]))
    return out


def _extract_xy(rows: List[Dict[str, str]], y_key: str) -> Tuple[List[int], List[float]]:
    xs = [int(r["seq_len"]) for r in rows]
    ys = [float(r[y_key]) for r in rows]
    return xs, ys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="results/bench")
    parser.add_argument("--in_name", type=str, default="bench_block.csv")
    parser.add_argument("--out_dir", type=str, default="assets/images")
    args = parser.parse_args()

    in_csv = os.path.join(args.in_dir, args.in_name)
    rows = _read_rows(in_csv)
    grouped = _group_by_variant(rows)
    _ensure_dir(args.out_dir)

    # Latency plot (use p50 as default)
    plt.figure(figsize=(7, 4))
    for variant, vrows in grouped.items():
        xs, ys = _extract_xy(vrows, "latency_ms_p50")
        plt.plot(xs, ys, marker="o", label=variant)
    plt.xlabel("Sequence Length (L)")
    plt.ylabel("Latency p50 (ms)")
    plt.title("Prefill Latency vs Sequence Length")
    plt.xscale("log", base=2)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    out_latency = os.path.join(args.out_dir, "system_latency_vs_seqlen.png")
    plt.tight_layout()
    plt.savefig(out_latency, dpi=200)
    plt.close()

    # Memory plot (CUDA only; skip if missing)
    has_mem = any(r.get("peak_mem_bytes_p95") not in (None, "", "None") for r in rows)
    if has_mem:
        plt.figure(figsize=(7, 4))
        for variant, vrows in grouped.items():
            xs, ys = _extract_xy(vrows, "peak_mem_bytes_p95")
            # Convert bytes -> GiB for readability
            ys_gib = [y / (1024**3) for y in ys]
            plt.plot(xs, ys_gib, marker="o", label=variant)
        plt.xlabel("Sequence Length (L)")
        plt.ylabel("Peak memory p95 (GiB)")
        plt.title("Prefill Peak Memory vs Sequence Length")
        plt.xscale("log", base=2)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        plt.legend()
        out_mem = os.path.join(args.out_dir, "system_mem_vs_seqlen.png")
        plt.tight_layout()
        plt.savefig(out_mem, dpi=200)
        plt.close()
    else:
        print("[plot_system_efficiency] No CUDA peak memory data found; skipping memory plot.")

    print(f"[plot_system_efficiency] Wrote {out_latency}")
    if has_mem:
        print(f"[plot_system_efficiency] Wrote {out_mem}")


if __name__ == "__main__":
    main()

