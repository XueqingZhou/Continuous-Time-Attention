"""
Compare vLLM baseline vs CTA runs and emit summary tables.

Inputs: two CSVs produced by `bench_vllm.py`.
Outputs:
- compare_vllm_bench.csv
- compare_vllm_bench.md
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_csv(path: str) -> Dict[int, Dict[str, str]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out: Dict[int, Dict[str, str]] = {}
    for r in rows:
        out[int(r["prompt_len"])] = r
    return out


def _f(val: str) -> float:
    return float(val) if val is not None and val != "" else 0.0


def _ratio(a: float, b: float) -> float:
    return (a / b) if b > 0 else 0.0


def _delta(a: float, b: float) -> float:
    return a - b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--cta", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="results/serving/vllm")
    parser.add_argument("--out-name", type=str, default="compare_vllm_bench")
    args = parser.parse_args()

    base = _read_csv(args.baseline)
    cta = _read_csv(args.cta)
    prompt_lens = sorted(set(base.keys()) & set(cta.keys()))
    if not prompt_lens:
        raise ValueError("no overlapping prompt_len between baseline and CTA")

    rows: List[Dict[str, float]] = []
    for L in prompt_lens:
        b = base[L]
        c = cta[L]

        b_lat = _f(b["latency_ms_p50"])
        c_lat = _f(c["latency_ms_p50"])
        b_pre = _f(b.get("prefill_latency_ms_p50", "0"))
        c_pre = _f(c.get("prefill_latency_ms_p50", "0"))
        b_tok = _f(b.get("decode_tokens_per_s", "0"))
        c_tok = _f(c.get("decode_tokens_per_s", "0"))
        b_mem = _f(b.get("peak_mem_bytes_p95", "0"))
        c_mem = _f(c.get("peak_mem_bytes_p95", "0"))

        rows.append(
            {
                "prompt_len": L,
                "baseline_latency_p50_ms": b_lat,
                "cta_latency_p50_ms": c_lat,
                "latency_p50_speedup": _ratio(b_lat, c_lat),
                "baseline_prefill_p50_ms": b_pre,
                "cta_prefill_p50_ms": c_pre,
                "prefill_p50_speedup": _ratio(b_pre, c_pre),
                "baseline_decode_toks_s": b_tok,
                "cta_decode_toks_s": c_tok,
                "decode_speedup": _ratio(c_tok, b_tok),
                "baseline_peak_mem_p95": b_mem,
                "cta_peak_mem_p95": c_mem,
                "peak_mem_delta_bytes": _delta(c_mem, b_mem),
                "peak_mem_ratio": _ratio(c_mem, b_mem),
            }
        )

    _ensure_dir(args.out_dir)
    out_csv = os.path.join(args.out_dir, f"{args.out_name}.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    out_md = os.path.join(args.out_dir, f"{args.out_name}.md")
    with open(out_md, "w") as f:
        headers = list(rows[0].keys())
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for r in rows:
            f.write("| " + " | ".join(f"{r[h]:.4g}" for h in headers) + " |\n")

    print(f"[compare_vllm_bench] Wrote {out_csv}")
    print(f"[compare_vllm_bench] Wrote {out_md}")


if __name__ == "__main__":
    main()
