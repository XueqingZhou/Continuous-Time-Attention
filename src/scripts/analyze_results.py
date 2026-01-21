"""Analyze experiment outputs and generate summary plots.

This script reads JSON artifacts produced by:
- src/experiments/run_classification.py
- src/experiments/run_char_level.py
- src/experiments/run_language_modeling.py
- src/experiments/run_ablation_*.py

and writes a compact, paper-style summary and plots under results/summary/.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import matplotlib.pyplot as plt


JsonDict = Dict[str, Any]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> Optional[JsonDict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


@dataclass(frozen=True)
class ClassificationRow:
    dataset: str
    standard_best: float
    pde_best: float

    @property
    def delta_abs_points(self) -> float:
        return (self.pde_best - self.standard_best) * 100.0


@dataclass(frozen=True)
class LmRow:
    standard_best_ppl: float
    pde_best_ppl: float

    @property
    def rel_improvement_percent(self) -> float:
        # Positive means PDE is better (lower PPL).
        if self.standard_best_ppl == 0:
            return 0.0
        return (self.standard_best_ppl - self.pde_best_ppl) / self.standard_best_ppl * 100.0


def _extract_classification_row(dataset: str, data: Mapping[str, Any]) -> Optional[ClassificationRow]:
    std = data.get("standard", {})
    pde = data.get("pde", {})
    std_best = _safe_float(std.get("best_accuracy"))
    pde_best = _safe_float(pde.get("best_accuracy"))
    if std_best is None or pde_best is None:
        return None
    return ClassificationRow(dataset=dataset, standard_best=std_best, pde_best=pde_best)


def _extract_lm_row(data: Mapping[str, Any]) -> Optional[LmRow]:
    std = data.get("standard", {})
    pde = data.get("pde", {})
    std_best = _safe_float(std.get("best_perplexity"))
    pde_best = _safe_float(pde.get("best_perplexity"))
    if std_best is None or pde_best is None:
        return None
    return LmRow(standard_best_ppl=std_best, pde_best_ppl=pde_best)


def _plot_classification_bars(rows: List[ClassificationRow], out_png: Path) -> None:
    if not rows:
        return
    labels = [r.dataset for r in rows]
    std_vals = [r.standard_best * 100.0 for r in rows]
    pde_vals = [r.pde_best * 100.0 for r in rows]

    x = list(range(len(rows)))
    width = 0.38

    plt.figure(figsize=(10, 5))
    plt.bar([i - width / 2 for i in x], std_vals, width=width, label="Standard")
    plt.bar([i + width / 2 for i in x], pde_vals, width=width, label="PDE")
    plt.xticks(x, labels)
    plt.ylabel("Best Accuracy (%)")
    plt.title("Classification: Best Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _plot_lm_curve(
    lm_data: Mapping[str, Any],
    out_png: Path,
) -> None:
    std_hist = (lm_data.get("standard", {}) or {}).get("history", {}) or {}
    pde_hist = (lm_data.get("pde", {}) or {}).get("history", {}) or {}
    std = std_hist.get("val_ppl", [])
    pde = pde_hist.get("val_ppl", [])
    if not std and not pde:
        return

    plt.figure(figsize=(9, 5))
    if std:
        plt.plot(list(range(1, len(std) + 1)), std, marker="o", label="Standard")
    if pde:
        plt.plot(list(range(1, len(pde) + 1)), pde, marker="s", label="PDE")
    plt.xlabel("Epoch")
    plt.ylabel("Validation PPL")
    plt.title("WikiText-103: Validation Perplexity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _write_markdown_summary(
    out_md: Path,
    cls_rows: List[ClassificationRow],
    char_row: Optional[ClassificationRow],
    lm_row: Optional[LmRow],
    missing: List[str],
) -> None:
    lines: List[str] = []
    lines.append("# Results Summary\n")

    if missing:
        lines.append("## Missing artifacts\n")
        for m in missing:
            lines.append(f"- {m}\n")
        lines.append("\n")

    if cls_rows:
        lines.append("## Table 1: Classification\n\n")
        lines.append("| Dataset | Standard (best acc) | PDE (best acc) | Δ (abs, points) |\n")
        lines.append("|---|---:|---:|---:|\n")
        for r in cls_rows:
            lines.append(
                f"| {r.dataset} | {r.standard_best:.4f} | {r.pde_best:.4f} | {r.delta_abs_points:+.2f} |\n"
            )
        lines.append("\n")

    if char_row is not None:
        lines.append("## Table 2: Character-level IMDb\n\n")
        r = char_row
        lines.append("| Task | Standard (best acc) | PDE (best acc) | Δ (abs, points) |\n")
        lines.append("|---|---:|---:|---:|\n")
        lines.append(
            f"| char_level_imdb | {r.standard_best:.4f} | {r.pde_best:.4f} | {r.delta_abs_points:+.2f} |\n"
        )
        lines.append("\n")

    if lm_row is not None:
        lines.append("## Tables 3–5: WikiText-103 LM\n\n")
        lines.append("| Model | Best PPL (↓) |\n")
        lines.append("|---|---:|\n")
        lines.append(f"| Standard | {lm_row.standard_best_ppl:.2f} |\n")
        lines.append(f"| PDE | {lm_row.pde_best_ppl:.2f} |\n")
        lines.append(f"\nRelative improvement: **{lm_row.rel_improvement_percent:.2f}%**\n\n")

    out_md.write_text("".join(lines), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze results JSONs and plot summaries")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing *_results.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/summary",
        help="Directory to write summary artifacts",
    )
    args = parser.parse_args(argv)

    repo = _repo_root()
    results_dir = (repo / args.results_dir).resolve()
    out_dir = (repo / args.output_dir).resolve()
    _mkdir(out_dir)

    missing: List[str] = []

    # Classification (Table 1)
    cls_rows: List[ClassificationRow] = []
    for name in ["imdb", "ag_news", "sst2"]:
        path = results_dir / f"{name}_results.json"
        data = _read_json(path)
        if data is None:
            missing.append(str(path.relative_to(repo)))
            continue
        row = _extract_classification_row(name, data)
        if row is None:
            missing.append(f"{path.relative_to(repo)} (missing keys)")
            continue
        cls_rows.append(row)

    # Character-level (Table 2)
    char_path = results_dir / "char_level_imdb_results.json"
    char_data = _read_json(char_path)
    char_row = None if char_data is None else _extract_classification_row("char_level_imdb", char_data)
    if char_data is None:
        missing.append(str(char_path.relative_to(repo)))
    elif char_row is None:
        missing.append(f"{char_path.relative_to(repo)} (missing keys)")

    # LM (Tables 3-5)
    lm_path = results_dir / "wikitext103_results.json"
    lm_data = _read_json(lm_path)
    lm_row = None if lm_data is None else _extract_lm_row(lm_data)
    if lm_data is None:
        missing.append(str(lm_path.relative_to(repo)))
    elif lm_row is None:
        missing.append(f"{lm_path.relative_to(repo)} (missing keys)")

    # Plots
    _plot_classification_bars(cls_rows, out_dir / "classification_best_accuracy.png")
    if lm_data is not None:
        _plot_lm_curve(lm_data, out_dir / "wikitext103_val_ppl.png")

    # Markdown summary
    _write_markdown_summary(
        out_md=out_dir / "summary.md",
        cls_rows=cls_rows,
        char_row=char_row,
        lm_row=lm_row,
        missing=missing,
    )

    print(f"Wrote summary to {out_dir}/summary.md")


if __name__ == "__main__":
    main()

