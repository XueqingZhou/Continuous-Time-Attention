"""Experiment metadata utilities."""

from __future__ import annotations

from typing import Any, Dict, Optional

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import datasets
import torch
import transformers


def _get_repo_root() -> Path:
    """Resolve repository root based on this file location."""
    return Path(__file__).resolve().parents[2]


def _get_git_commit(repo_root: Path) -> Optional[str]:
    """Return current git commit hash if available."""
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return output.strip()
    except Exception:
        return None


def collect_metadata(
    command: str,
    config_path: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Collect environment and run metadata.

    Args:
        command: Full command line used to launch the run.
        config_path: Optional config path.
        extra: Optional extra fields to include.

    Returns:
        A metadata dictionary.
    """
    repo_root = _get_repo_root()
    meta: Dict[str, Any] = {
        "command": command,
        "config_path": config_path,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "datasets_version": datasets.__version__,
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "hf_home": os.environ.get("HF_HOME"),
        "cache_dir": os.environ.get("HF_DATASETS_CACHE"),
        "git_commit": _get_git_commit(repo_root),
    }
    if extra:
        meta.update(extra)
    return meta


def write_metadata(path: str, metadata: Dict[str, Any]) -> None:
    """Write metadata JSON to disk.

    Args:
        path: Output file path.
        metadata: Metadata dict.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
