"""Configuration utilities for experiment scripts."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Tuple

import argparse
import yaml


ConfigDict = Dict[str, Any]


def load_yaml_config(path: str) -> ConfigDict:
    """Load a YAML configuration file.

    Args:
        path: Path to YAML file.

    Returns:
        Parsed configuration dictionary.
    """
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def get_nested(config: Mapping[str, Any], keys: Iterable[str]) -> Any:
    """Get a nested value from a dict using a sequence of keys.

    Args:
        config: Mapping to search.
        keys: Key path.

    Returns:
        The nested value if found, otherwise None.
    """
    current: Any = config
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def set_defaults_from_config(
    parser: argparse.ArgumentParser,
    config: Mapping[str, Any],
    mapping: Mapping[str, Tuple[str, ...]],
) -> None:
    """Set argparse defaults from a nested config dict.

    Args:
        parser: Argument parser to update.
        config: Loaded configuration dictionary.
        mapping: Mapping from argparse dest -> nested key path.
    """
    defaults: Dict[str, Any] = {}
    for dest, keys in mapping.items():
        value = get_nested(config, keys)
        if value is not None:
            defaults[dest] = value
    if defaults:
        parser.set_defaults(**defaults)
