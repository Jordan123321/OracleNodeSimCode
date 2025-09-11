# src/utils.py
from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import numpy as np

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # Loaded only if available


def ensure_outdir(path: str) -> None:
    """Create path if missing (idempotent)."""
    os.makedirs(path, exist_ok=True)


def timestamp_dir(base: str) -> str:
    """Return base/<YYYY-mm-dd_HH-MM-SS> (does not create)."""
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(base, ts)


def set_seed(seed: Optional[int]) -> int:
    """
    Set numpy RNG seed. If None, sample a fresh seed and use that.
    Returns the concrete seed used (for logging/repro).
    """
    used = seed if seed is not None else int(np.random.randint(0, 2**31 - 1))
    np.random.seed(used)
    return used


def load_yaml(path: Optional[str]) -> Dict[str, Any]:
    """
    Load a YAML mapping. Returns {} if path is None.
    Raises if the file doesn't exist or top-level is not a mapping.
    """
    if path is None:
        return {}
    if yaml is None:
        raise RuntimeError("pyyaml not installed. `pip install pyyaml`")
    if not os.path.exists(path):
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML must parse into a top-level mapping/object.")
    return data