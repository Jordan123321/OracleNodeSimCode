# scripts/run_test.py
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

# ------------------------------- Defaults -------------------------------
DEFAULT_YAML = "config/run_kicks.yaml"
DEFAULT_OUT = "results/run"

# ------------------------------- Robust imports --------------------------
try:
    # Preferred when invoked as: python -m scripts.run_test
    import src.oracletest as ot  # type: ignore
    from src.utils import ensure_outdir, load_yaml, timestamp_dir  # type: ignore
except Exception:
    # Fallback for Spyder/IPython "Run"
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(REPO_ROOT)
    import src.oracletest as ot  # type: ignore
    from src.utils import ensure_outdir, load_yaml, timestamp_dir  # type: ignore

# Compute repo root (parent of this script’s directory)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _coerce_cfg(d: Dict[str, Any]) -> Any:
    """
    Turn a plain dict into the config object OracleTest expects.
    Preference:
      1) A dataclass exported by src.oracletest named Config/TestConfig/OracleTestConfig
      2) A SimpleNamespace for attribute-style access
    """
    for name in ("Config", "TestConfig", "OracleTestConfig"):
        cls = getattr(ot, name, None)
        if cls is not None:
            try:
                return cls(**d)
            except TypeError:
                pass
    return SimpleNamespace(**d)


def _resolve_yaml_path(path_like: Optional[str]) -> Tuple[str, str]:
    """
    Resolve YAML path robustly. Returns (resolved_path, basis):
      - Tries absolute path first (if given)
      - Then tries CWD / path_like
      - Then tries REPO_ROOT / path_like
    Raises FileNotFoundError if not found.
    """
    if not path_like:
        raise FileNotFoundError("No YAML path provided.")

    p = os.path.expanduser(path_like)

    # Absolute
    if os.path.isabs(p) and os.path.isfile(p):
        return p, "absolute"

    # Relative to CWD
    cwd_rel = os.path.abspath(p)
    if os.path.isfile(cwd_rel):
        return cwd_rel, "cwd"

    # Relative to repo root
    root_rel = os.path.abspath(os.path.join(REPO_ROOT, p))
    if os.path.isfile(root_rel):
        return root_rel, "repo_root"

    raise FileNotFoundError(f"config file not found: {path_like}")


def _save_effective_config(cfg_dict: Dict[str, Any], out_dir: str) -> None:
    """Persist the effective config used for the run to out_dir/config_used.yaml."""
    try:
        import yaml  # type: ignore
        with open(os.path.join(out_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg_dict, f, sort_keys=True)
    except Exception:
        # YAML not available or write failed; skip silently.
        pass


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run a single OracleTest (loads YAML, writes to a timestamped folder).",
        add_help=True,
    )
    p.add_argument("--yaml", type=str, default=None, help="Path to YAML config file (relative or absolute).")
    p.add_argument("--out", type=str, default=DEFAULT_OUT, help="Base output directory (a timestamped subdir will be created).")
    p.add_argument("--seed", type=int, default=None, help="Override PRNG seed.")
    p.add_argument("--no-plots", action="store_true", help="Disable plot generation.")
    p.add_argument("--no-csv", action="store_true", help="Disable CSV output.")
    # Spyder sometimes injects extra args; ignore them
    args, _unknown = p.parse_known_args()

    # Resolve YAML: CLI -> ENV -> default, then path resolution (abs, CWD, repo root)
    yaml_hint = args.yaml or os.environ.get("ORACLE_RUN_YAML") or DEFAULT_YAML
    try:
        yaml_path, basis = _resolve_yaml_path(yaml_hint)
    except FileNotFoundError as e:
        print(f"[error] failed to load config '{yaml_hint}': {e}")
        print("        Try an absolute path or ensure you run from the project root.")
        print("        Example: python -m scripts.run_test --yaml config/run_kicks.yaml")
        sys.exit(2)

    # Load YAML (supports flat dict or top-level key "test")
    try:
        raw: Dict[str, Any] = load_yaml(yaml_path)
    except Exception as e:
        print(f"[error] failed to parse YAML '{yaml_path}': {e}")
        sys.exit(2)

    cfg_dict: Dict[str, Any] = raw.get("test", raw)

    # Prepare output dir
    out_dir = timestamp_dir(args.out)
    ensure_outdir(out_dir)

    # Apply CLI overrides
    cfg_dict["out_dir"] = out_dir
    cfg_dict["save_plots"] = not args.no_plots
    cfg_dict["save_csv"] = not args.no_csv
    if args.seed is not None:
        cfg_dict["seed"] = int(args.seed)

    # Persist effective config
    _save_effective_config(cfg_dict, out_dir)

    # Build test object and run
    cfg_obj = _coerce_cfg(cfg_dict)
    test = ot.OracleTest(cfg_obj)
    res = test.run()

    # Support both dataclass result and a plain object with attributes
    if hasattr(res, "__dataclass_fields__"):
        summary = {k: getattr(res, k) for k in res.__dataclass_fields__}  # type: ignore
    else:
        summary = {
            "draws": getattr(res, "draws", None),
            "successes": getattr(res, "successes", None),
            "failures": getattr(res, "failures", None),
            "reached_target": getattr(res, "reached_target", None),
            "false_positives": getattr(res, "false_positives", None),
            "false_negatives": getattr(res, "false_negatives", None),
            "incompetent_removed": getattr(res, "incompetent_removed", None),
        }

    print(
        f"[cfg] loaded from ({basis}): {yaml_path}\n"
        f"draws={summary['draws']}, succ={summary['successes']}, fail={summary['failures']}, "
        f"reached_target={summary['reached_target']}, FP={summary['false_positives']}, "
        f"FN={summary['false_negatives']}, incompetent_removed={summary['incompetent_removed']}\n"
        f"[out] {out_dir}"
    )


if __name__ == "__main__":
    main()