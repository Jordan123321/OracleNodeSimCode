# scripts/run_batch.py
from __future__ import annotations

import argparse
import copy
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ------------------------------- Defaults -------------------------------
DEFAULT_YAML = "config/batch.yaml"      # You can point this to your single-run YAML too
DEFAULT_OUT  = "results/batch"
DEFAULT_RUNS = 100

# ------------------------------- Robust imports --------------------------
try:
    import src.oracletest as ot  # type: ignore
    from src.utils import ensure_outdir, load_yaml, timestamp_dir  # type: ignore
    from src.plotting import plot_hist, plot_ecdf, plot_box  # type: ignore
except Exception:
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(REPO_ROOT)
    import src.oracletest as ot  # type: ignore
    from src.utils import ensure_outdir, load_yaml, timestamp_dir  # type: ignore
    from src.plotting import plot_hist, plot_ecdf, plot_box  # type: ignore

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ------------------------------- Helpers ---------------------------------
def _coerce_cfg(d: Dict[str, Any]) -> Any:
    """
    Turn a plain dict into the config object OracleTest expects.
    Tries dataclasses 'Config'/'TestConfig'/'OracleTestConfig', else SimpleNamespace.
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
    Resolve YAML path robustly. Returns (resolved_path, basis).
    Search order: absolute -> CWD -> repo root.
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


def _save_effective_config(cfg_dict: Dict[str, Any], out_dir: str, name: str = "config_used.yaml") -> None:
    """Persist the effective config to out_dir/name (best-effort)."""
    try:
        import yaml  # type: ignore
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg_dict, f, sort_keys=True)
    except Exception:
        pass


# ------------------------------- Main ------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Run multiple OracleTest simulations and aggregate results.",
        add_help=True,
    )
    p.add_argument("--yaml", type=str, default=None, help="Path to YAML config (relative/absolute).")
    p.add_argument("--out", type=str, default=DEFAULT_OUT, help="Base output directory.")
    p.add_argument("--runs", type=int, default=None, help=f"Number of independent runs (default {DEFAULT_RUNS}).")
    p.add_argument("--seed", type=int, default=None, help="Base PRNG seed override (each run offsets by +i).")
    p.add_argument("--no-per-run-plots", action="store_true", help="Disable saving per-run plots to speed up.")
    p.add_argument("--no-per-run-csv", action="store_true", help="Disable saving per-run CSV traces.")
    args, _unknown = p.parse_known_args()

    # Resolve YAML path
    yaml_hint = args.yaml or os.environ.get("ORACLE_BATCH_YAML") or DEFAULT_YAML
    try:
        yaml_path, basis = _resolve_yaml_path(yaml_hint)
    except FileNotFoundError as e:
        print(f"[error] failed to load config '{yaml_hint}': {e}")
        print("        Try an absolute path or run from repo root:")
        print("        python -m scripts.run_batch --yaml config/batch.yaml")
        sys.exit(2)

    # Load YAML (accept either a flat dict or top-level key 'test'/'batch')
    try:
        raw: Dict[str, Any] = load_yaml(yaml_path)
    except Exception as e:
        print(f"[error] failed to parse YAML '{yaml_path}': {e}")
        sys.exit(2)

    # Flatten config
    cfg: Dict[str, Any] = raw.get("batch", raw.get("test", raw))

    # Determine how many runs
    num_runs = int(args.runs) if args.runs is not None else int(cfg.get("num_runs", DEFAULT_RUNS))

    # Prepare parent output folder
    out_parent = timestamp_dir(args.out)
    ensure_outdir(out_parent)

    # Store top-level batch settings used
    effective_batch = {
        "yaml_source": yaml_path,
        "basis": basis,
        "num_runs": num_runs,
        "base_seed": args.seed if args.seed is not None else cfg.get("seed", 42),
        "seed_stride": cfg.get("seed_stride", 1),
        "per_run_plots": not args.no_per_run_plots and bool(cfg.get("save_plots", True)),
        "per_run_csv": not args.no_per_run_csv and bool(cfg.get("save_csv", True)),
    }
    _save_effective_config(effective_batch, out_parent, "batch_used.yaml")

    # Aggregate results
    summaries: List[Dict[str, Any]] = []

    base_seed = int(effective_batch["base_seed"])
    seed_stride = int(effective_batch["seed_stride"])

    for i in range(num_runs):
        run_cfg = copy.deepcopy(cfg)

        # Per-run output directory
        run_dir = os.path.join(out_parent, f"run_{i:04d}")
        ensure_outdir(run_dir)

        # Controls: seed, plotting, CSV
        run_cfg["seed"] = base_seed + i * seed_stride
        run_cfg["out_dir"] = run_dir
        run_cfg["save_plots"] = bool(effective_batch["per_run_plots"])
        run_cfg["save_csv"] = bool(effective_batch["per_run_csv"])

        # Persist per-run config
        _save_effective_config(run_cfg, run_dir, "config_used.yaml")

        # Build test and run
        cfg_obj = _coerce_cfg(run_cfg)
        test = ot.OracleTest(cfg_obj)
        res = test.run()

        # Normalize to a dict
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

        summary["run_idx"] = i
        summary["seed"] = run_cfg["seed"]
        summaries.append(summary)

        # Console ping every so often
        if (i + 1) % max(1, num_runs // 10) == 0:
            print(f"[batch] completed {i+1}/{num_runs} runs...")

    # Save aggregated results
    df = pd.DataFrame(summaries)
    df.to_csv(os.path.join(out_parent, "batch_summary.csv"), index=False)

    # Aggregated plots
    try:
        # draws distribution
        plot_hist(
            df=df,
            col="draws",
            bins=30,
            title="Distribution of Draws (Batch)",
            xlabel="Number of draws until target",
            out_base=os.path.join(out_parent, "draws_hist"),
        )
        plot_ecdf(
            df=df,
            col="draws",
            title="ECDF of Draws (Batch)",
            xlabel="Number of draws until target",
            out_base=os.path.join(out_parent, "draws_ecdf"),
        )
        # boxplots for FP/FN/incompetent_removed
        for metric, title in [
            ("false_positives", "False Positives per Run"),
            ("false_negatives", "Residual Unreliable Nodes per Run"),
            ("incompetent_removed", "Incompetent Nodes Removed per Run"),
        ]:
            plot_box(
                df=df,
                x=None if "x" not in df.columns else "x",  # noop, we just pass y
                y=metric,
                title=title,
                xlabel="",
                ylabel=metric.replace("_", " ").title(),
                out_base=os.path.join(out_parent, f"{metric}_box"),
            )
    except Exception as e:
        print(f"[warn] plotting failed: {e}")

    # Print quick stats
    med = df["draws"].median()
    p25 = df["draws"].quantile(0.25)
    p75 = df["draws"].quantile(0.75)
    reached = int(df["reached_target"].sum())
    print(
        f"\n[batch summary] runs={num_runs}, reached_target={reached}/{num_runs}\n"
        f"draws median={med:.1f} (IQR {p25:.1f}–{p75:.1f})\n"
        f"[out] {out_parent}\n"
    )


if __name__ == "__main__":
    main()