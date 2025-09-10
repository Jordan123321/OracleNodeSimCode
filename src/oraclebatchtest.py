# src/oraclebatchtest.py
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ---- Robust imports so it works both as a module and a script
try:  # package-style
    from .oracletest import OracleTest  # type: ignore
    try:
        from .oracletest import TestConfig  # type: ignore
    except Exception:
        TestConfig = None  # type: ignore
    from .plotting import (
        ensure_outdir,
        plot_box,
        plot_ecdf,
        plot_hist,
    )  # type: ignore
except Exception:  # script-style
    sys.path.append(os.path.dirname(__file__))
    from oracletest import OracleTest  # type: ignore
    try:
        from oracletest import TestConfig  # type: ignore
    except Exception:
        TestConfig = None  # type: ignore
    from plotting import ensure_outdir, plot_box, plot_ecdf, plot_hist  # type: ignore

try:
    import yaml  # pyyaml
except Exception:
    yaml = None  # handled below


def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    if yaml is None:
        raise RuntimeError("pyyaml not installed. `pip install pyyaml`")
    if not os.path.exists(path):
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML must parse into a mapping/object at the top level.")
    return data


def _make_test_config(raw: Dict[str, Any]):
    """
    Return a config acceptable by OracleTest:
      - If TestConfig dataclass is available, instantiate it.
      - Otherwise, return the plain dict and rely on OracleTest to handle it.
    """
    if TestConfig is None:
        return raw
    # Filter unexpected keys (keeps it tolerant)
    valid_keys = {f.name for f in TestConfig.__dataclass_fields__.values()}  # type: ignore
    filtered = {k: v for k, v in raw.items() if k in valid_keys}
    return TestConfig(**filtered)  # type: ignore


def _timestamp_dir(base: str) -> str:
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(base, ts)


def run_batch(
    base_cfg: Dict[str, Any],
    runs: int,
    out_dir: str,
    base_seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run `runs` independent OracleTest experiments.
    Returns:
        (summary_df, meta) where summary_df has one row per run.
    """
    ensure_outdir(out_dir)

    # Prepare a clean copy of the single-run config; we’ll override seed per replicate
    single_cfg = dict(base_cfg)

    rows = []
    t0 = time.time()
    for r in range(runs):
        seed = (base_seed + r) if base_seed is not None else np.random.randint(0, 2**31 - 1)
        single_cfg_run = dict(single_cfg)
        single_cfg_run["seed"] = int(seed)
        # avoid run-by-run plot spam & heavy I/O; the batch will plot aggregated results
        single_cfg_run.setdefault("save_plots", False)
        single_cfg_run.setdefault("save_csv", False)

        cfg_obj = _make_test_config(single_cfg_run)
        test = OracleTest(cfg_obj)
        res = test.run()

        # Flexible extraction in case res is a dataclass or a simple namespace/dict
        if hasattr(res, "__dataclass_fields__"):
            res_d = asdict(res)  # type: ignore
        elif isinstance(res, dict):
            res_d = dict(res)
        else:
            res_d = res.__dict__  # type: ignore

        row = {
            "run": r + 1,
            "seed": seed,
            # Core outcomes
            "draws": int(res_d.get("draws", np.nan)),
            "reached_target": bool(res_d.get("reached_target", False)),
            "successes": int(res_d.get("successes", np.nan)),
            "failures": int(res_d.get("failures", np.nan)),
            "wilson_lb_last": float(res_d.get("wilson_lb_last", np.nan)),
            # Node quality metrics
            "false_positives": int(res_d.get("false_positives", np.nan)),
            "false_negatives": int(res_d.get("false_negatives", np.nan)),
            "incompetent_removed": int(res_d.get("incompetent_removed", np.nan)),
            "active_nodes_end": int(res_d.get("active_nodes_end", np.nan)),
            # Timing if you later add it to TestResult
            "elapsed_sec": float(res_d.get("elapsed_sec", np.nan)),
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    meta = {
        "runs": runs,
        "base_seed": base_seed,
        "elapsed_total_sec": time.time() - t0,
    }
    return summary, meta


def _write_outputs(
    df: pd.DataFrame,
    meta: Dict[str, Any],
    out_dir: str,
    do_plots: bool,
) -> None:
    # CSV
    df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    pd.Series(meta).to_csv(os.path.join(out_dir, "meta.csv"))

    # Plots (basic but useful for a response letter)
    if do_plots:
        # Distribution of draws to target
        plot_hist(
            df=df,
            col="draws",
            bins=30,
            title="Distribution of Draws to Target",
            xlabel="Draws",
            out_base=os.path.join(out_dir, "draws_hist"),
        )
        plot_ecdf(
            df=df,
            col="draws",
            title="ECDF of Draws to Target",
            xlabel="Draws",
            out_base=os.path.join(out_dir, "draws_ecdf"),
        )
        plot_box(
            df=df.assign(dummy="All"),
            x="dummy",
            y="draws",
            title="Draws to Target (Boxplot)",
            xlabel="",
            ylabel="Draws",
            out_base=os.path.join(out_dir, "draws_box"),
        )

        # FP/FN distributions
        plot_hist(
            df=df,
            col="false_positives",
            bins=20,
            title="False Positives (Distribution)",
            xlabel="# False Positives",
            out_base=os.path.join(out_dir, "fp_hist"),
        )
        plot_hist(
            df=df,
            col="false_negatives",
            bins=20,
            title="False Negatives (Distribution)",
            xlabel="# False Negatives",
            out_base=os.path.join(out_dir, "fn_hist"),
        )
        # Active nodes at end
        plot_hist(
            df=df,
            col="active_nodes_end",
            bins=20,
            title="Active Nodes at End (Distribution)",
            xlabel="Active Nodes",
            out_base=os.path.join(out_dir, "active_nodes_end_hist"),
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run many OracleTest replicates and aggregate results."
    )
    p.add_argument(
        "--yaml",
        type=str,
        default=None,
        help="YAML config for a single run (node mix, thresholds, etc.).",
    )
    p.add_argument("--runs", type=int, default=100, help="Number of replicates.")
    p.add_argument("--seed", type=int, default=None, help="Base seed (optional).")
    p.add_argument(
        "--out",
        type=str,
        default="results/batch",
        help="Output directory for the batch. A timestamped subdir is created.",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip aggregated plots (still writes summary.csv).",
    )
    args = p.parse_args()

    raw_cfg = _load_yaml(args.yaml)
    # If your YAML has nested structure like {test: {...}}, allow that too:
    test_cfg = raw_cfg.get("test", raw_cfg)

    out_dir = _timestamp_dir(args.out)
    ensure_outdir(out_dir)

    print("[batch] config (single-run) keys:", ", ".join(test_cfg.keys()))
    print(f"[batch] runs={args.runs}, base_seed={args.seed}, out_dir={out_dir}")

    df, meta = run_batch(
        base_cfg=test_cfg,
        runs=args.runs,
        out_dir=out_dir,
        base_seed=args.seed,
    )

    # Quick console summary
    reached_rate = df["reached_target"].mean() if "reached_target" in df else float("nan")
    print(
        f"[batch done] n={len(df)}, reached_target_rate={reached_rate:.3f}, "
        f"draws(mean/median)={df['draws'].mean():.1f}/{df['draws'].median():.1f}"
    )

    _write_outputs(df, meta, out_dir, do_plots=not args.no_plots)


if __name__ == "__main__":
    main()