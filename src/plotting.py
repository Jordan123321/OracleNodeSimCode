# src/plotting.py
from __future__ import annotations

from typing import Optional, Dict, List, Tuple
import os
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# ============================= Styling & IO helpers ============================= #

def set_style() -> None:
    """
    Set a clean, publication-ready style.
    """
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.0)
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.titleweight": "semibold",
        "axes.labelweight": "regular",
        "legend.frameon": False,
        "pdf.fonttype": 42,  # editable text in vector outputs
        "ps.fonttype": 42,
    })


def ensure_outdir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def savefig(fig: plt.Figure, out_base: str | Path) -> None:
    """
    Save both PNG and EPS variants given a base path (without extension).
    Example: out_base="output/num_tests_over_time" -> saves PNG and EPS.
    """
    out_base = str(out_base)
    png_path = f"{out_base}.png"
    eps_path = f"{out_base}.eps"
    # Avoid transparency in EPS backend
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(eps_path, bbox_inches="tight")
    plt.close(fig)


# ============================= Generic plotting helpers ========================= #

def plot_timeseries(df: pd.DataFrame,
                    x: str,
                    y: str,
                    title: str,
                    xlabel: str,
                    ylabel: str,
                    out_base: str | Path,
                    hue: Optional[str] = None) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    sns.lineplot(data=df, x=x, y=y, hue=hue, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if hue:
        ax.legend(title=hue)
    savefig(fig, out_base)


def plot_hist(df: pd.DataFrame,
              col: str,
              bins: int,
              title: str,
              xlabel: str,
              out_base: str | Path) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    sns.histplot(df[col], bins=bins, kde=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    savefig(fig, out_base)


def plot_box(df: pd.DataFrame,
             x: Optional[str],
             y: str,
             title: str,
             xlabel: str,
             ylabel: str,
             out_base: str) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    if x is None:
        # y-only boxplot
        sns.boxplot(y=df[y], ax=ax)
    else:
        sns.boxplot(data=df, x=x, y=y, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    savefig(fig, out_base)


def plot_ecdf(df: pd.DataFrame,
              col: str,
              title: str,
              xlabel: str,
              out_base: str | Path) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    sns.ecdfplot(df[col], ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ECDF")
    savefig(fig, out_base)


# ============================= Oracletest-specific wrappers ===================== #
# These match the calls made in src/oracletest.py

def plot_active_nodes_vs_draws(res, out_dir: str | Path) -> None:
    """
    Reconstruct 'active nodes over time' from the trace.
    Assumptions:
      - Nodes only change active status when they are picked (consistent with the code).
      - Initially all nodes are active (initial trust >= threshold).
    This uses the per-draw 'active_after' for the picked node to update its status,
    then counts how many nodes are still active after each draw.
    """
    ensure_outdir(out_dir)
    # Determine node universe from trace
    node_ids = set(t.node_idx for t in res.trace)
    # Start with all 'seen' nodes active=True (reasonable for initial trust >= threshold)
    status: Dict[int, bool] = {i: True for i in node_ids}

    rows: List[Tuple[int, int]] = []  # (draw_idx, active_count)
    for t in res.trace:
        status[t.node_idx] = bool(t.active_after)
        active_count = sum(1 for v in status.values() if v)
        rows.append((t.draw_idx, active_count))

    df = pd.DataFrame(rows, columns=["draw_idx", "active_nodes"])
    plot_timeseries(
        df, x="draw_idx", y="active_nodes",
        title="Active Nodes vs. Draws",
        xlabel="Draws",
        ylabel="Active Nodes",
        out_base=Path(out_dir) / "active_nodes_vs_draws"
    )


def plot_confidence_lower_vs_draws(res, out_dir: str | Path) -> None:
    """
    Plot Wilson lower bound of correctness vs draws.
    """
    ensure_outdir(out_dir)
    df = pd.DataFrame(
        [(t.draw_idx, t.wilson_lower) for t in res.trace],
        columns=["draw_idx", "wilson_lower"]
    )
    plot_timeseries(
        df, x="draw_idx", y="wilson_lower",
        title="Confidence (Wilson Lower Bound) vs. Draws",
        xlabel="Draws",
        ylabel="Wilson Lower Bound",
        out_base=Path(out_dir) / "confidence_lower_vs_draws"
    )


def plot_mean_trust_vs_draws(res, out_dir: str | Path, nodes=None) -> None:
    """
    Plot the cumulative mean of 'trust_after' for the nodes that were selected up to each draw.
    (This doesn't require full network trust snapshots.)
    """
    ensure_outdir(out_dir)
    trusts = [t.trust_after for t in res.trace]
    cum_mean = []
    s = 0.0
    for i, v in enumerate(trusts, 1):
        s += float(v)
        cum_mean.append(s / i)

    df = pd.DataFrame(
        {"draw_idx": [t.draw_idx for t in res.trace],
         "mean_trust": cum_mean}
    )
    plot_timeseries(
        df, x="draw_idx", y="mean_trust",
        title="Mean Trust (Cumulative Over Selected Nodes) vs. Draws",
        xlabel="Draws",
        ylabel="Cumulative Mean Trust",
        out_base=Path(out_dir) / "mean_trust_vs_draws"
    )


def plot_p_hat_hist(res, out_dir: str | Path) -> None:
    """
    Build per-node empirical accuracy (p-hat) for nodes that were selected at least once:
        p_hat_i = (# correct for node i) / (# selections of node i)
    and plot a histogram.
    """
    ensure_outdir(out_dir)
    # Aggregate per-node statistics from trace
    succ: Dict[int, int] = {}
    cnt: Dict[int, int] = {}
    for t in res.trace:
        cnt[t.node_idx] = cnt.get(t.node_idx, 0) + 1
        succ[t.node_idx] = succ.get(t.node_idx, 0) + int(t.correct)

    data = []
    for nid, c in cnt.items():
        if c > 0:
            data.append(succ[nid] / c)

    df = pd.DataFrame({"p_hat": data})
    plot_hist(
        df, col="p_hat", bins=20,
        title="Per-Node Empirical Accuracy (selected nodes)",
        xlabel="p̂ (successes / selections)",
        out_base=Path(out_dir) / "p_hat_hist"
    )