# src/oracletest.py
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.stats import norm

# ---- Local imports (works both as module and script) ---------------------------------
try:
    from .oraclenode import OracleNode, NodeType, TrustMode
except ImportError:  # running as script
    from oraclenode import OracleNode, NodeType, TrustMode

# Optional plotting utils (safe if absent)
try:
    from . import plotting  # your helper module (optional)
except Exception:
    plotting = None


# ============================== Configuration =========================================

@dataclass
class Config:
    # RNG / size
    seed: int = 123
    num_nodes: int = 1000

    # population mix (competent = 1 - malicious - incompetent)
    malicious_frac: float = 0.01
    incompetent_frac: float = 0.09

    # accuracy model per type
    malicious_mean: float = 0.25
    incompetent_mean: float = 0.55
    competent_mean: float = 0.90
    accuracy_std: float = 0.05

    # trust model (mapped to TrustMode)
    initial_trust: float = 0.65
    trust_cap: float = 0.95
    trust_mode: str = "ewma"       # "ewma" | "beta" (alias "discounted" -> "ewma")
    trust_decay: float = 0.90      # EWMA decay
    prior_a: float = 6.5           # Beta prior a
    prior_b: float = 3.5           # Beta prior b
    update_every: int = 10         # how often to refresh trust
    kick_threshold: float = 0.55   # node kicked if trust < threshold

    # selection policy
    selection_exponent: float = 2.0
    sel_without_replacement: bool = True

    # stopping policy (Wilson lower bound >= target_accuracy)
    target_accuracy: float = 0.50
    confidence: float = 0.999
    min_draws: int = 0
    max_draws: int = 10000

    # outputs
    out_dir: str = "results/latest"
    save_csv: bool = True
    save_plots: bool = True


def _default_cfg() -> Config:
    return Config()


def _apply_yaml(cfg: Config, path_str: str) -> Config:
    """Strictly load YAML and apply onto cfg; fail on unknown keys; echo final cfg."""
    import yaml
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p.resolve()}")

    with p.open("r") as f:
        data = yaml.safe_load(f) or {}

    fields = set(cfg.__dataclass_fields__.keys())
    unknown = set(data) - fields
    if unknown:
        raise KeyError(f"Unknown config keys in {p.name}: {sorted(unknown)}")

    for k, v in data.items():
        setattr(cfg, k, v)

    print("[cfg loaded from YAML]")
    for k in sorted(fields):
        print(f"  {k}: {getattr(cfg, k)}")
    return cfg


# ============================== Helpers ===============================================

def _to_trust_mode(name: str) -> TrustMode:
    n = (name or "").strip().lower()
    if n in ("ewma", "discounted"):
        return TrustMode.EWMA
    if n == "beta":
        return TrustMode.BETA
    # default
    return TrustMode.EWMA


def _wilson_lower(successes: int, n: int, conf: float) -> float:
    """Wilson score interval (lower bound) for Bernoulli p with confidence `conf`."""
    if n <= 0:
        return 0.0
    z = norm.ppf(0.5 * (1.0 + conf))
    p = successes / n
    denom = 1.0 + (z * z) / n
    center = p + (z * z) / (2.0 * n)
    margin = z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)
    lower = (center - margin) / denom
    return max(0.0, min(1.0, lower))


# ============================== Core Test =============================================

@dataclass
class DrawTrace:
    draw_idx: int
    node_idx: int
    correct: bool
    trust_after: float
    active_after: bool
    successes: int
    draws: int
    wilson_lower: float

@dataclass
class TestResult:
    draws: int
    successes: int
    failures: int
    reached_target: bool
    false_positives: int
    false_negatives: int
    incompetent_removed: int
    trace: List[DrawTrace]


class OracleTest:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.RandomState(cfg.seed)

        # build population
        self.nodes: List[OracleNode] = self._build_nodes()

    def _build_nodes(self) -> List[OracleNode]:
        n = self.cfg.num_nodes
        m = int(round(self.cfg.malicious_frac * n))
        i = int(round(self.cfg.incompetent_frac * n))
        c = max(0, n - m - i)

        nodes: List[OracleNode] = []

        def mk(ntype: NodeType, mean: float, count: int):
            for _ in range(count):
                nodes.append(
                    OracleNode(
                        node_type=ntype,
                        accuracy_mean=mean,
                        accuracy_std=self.cfg.accuracy_std,
                        trust=self.cfg.initial_trust,
                        trust_cap=self.cfg.trust_cap,
                        trust_mode=_to_trust_mode(self.cfg.trust_mode),
                        decay=self.cfg.trust_decay,
                        prior_a=self.cfg.prior_a,
                        prior_b=self.cfg.prior_b,
                        update_every=self.cfg.update_every,
                        kick_threshold=self.cfg.kick_threshold,
                        rng=np.random.RandomState(self.rng.randint(0, 2**31 - 1)),
                    )
                )

        mk(NodeType.MALICIOUS, self.cfg.malicious_mean, m)
        mk(NodeType.INCOMPETENT, self.cfg.incompetent_mean, i)
        mk(NodeType.COMPETENT, self.cfg.competent_mean, c)
        self.rng.shuffle(nodes)
        return nodes

    def _choose_node(self, used_this_draw: Optional[set[int]] = None) -> int:
        used = used_this_draw or set()
        # choose among active nodes; if none active, fall back to all nodes
        pool_idx = [k for k, n in enumerate(self.nodes) if n.is_active()]
        if not pool_idx:
            pool_idx = list(range(len(self.nodes)))

        # if sampling without replacement, remove already used for this draw
        if self.cfg.sel_without_replacement:
            pool_idx = [k for k in pool_idx if k not in used]
            if not pool_idx:
                pool_idx = [k for k in range(len(self.nodes)) if k not in used]
                if not pool_idx:
                    pool_idx = list(range(len(self.nodes)))

        # weights ~ trust^exponent (with floor at 0)
        weights = np.array([max(0.0, self.nodes[k].trust) ** self.cfg.selection_exponent for k in pool_idx], dtype=float)
        s = float(weights.sum())
        if s <= 0.0:
            # uniform fallback
            return int(self.rng.choice(pool_idx))
        probs = weights / s
        return int(self.rng.choice(pool_idx, p=probs))

    def run(self) -> TestResult:
        traces: List[DrawTrace] = []
        successes = 0
        draws = 0

        # For “without replacement”, we only prevent reusing within the same draw.
        # Each iteration is a fresh selection; we don’t maintain a global exclusion set.
        while draws < self.cfg.max_draws:
            used_this_draw: set[int] = set()
            idx = self._choose_node(used_this_draw)
            used_this_draw.add(idx)

            node = self.nodes[idx]

            # simulate correctness based on node's accuracy_mean
            correct = bool(self.rng.rand() < float(getattr(node, "accuracy_mean", 0.0)))
            node.pick(correct)

            draws += 1
            if correct:
                successes += 1

            wl = _wilson_lower(successes, draws, self.cfg.confidence)

            traces.append(
                DrawTrace(
                    draw_idx=draws,
                    node_idx=idx,
                    correct=correct,
                    trust_after=node.trust,
                    active_after=node.is_active(),
                    successes=successes,
                    draws=draws,
                    wilson_lower=wl,
                )
            )

            if draws >= self.cfg.min_draws and wl >= self.cfg.target_accuracy:
                break

        # final tallies
        false_pos = sum(1 for n in self.nodes if (n.node_type == NodeType.COMPETENT and not n.is_active()))
        false_neg = sum(1 for n in self.nodes if (n.node_type != NodeType.COMPETENT and n.is_active()))
        incompetent_removed = sum(1 for n in self.nodes if (n.node_type == NodeType.INCOMPETENT and not n.is_active()))

        return TestResult(
            draws=draws,
            successes=successes,
            failures=draws - successes,
            reached_target=(_wilson_lower(successes, draws, self.cfg.confidence) >= self.cfg.target_accuracy),
            false_positives=false_pos,
            false_negatives=false_neg,
            incompetent_removed=incompetent_removed,
            trace=traces,
        )


# ============================== I/O & CLI =============================================

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_trace_csv(out_dir: Path, res: TestResult) -> None:
    import csv
    p = out_dir / "trace.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["draw_idx", "node_idx", "correct", "trust_after", "active_after", "successes", "draws", "wilson_lower"])
        for t in res.trace:
            w.writerow([t.draw_idx, t.node_idx, int(t.correct), f"{t.trust_after:.6f}", int(t.active_after), t.successes, t.draws, f"{t.wilson_lower:.6f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", type=str, default=None, help="Path to YAML config.")
    ap.add_argument("--echo-config", action="store_true", help="Print effective config before running.")
    args = ap.parse_args()

    cfg = _default_cfg()
    if args.yaml:
        try:
            cfg = _apply_yaml(cfg, args.yaml)
        except Exception as e:
            raise SystemExit(f"[error] failed to load config: {e}")
    elif args.echo_config:
        print("[cfg using defaults]")
        for k, v in asdict(cfg).items():
            print(f"  {k}: {v}")

    # Run
    test = OracleTest(cfg)
    res = test.run()

    print(
        f"draws={res.draws}, succ={res.successes}, fail={res.failures}, "
        f"reached_target={res.reached_target}, FP={res.false_positives}, "
        f"FN={res.false_negatives}, incompetent_removed={res.incompetent_removed}"
    )

    # Outputs
    out_dir = Path(cfg.out_dir)
    _ensure_dir(out_dir)

    if cfg.save_csv:
        _save_trace_csv(out_dir, res)

    if cfg.save_plots and plotting is not None:
        try:
            plotting.plot_active_nodes_vs_draws(res, out_dir)
            plotting.plot_confidence_lower_vs_draws(res, out_dir)
            plotting.plot_mean_trust_vs_draws(res, out_dir, test.nodes)
            plotting.plot_p_hat_hist(res, out_dir)
        except Exception as e:
            print(f"[warn] plotting failed: {e}")
    elif cfg.save_plots:
        print("[warn] plotting utilities not available; skipping figure generation.")


if __name__ == "__main__":
    main()