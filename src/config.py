# src/config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Iterable, List, Tuple
import itertools
import yaml

from .oraclenode import TrustMode  # enum reuse


# ------------------------------- Config Dataclasses -------------------------------

@dataclass(frozen=True)
class PopulationMix:
    malicious: float
    incompetent: float
    competent: float

    def validate(self) -> None:
        s = self.malicious + self.incompetent + self.competent
        if abs(s - 1.0) > 1e-9:
            raise ValueError(f"PopulationMix must sum to 1.0, got {s}")
        for name, v in [("malicious", self.malicious),
                        ("incompetent", self.incompetent),
                        ("competent", self.competent)]:
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} proportion must be in [0,1], got {v}")


@dataclass(frozen=True)
class AccuracyProfile:
    mean_malicious: float = 0.25
    std_malicious: float = 0.05
    mean_incompetent: float = 0.55
    std_incompetent: float = 0.05
    mean_competent: float = 0.90
    std_competent: float = 0.05


@dataclass(frozen=True)
class TrustConfig:
    initial_trust: float = 0.65
    trust_cap: float = 0.95
    trust_mode: TrustMode = TrustMode.BETA_DISCOUNTED
    decay: float = 0.9          # EWMA gamma or Beta-discounter
    prior_a: float = 1.0
    prior_b: float = 1.0
    update_every: int = 10
    kick_threshold: float = 0.55


@dataclass(frozen=True)
class SelectionConfig:
    exponent: float = 2.0
    without_replacement: bool = True
    min_weight: float = 0.0


@dataclass(frozen=True)
class StoppingRuleConfig:
    mode: str = "wilson"              # "wilson" | "jeffreys"
    target: float = 0.5
    confidence: float = 0.999
    phi_overdispersion: float = 0.0   # >= 0, inflates variance if > 0
    max_draws: int = 10_000


@dataclass(frozen=True)
class OracleTestConfig:
    seed: int = 42
    num_nodes: int = 1000
    mix: PopulationMix = PopulationMix(0.01, 0.09, 0.90)
    acc: AccuracyProfile = AccuracyProfile()
    trust: TrustConfig = TrustConfig()
    select: SelectionConfig = SelectionConfig()
    stop: StoppingRuleConfig = StoppingRuleConfig()


# ------------------------------- YAML Helpers -------------------------------

def _get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _mk_mix(d: Dict[str, Any]) -> PopulationMix:
    return PopulationMix(
        malicious=float(_get(d, "mix.malicious", 0.01)),
        incompetent=float(_get(d, "mix.incompetent", 0.09)),
        competent=float(_get(d, "mix.competent", 0.90)),
    )


def _mk_acc(d: Dict[str, Any]) -> AccuracyProfile:
    return AccuracyProfile(
        mean_malicious=float(_get(d, "acc.mean_malicious", 0.25)),
        std_malicious=float(_get(d, "acc.std_malicious", 0.05)),
        mean_incompetent=float(_get(d, "acc.mean_incompetent", 0.55)),
        std_incompetent=float(_get(d, "acc.std_incompetent", 0.05)),
        mean_competent=float(_get(d, "acc.mean_competent", 0.90)),
        std_competent=float(_get(d, "acc.std_competent", 0.05)),
    )


def _mk_trust(d: Dict[str, Any]) -> TrustConfig:
    mode = str(_get(d, "trust.trust_mode", "beta_discounted")).lower()
    return TrustConfig(
        initial_trust=float(_get(d, "trust.initial_trust", 0.65)),
        trust_cap=float(_get(d, "trust.trust_cap", 0.95)),
        trust_mode=TrustMode(mode),
        decay=float(_get(d, "trust.decay", 0.9)),
        prior_a=float(_get(d, "trust.prior_a", 1.0)),
        prior_b=float(_get(d, "trust.prior_b", 1.0)),
        update_every=int(_get(d, "trust.update_every", 10)),
        kick_threshold=float(_get(d, "trust.kick_threshold", 0.55)),
    )


def _mk_select(d: Dict[str, Any]) -> SelectionConfig:
    return SelectionConfig(
        exponent=float(_get(d, "select.exponent", 2.0)),
        without_replacement=bool(_get(d, "select.without_replacement", True)),
        min_weight=float(_get(d, "select.min_weight", 0.0)),
    )


def _mk_stop(d: Dict[str, Any]) -> StoppingRuleConfig:
    return StoppingRuleConfig(
        mode=str(_get(d, "stop.mode", "wilson")),
        target=float(_get(d, "stop.target", 0.5)),
        confidence=float(_get(d, "stop.confidence", 0.999)),
        phi_overdispersion=float(_get(d, "stop.phi_overdispersion", 0.0)),
        max_draws=int(_get(d, "stop.max_draws", 10_000)),
    )


def load_test_config(yaml_path: str) -> OracleTestConfig:
    """
    Load a single-run OracleTestConfig from a YAML file.
    If lists are present under 'sweep.grid', they are ignored here.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    otc = OracleTestConfig(
        seed=int(_get(cfg, "seed", 42)),
        num_nodes=int(_get(cfg, "num_nodes", 1000)),
        mix=_mk_mix(cfg),
        acc=_mk_acc(cfg),
        trust=_mk_trust(cfg),
        select=_mk_select(cfg),
        stop=_mk_stop(cfg),
    )
    otc.mix.validate()
    return otc


# ------------------------------- Parameter Sweep -------------------------------

def _set_nested(d: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def expand_sweep_grid(yaml_path: str) -> List[Tuple[OracleTestConfig, Dict[str, Any]]]:
    """
    Expand a sweep grid from YAML. Returns a list of (config, overrides) pairs.

    YAML example:
      sweep:
        repeats: 50
        grid:
          trust.decay: [0.7, 0.8, 0.9]
          select.exponent: [1.0, 2.0]

    Each combination is replicated 'repeats' times with different seeds.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}

    base_cfg = load_test_config(yaml_path)

    grid: Dict[str, Iterable[Any]] = (_get(base, "sweep.grid", {}) or {})
    if not grid:
        return [(base_cfg, {})]

    # Cartesian product of all grid lists
    items = sorted(grid.items(), key=lambda kv: kv[0])
    keys = [k for k, _ in items]
    vals_lists = [list(v) for _, v in items]
    combos = list(itertools.product(*vals_lists))

    repeats = int(_get(base, "sweep.repeats", 1))
    out: List[Tuple[OracleTestConfig, Dict[str, Any]]] = []

    for combo in combos:
        # Build an overrides dict -> then materialize a config
        overrides: Dict[str, Any] = {}
        for k, v in zip(keys, combo):
            overrides[k] = v

        for r in range(repeats):
            # Apply overrides onto a shallow dict form, then reconstruct config
            d: Dict[str, Any] = {}
            _set_nested(d, "seed", int(base_cfg.seed + r))
            _set_nested(d, "num_nodes", base_cfg.num_nodes)

            # mix
            _set_nested(d, "mix.malicious", base_cfg.mix.malicious)
            _set_nested(d, "mix.incompetent", base_cfg.mix.incompetent)
            _set_nested(d, "mix.competent", base_cfg.mix.competent)

            # acc
            _set_nested(d, "acc.mean_malicious", base_cfg.acc.mean_malicious)
            _set_nested(d, "acc.std_malicious", base_cfg.acc.std_malicious)
            _set_nested(d, "acc.mean_incompetent", base_cfg.acc.mean_incompetent)
            _set_nested(d, "acc.std_incompetent", base_cfg.acc.std_incompetent)
            _set_nested(d, "acc.mean_competent", base_cfg.acc.mean_competent)
            _set_nested(d, "acc.std_competent", base_cfg.acc.std_competent)

            # trust
            _set_nested(d, "trust.initial_trust", base_cfg.trust.initial_trust)
            _set_nested(d, "trust.trust_cap", base_cfg.trust.trust_cap)
            _set_nested(d, "trust.trust_mode", base_cfg.trust.trust_mode.value)
            _set_nested(d, "trust.decay", base_cfg.trust.decay)
            _set_nested(d, "trust.prior_a", base_cfg.trust.prior_a)
            _set_nested(d, "trust.prior_b", base_cfg.trust.prior_b)
            _set_nested(d, "trust.update_every", base_cfg.trust.update_every)
            _set_nested(d, "trust.kick_threshold", base_cfg.trust.kick_threshold)

            # select
            _set_nested(d, "select.exponent", base_cfg.select.exponent)
            _set_nested(d, "select.without_replacement", base_cfg.select.without_replacement)
            _set_nested(d, "select.min_weight", base_cfg.select.min_weight)

            # stop
            _set_nested(d, "stop.mode", base_cfg.stop.mode)
            _set_nested(d, "stop.target", base_cfg.stop.target)
            _set_nested(d, "stop.confidence", base_cfg.stop.confidence)
            _set_nested(d, "stop.phi_overdispersion", base_cfg.stop.phi_overdispersion)
            _set_nested(d, "stop.max_draws", base_cfg.stop.max_draws)

            # Apply overrides from grid onto d
            for path, val in overrides.items():
                _set_nested(d, path, val)

            # Reconstruct a config from d
            cfg = OracleTestConfig(
                seed=int(d["seed"]),
                num_nodes=int(d["num_nodes"]),
                mix=PopulationMix(**d["mix"]),
                acc=AccuracyProfile(**d["acc"]),
                trust=TrustConfig(
                    **{**d["trust"], "trust_mode": TrustMode(str(d["trust"]["trust_mode"]).lower())}
                ),
                select=SelectionConfig(**d["select"]),
                stop=StoppingRuleConfig(**d["stop"]),
            )
            cfg.mix.validate()
            out.append((cfg, overrides))

    return out