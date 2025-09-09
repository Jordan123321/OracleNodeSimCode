from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

@dataclass(frozen=True)
class NodeDist:
    proportion: float
    mean: float
    std: float

@dataclass(frozen=True)
class NodesConfig:
    malicious: NodeDist
    incompetent: NodeDist
    competent: NodeDist

@dataclass(frozen=True)
class TrustConfig:
    mode: str                # "ewma" | "beta_discounted"
    decay: float             # EWMA γ
    prior_a: float           # Beta prior
    prior_b: float
    cap: float
    kick_threshold: float
    update_every: int

@dataclass(frozen=True)
class SelectionConfig:
    policy: str              # "trust_weighted" | "uniform"
    exponent: float          # trust^exponent

@dataclass(frozen=True)
class StoppingConfig:
    target_accuracy: float
    confidence: float
    interval: str            # "jeffreys" | "wald"

@dataclass(frozen=True)
class CorrelationConfig:
    bb_kappa: Optional[float]
    icc_rho: float
    collusion_frac: float

@dataclass(frozen=True)
class RunConfig:
    num_nodes: int
    replicates: int
    seed: int
    hours_per_test: float

@dataclass(frozen=True)
class ExperimentConfig:
    run: RunConfig
    trust: TrustConfig
    selection: SelectionConfig
    stopping: StoppingConfig
    correlation: CorrelationConfig
    nodes: NodesConfig

def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out

def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    # handle base_config: merge then re-parse final
    if 'base_config' in raw:
        base = yaml.safe_load(Path(raw['base_config']).read_text())
        raw = _deep_update(base, {k: v for k, v in raw.items() if k != 'base_config'})
    # simple validation could be added here
    def nd(d): return NodeDist(**d)
    cfg = ExperimentConfig(
        run=RunConfig(**raw['run']),
        trust=TrustConfig(**raw['trust']),
        selection=SelectionConfig(**raw['selection']),
        stopping=StoppingConfig(**raw['stopping']),
        correlation=CorrelationConfig(**raw['correlation']),
        nodes=NodesConfig(
            malicious=nd(raw['nodes']['malicious']),
            incompetent=nd(raw['nodes']['incompetent']),
            competent=nd(raw['nodes']['competent']),
        ),
    )
    return cfg