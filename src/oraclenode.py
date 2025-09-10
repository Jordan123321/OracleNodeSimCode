# src/oraclenode.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Optional
from collections import deque
import numpy as np


class NodeType(Enum):
    MALICIOUS = auto()
    INCOMPETENT = auto()
    COMPETENT = auto()


class TrustMode(Enum):
    """How we compute/refresh the node's trust value."""
    EWMA = auto()  # Exponentially Weighted Moving Average of recent correctness
    BETA = auto()  # Beta-Bernoulli posterior mean with (a,b) priors


@dataclass
class OracleNode:
    """
    Oracle node with a simple behavior model (fixed accuracy) and a pluggable
    trust update mechanism (EWMA or Beta).

    Parameters
    ----------
    node_type : NodeType
        Category of node (malicious / incompetent / competent).
    accuracy_mean : float
        Probability that this node answers correctly on any draw.
    accuracy_std : float
        Included for completeness; not used by the default correctness sampler.
    trust : float
        Initial trust value in [0, 1].
    trust_cap : float
        Upper bound on trust (prevents runaway overconfidence).
    trust_mode : TrustMode
        EWMA or BETA. See update rules below.
    decay : float
        EWMA decay (0<decay<=1). Larger → slower decay (more history retained).
    prior_a, prior_b : float
        Beta prior hyperparameters if using TrustMode.BETA.
    update_every : int
        Recompute trust every N picks for stability/perf.
    kick_threshold : float
        If trust falls below this threshold, the node is marked inactive.
    rng : np.random.RandomState
        Node-local RNG for any stochastic behavior.

    Notes
    -----
    - In this simulation, the test harness decides whether an answer is
      correct by comparing a uniform random to `accuracy_mean`. The node
      simply records that outcome via `pick(correct)`.
    - If you prefer nodes to sample their own correctness (e.g., with noise),
      you can add a `sample_correct()` method and use it in the harness.
    """
    node_type: NodeType
    accuracy_mean: float
    accuracy_std: float
    trust: float
    trust_cap: float
    trust_mode: TrustMode
    decay: float
    prior_a: float
    prior_b: float
    update_every: int
    kick_threshold: float
    rng: np.random.RandomState

    # Internal state
    active: bool = field(default=True, init=False)
    picks: int = field(default=0, init=False)
    successes: int = field(default=0, init=False)
    failures: int = field(default=0, init=False)
    recent_results: Deque[bool] = field(default_factory=lambda: deque(maxlen=200), init=False)

    # -------------------------- Public interface --------------------------

    def is_active(self) -> bool:
        """Return True iff the node is currently allowed to participate."""
        return self.active and (self.trust >= self.kick_threshold)

    def pick(self, correct: bool) -> None:
        """
        Record the outcome of one selection (draw). Periodically updates trust.
        If trust drops below `kick_threshold`, the node becomes inactive.
        """
        self.picks += 1
        if correct:
            self.successes += 1
        else:
            self.failures += 1
        self.recent_results.append(bool(correct))

        if (self.picks % max(1, self.update_every)) == 0:
            self._refresh_trust()

        # Apply cap & kick policy
        self.trust = float(min(max(self.trust, 0.0), self.trust_cap))
        if self.trust < self.kick_threshold:
            self.active = False

    def weight(self, exponent: float = 2.0, min_weight: float = 0.0) -> float:
        """
        Selection weight = max(min_weight, trust)^exponent.
        """
        return float(max(min_weight, self.trust)) ** float(exponent)

    # ------------------------ Trust update internals ------------------------

    def _refresh_trust(self) -> None:
        """Recompute trust according to the chosen TrustMode."""
        if self.trust_mode == TrustMode.EWMA:
            self.trust = self._trust_ewma()
        elif self.trust_mode == TrustMode.BETA:
            self.trust = self._trust_beta_mean()
        else:
            # Fallback to EWMA if an unknown mode is passed
            self.trust = self._trust_ewma()

    def _trust_ewma(self) -> float:
        """
        Exponentially weighted average over recent outcomes (1=correct, 0=incorrect).

        Let x_t be outcomes in chronological order. We compute:
            w_0 = 1
            w_{k+1} = w_k * decay
            trust = sum_k x_{T-k} * w_k / sum_k w_k
        where `decay` in (0,1] controls the half-life of past observations.
        """
        if not self.recent_results:
            # No evidence yet; keep current trust (or return initial trust).
            return float(self.trust)

        # Iterate most-recent first to apply decay
        weighted_sum = 0.0
        weight = 1.0
        weight_sum = 0.0

        for outcome in reversed(self.recent_results):
            weighted_sum += (1.0 if outcome else 0.0) * weight
            weight_sum += weight
            weight *= float(self.decay)

        if weight_sum <= 0.0:
            return float(self.trust)
        return float(weighted_sum / weight_sum)

    def _trust_beta_mean(self) -> float:
        """
        Beta-Bernoulli posterior mean:
            trust = (a + successes) / (a + b + picks)
        This provides a principled estimate with built-in smoothing.
        """
        a = max(1e-6, float(self.prior_a))
        b = max(1e-6, float(self.prior_b))
        n = self.picks
        s = self.successes
        return float((a + s) / (a + b + n))

    # ---------------------------- Convenience ----------------------------

    def sample_correct(self) -> bool:
        """
        Optional: have the node sample its own correctness from accuracy_mean.
        Currently unused by the harness (which does the Bernoulli), but kept for flexibility.
        """
        return bool(self.rng.rand() < float(self.accuracy_mean))

    def as_dict(self) -> dict:
        """Lightweight snapshot of node state (useful for logging/debugging)."""
        return {
            "type": self.node_type.name.lower(),
            "trust": self.trust,
            "active": self.is_active(),
            "picks": self.picks,
            "successes": self.successes,
            "failures": self.failures,
            "accuracy_mean": self.accuracy_mean,
        }