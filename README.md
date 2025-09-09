Oracle Node Trust Simulation

A small, reproducible codebase for evaluating adaptive trust mechanisms for blockchain oracle nodes.
It supports single-run experiments and parameter sweeps, with plots and CSV outputs suitable for inclusion in a paper’s artifact appendix.

TL;DR

# 1) create env + install deps
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# 2) run a single experiment
python -m scripts.run_test --config configs/default.yaml --out results/run_$(date +%F_%H%M)

# 3) run a sweep
python -m scripts.run_sweep --config configs/sweep_example.yaml --out results_sweep/$(date +%F_%H%M)

All outputs (CSV + figures + resolved config) are written under the chosen --out directory.

⸻

Features
	•	Trust updates: discounted (exponentially-weighted) frequentist updates and/or discounted Beta (Bayesian) updates.
	•	Selection policies: trust-weighted sampling with configurable exponent; hooks for VRF-like or uniform baselines.
	•	Stopping rules: target accuracy + confidence using Jeffreys or Wald intervals.
	•	Correlation models (optional): Beta–Binomial over-dispersion, intra-class correlation, and simple collusion modes.
	•	Parameter sweeps: grid/random sweeps over thresholds, decay, selection exponent, node mix, etc.
	•	Reproducible outputs: resolved config, per-replicate CSVs, summaries, and publication-ready plots (PNG/PDF/EPS).

⸻

Layout

.
├─ src/
│  ├─ oraclenode.py           # OracleNode class (types, accuracy, trust update)
│  ├─ oracletest.py           # OracleTest (single run; selection + stopping rule)
│  ├─ oraclebatchtest.py      # OracleBatchTest (multiple replicates; aggregates)
│  ├─ oracleparamsweep.py     # ParameterSweepOracleBatchTest (grid/random sweeps)
│  ├─ plotting.py             # Centralized plotting functions (seaborn/matplotlib)
│  ├─ config.py               # Dataclasses + YAML load/validate + defaults
│  └─ utils.py                # Seeding, hashing, timers, I/O helpers
├─ scripts/
│  ├─ run_test.py             # CLI for a single config
│  └─ run_sweep.py            # CLI for parameter sweeps
├─ configs/
│  ├─ default.yaml            # Single-run example
│  └─ sweep_example.yaml      # Example sweep over a few parameters
├─ results/                   # (auto-created) outputs live here
├─ requirements.txt
├─ .gitignore
└─ README.md

We intentionally keep no notebooks in this repo. All figures are generated via the CLI and src/plotting.py.

⸻

Install

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

requirements.txt (pinned):

python_version==3.11
numpy==1.26.4
scipy==1.13.1
pandas==2.2.2
matplotlib==3.8.4
seaborn==0.13.2
pyyaml==6.0.2


⸻

Configuration

All “magic numbers” live in YAML. Below are short examples (use them as starters).

configs/default.yaml

run:
  num_nodes: 1000
  replicates: 100
  seed: 123
  hours_per_test: 0.5

trust:
  mode: "ewma"         # or: "beta_discounted"
  decay: 0.98          # EWMA γ; ignored if mode=beta_discounted
  prior_a: 1.0         # for beta_discounted
  prior_b: 1.0
  cap: 0.99
  kick_threshold: 0.55
  update_every: 10     # update trust every N picks

selection:
  policy: "trust_weighted"  # or: "uniform"
  exponent: 2.0             # weight = trust^exponent

stopping:
  target_accuracy: 0.95
  confidence: 0.999
  interval: "jeffreys"      # or: "wald"

correlation:
  bb_kappa: null            # Beta-Binomial over-dispersion (null = off)
  icc_rho: 0.0              # intra-class correlation (0.0 = off)
  collusion_frac: 0.0

nodes:
  malicious:   { proportion: 0.01, mean: 0.25, std: 0.05 }
  incompetent: { proportion: 0.09, mean: 0.55, std: 0.05 }
  competent:   { proportion: 0.90, mean: 0.90, std: 0.05 }

configs/sweep_example.yaml

base_config: configs/default.yaml

sweep:
  trust.decay:               [0.90, 0.95, 0.98]
  trust.kick_threshold:      [0.50, 0.55, 0.60]
  selection.exponent:        [1.0, 2.0, 3.0]
  nodes.incompetent.proportion: [0.05, 0.10, 0.20]
  correlation.icc_rho:       [0.0, 0.1]

Each run saves the resolved config alongside results so you can trace exactly what was executed.

⸻

Usage

Single run

python -m scripts.run_test \
  --config configs/default.yaml \
  --out results/run_$(date +%F_%H%M)

Parameter sweep

python -m scripts.run_sweep \
  --config configs/sweep_example.yaml \
  --out results_sweep/$(date +%F_%H%M)

Outputs (per run):
	•	config_resolved.yaml — exact config used
	•	replicates.csv — one row per replicate (time-to-confidence, premature deactivations, temporary retentions, survival counts, etc.)
	•	summary.csv — aggregates (mean/CI across replicates)
	•	plots/ — publication-ready figures:
	•	time_to_confidence.(png|pdf|eps)
	•	false_positives_over_time.(png|pdf|eps)
	•	false_negatives_over_time.(png|pdf|eps)
	•	incompetent_survival.(png|pdf|eps)
	•	trust_distributions.(png|pdf|eps) (optional, if enabled)

⸻

Reproducibility
	•	All randomness is seeded (run.seed) and propagated to NumPy/Scipy.
	•	Library versions are pinned in requirements.txt.
	•	Every run stores its resolved config, CSVs, and plots under results/….

⸻

.gitignore (suggestion)

# env
.venv/
__pycache__/
*.pyc

# results & caches
results*/
**/plots/
*.log

# OS / editor
.DS_Store
.idea/
.vscode/


⸻

License

MIT License — see LICENSE.

⸻

Support / Questions

Please open an issue or PR with a minimal reproducible example (config + command).
