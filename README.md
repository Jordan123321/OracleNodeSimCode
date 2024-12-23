# OracleNodeSimCode

# OracleNodeSimCode

This repository contains the Python implementation of the oracle node reliability framework. The framework is designed to assess the reliability of oracle nodes in decentralised networks using trust score mechanisms and adaptive testing.

## Overview

The goal of this project is to:
- Simulate the performance of oracle nodes in a decentralised network.
- Demonstrate the effectiveness of the trust score mechanism in achieving a 99.9% confidence level.
- Filter unreliable nodes (malicious or incompetent) while retaining competent ones.

## Features

- **Trust Score Updates**: Implements an exponential decay mechanism for updating trust scores based on test results.
- **Node Selection**: Nodes are selected based on their trust-weighted probabilities.
- **Simulation Results**: Provides visualizations for metrics like false positives, false negatives, and the number of tests required to achieve confidence.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required Python packages:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scipy`

