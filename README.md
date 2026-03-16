# bcrnnoise

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![static analysis workflow](https://github.com/BioDisCo/bcrnnoise/actions/workflows/static-analysis.yaml/badge.svg)](https://github.com/BioDisCo/bcrnnoise/actions/workflows/static-analysis.yaml/)

A Python library for simulating Bio-Chemical Reaction Networks (BCRNs) with noise. Supports deterministic ODE integration, exact Markov chain simulation via Gillespie's algorithm, and stochastic SDE simulation via Euler-Maruyama — including a vectorised batch mode for efficient multi-trajectory sampling.

Available on [PyPI](https://pypi.org/project/bcrnnoise/):

```bash
pip install bcrnnoise
```

## Overview

Models are defined by subclassing `BCRN` and implementing two things:

- **`stoichiometry`** — numpy array of shape `(n_reactions, n_species)` giving net stoichiometric changes per reaction
- **`reaction_rates(state)`** — returns per-reaction propensities as `list[Quantity]`

All quantities carry physical units via [pint](https://pint.readthedocs.io), so dimensional errors are caught at runtime rather than buried in wrong results.

## Simulation modes

| Method | Type | Description |
|---|---|---|
| `simulate_ode()` | deterministic | Integrates the mean-field ODE via `scipy.solve_ivp` |
| `simulate_markov_chain(seed)` | stochastic | Exact sample path via Gillespie's algorithm |
| `simulate_sde(noise_fun, seed)` | stochastic | Euler-Maruyama with custom additive noise |
| `simulate_sde_batch(noise_fun, n, seed)` | stochastic | Vectorised Euler-Maruyama for `n` trajectories simultaneously |

All methods return a `Timeseries` (or `BatchTimeseries` for the batch variant) with a shared time axis and unit-carrying states.

## Quick start

```python
from collections.abc import Sequence
import numpy as np
from pint import Quantity, UnitRegistry
from bcrnnoise import BCRN, plot_timeseries

u = UnitRegistry()

class TranscriptionSystem(BCRN):
    """0 -> mRNA [alpha],  mRNA -> 0 [delta]"""

    def __init__(self, alpha, delta, **kwargs):
        super().__init__(**kwargs)
        self._alpha, self._delta = alpha, delta

    @property
    def stoichiometry(self) -> np.ndarray:
        return np.array([[1], [-1]])

    def reaction_rates(self, state: Sequence[Quantity]) -> list[Quantity]:
        return [self._alpha, self._delta * state[0]]

sys = TranscriptionSystem(
    alpha=1.0 / u.minute / u.femtoliter,
    delta=0.1 / u.minute,
    init_state=[0.0 / u.femtoliter],
    volume=1.0 * u.femtoliter,
    time_horizon=200.0 * u.minute,
    dt=0.1 * u.minute,
)

# Deterministic
ode_ts = sys.simulate_ode()

# Exact stochastic
mc_ts = sys.simulate_markov_chain(seed=0)

# SDE with Gaussian noise
sigma = 5.0 / u.femtoliter / u.minute**0.5
def gaussian_noise(rng, t, state):
    return [sigma * np.sqrt(sys.dt) * rng.normal()]

sde_ts = sys.simulate_sde(noise_fun=gaussian_noise, seed=0)

plot_timeseries([ode_ts, mc_ts, sde_ts], labels=["ODE", "Markov chain", "SDE"])
```

## Batch SDE simulation

For multi-trajectory studies, `simulate_sde_batch` runs `n` trajectories in a single vectorised loop — roughly `n`× faster than calling `simulate_sde` repeatedly:

```python
def gaussian_batch_noise(rng, t, states):
    n = states[0].magnitude.shape[0]
    return [sigma * np.sqrt(sys.dt) * rng.normal(size=n)]

batch = sys.simulate_sde_batch(noise_fun=gaussian_batch_noise, n=500, seed=0)

# Access individual trajectory
ts_42 = batch.trajectory(42)

# Unpack all at once (single pass, more efficient than looping trajectory())
all_ts = batch.to_timeseries_list()
```

The noise function receives and returns `list[Quantity]` where each Quantity wraps a `(n,)` numpy array. Any `reaction_rates` implementation that uses only numpy-compatible arithmetic works in batch mode without modification.

## Examples

- [`examples/example_vanilla.py`](examples/example_vanilla.py) — minimal one-species model with ODE, Markov chain, Gaussian and Pareto SDE
- [`examples/example_rna.py`](examples/example_rna.py) — multi-species transcription models including two-state promoter and geometric noise
- [`examples/example_batch.py`](examples/example_batch.py) — batch SDE simulation with ensemble mean vs ODE comparison

## Citation

```bibtex
@misc{bcrnnoise:github,
  author       = {Arman Ferdowsi and Mattias F{\"u}gger and Thomas Nowak},
  title        = {bcrnnoise: simulate Bio-Chemical Reaction Networks with noise},
  year         = {2025},
  howpublished = {\url{https://github.com/BioDisCo/bcrnnoise}},
}
```
