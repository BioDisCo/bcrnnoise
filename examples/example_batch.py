"""Batch SDE simulation: N trajectories of a birth-death system in one vectorised call.

Demonstrates simulate_sde_batch, which runs N simultaneous Euler-Maruyama
trajectories by storing species states as Quantities with array magnitudes.
The speedup over N sequential simulate_sde calls is roughly linear in N.

Reactions:
    0 -> mRNA  [alpha]   (production)
    mRNA -> 0  [delta]   (degradation)

Steady state: alpha / delta = 10 / fL.
"""

from collections.abc import Sequence
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from pint import Quantity, UnitRegistry

from bcrnnoise import BCRN, BatchTimeseries


class VanillaParams(NamedTuple):
    alpha: Quantity
    delta: Quantity
    sigma: Quantity
    init_mrna: Quantity
    volume: Quantity
    time_horizon: Quantity
    dt: Quantity


class VanillaSystem(BCRN):
    def __init__(self, params: VanillaParams) -> None:
        super().__init__(
            init_state=[params.init_mrna],
            time_horizon=params.time_horizon,
            volume=params.volume,
            dt=params.dt,
        )
        self.params = params

    @property
    def stoichiometry(self) -> np.ndarray:
        return np.array([[1], [-1]])

    def reaction_rates(self, state: Sequence[Quantity]) -> list[Quantity]:
        return [self.params.alpha, self.params.delta * state[0]]

    def gaussian_batch_noise(self, rng: np.random.Generator, _t: Quantity, states: list[Quantity]) -> list[Quantity]:
        n = states[0].magnitude.shape[0]
        return [self.params.sigma * np.sqrt(self.params.dt) * rng.normal(size=n)]


def main() -> None:
    u = UnitRegistry()

    params = VanillaParams(
        alpha=1.0 / u.minute / u.femtoliter,
        delta=0.1 / u.minute,
        sigma=2.0 / u.femtoliter / u.minute**0.5,
        init_mrna=0.0 / u.femtoliter,
        volume=1.0 * u.femtoliter,
        time_horizon=80.0 * u.minute,
        dt=0.1 * u.minute,
    )
    sys = VanillaSystem(params)

    n = 300
    batch: BatchTimeseries = sys.simulate_sde_batch(noise_fun=sys.gaussian_batch_noise, n=n, seed=0)
    ode_ts = sys.simulate_ode()

    times_min = [t.to(u.minute).magnitude for t in batch.times]
    ode_times_min = [t.to(u.minute).magnitude for t in ode_ts.times]
    ode_vals = [s[0].to(u.femtoliter**-1).magnitude for s in ode_ts.states]

    _, ax = plt.subplots(figsize=(7, 4))

    for j in range(n):
        vals = [step[0].magnitude[j] for step in batch.states]
        ax.plot(times_min, vals, color="steelblue", alpha=0.04, lw=0.6)

    mean_vals = [step[0].magnitude.mean() for step in batch.states]
    ax.plot(times_min, mean_vals, color="steelblue", lw=2, label=f"SDE mean (N={n})")
    ax.plot(ode_times_min, ode_vals, "r--", lw=2, label="ODE")

    ax.set_xlabel(f"Time ({u.minute})")
    ax.set_ylabel(f"mRNA ({u.femtoliter**-1})")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
