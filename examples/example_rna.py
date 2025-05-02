"""Transcription example.

This implements 2 systems with different noise/drift terms.

One state:
    0 -> RNA
    RNA -> 0

Two state:
    On <-> Off
    On -> On + RNA
    RNA -> 0

The two state system is seen to have a heavy tail distribution
    in the steady state.
"""

import logging
from abc import abstractmethod
from collections.abc import Sequence
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from bcrnnoise import BCRN, Timeseries, plot_timeseries
from pint import Quantity, UnitRegistry

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TranscriptionParameters(NamedTuple):
    """Parameters for a simple birth-death system with potential expansions.

    alpha: Rate of birth events [1/time/volume]
    delta: Rate of death [1/time]
    init_mRNA: Initial concentration [1/volume]
    volume: System volume
    time_horizon: Maximum simulation time
    dt: Step size for SDE
    sigma: Gaussian noise amplitude
    pareto_alpha: Pareto shape param (1 < alpha < 2)
    x_m: Pareto minimal jump [1/volume]
    """

    alpha: Quantity
    delta: Quantity
    on_to_off: Quantity
    off_to_on: Quantity

    init_mRNA: Quantity  # noqa: N815
    volume: Quantity
    time_horizon: Quantity
    dt: Quantity

    # for Gauss
    sigma: Quantity

    # for Pareto
    pareto_alpha: Quantity
    pareto_alpha_factor: Quantity
    x_m: Quantity

    # for Geom
    geom_alpha_factor: Quantity
    geom_p: Quantity


class TranscriptionBCRN(BCRN):
    @abstractmethod
    def simulate(self, seed: int = 42) -> Timeseries: ...


class OriginalSystem(TranscriptionBCRN):
    """Stage 0: original system."""

    params: TranscriptionParameters

    def __init__(self, params: TranscriptionParameters) -> None:
        super().__init__(
            init_state=[params.init_mRNA], time_horizon=params.time_horizon, volume=params.volume, dt=params.dt
        )
        self.params = params

    @property
    def stoichiometry(self) -> np.ndarray:
        return np.array([[1], [-1]])

    def reaction_rates(self, state: Sequence[Quantity]) -> list[Quantity]:
        return [self.params.alpha, self.params.delta * state[0]]

    def simulate(self, seed: int = 42) -> Timeseries:
        """Simulate a run."""
        return self.simulate_markov_chain(seed=seed)


class TwoStateSystem(TranscriptionBCRN):
    """Stage 0': like original system but with 2 states (on/off) of promoter."""

    params: TranscriptionParameters

    def __init__(self, params: TranscriptionParameters) -> None:
        # there is only 1 on/off promoter
        init_on = 0 * 1 / params.volume
        init_off = 1 * 1 / params.volume

        super().__init__(
            init_state=[params.init_mRNA, init_on, init_off],
            time_horizon=params.time_horizon,
            volume=params.volume,
            dt=params.dt,
        )
        self.params = params

    @property
    def stoichiometry(self) -> np.ndarray:
        return np.array([[1, 0, 0], [-1, 0, 0], [0, -1, 1], [0, 1, -1]])

    def reaction_rates(self, state: Sequence[Quantity]) -> list[Quantity]:
        rna = state[0]
        on = state[1]
        off = state[2]
        return [
            self.params.alpha * on * self.volume,
            self.params.delta * rna,
            self.params.on_to_off * on,
            self.params.off_to_on * off,
        ]

    def simulate(self, seed: int = 42) -> Timeseries:
        """Simulate a run."""
        return self.simulate_markov_chain(seed=seed)


class NoDetSystem(TranscriptionBCRN):
    """Stage 1: system without deterministic drift.

    Pure CME, but in a fixed-step approach (tau-leaping).
    No drift => reaction_rates=0, only random +/-1 events in birth_death_fun.
    """

    params: TranscriptionParameters

    def __init__(self, params: TranscriptionParameters) -> None:
        super().__init__(
            init_state=[params.init_mRNA], time_horizon=params.time_horizon, volume=params.volume, dt=params.dt
        )
        self.params = params

    @property
    def stoichiometry(self) -> np.ndarray:
        return np.array([[1], [-1]])

    def reaction_rates(self, state: Sequence[Quantity]) -> list[Quantity]:
        """Zero drift => no deterministic update."""
        return [0 * self.params.alpha, 0 * self.params.delta * state[0]]

    def noise(self, rng: np.random.Generator, t, y):
        """+/-1 jumps at rates alpha, delta*x."""
        x_now = y[-1]
        alpha = self.params.alpha
        delta = self.params.delta
        dt = self.dt
        vol = self.volume

        total_rate = (alpha + delta * x_now) * dt * vol
        if rng.random() < total_rate.magnitude:
            # decide birth vs. death
            birth_prob = (alpha / (alpha + delta * x_now)).magnitude
            if rng.random() < birth_prob:
                return [1 / vol]  # +1
            return [-1 / vol]  # -1
        return [0.0 / vol]

    def simulate(self, seed: int = 42) -> Timeseries:
        """Simulate a run."""
        return self.simulate_sde(seed=seed, noise_fun=self.noise)


class OnlyDilutionDetSystem(TranscriptionBCRN):
    """Stage 2: only the dilution term is deterministic."""

    params: TranscriptionParameters

    def __init__(self, params: TranscriptionParameters) -> None:
        super().__init__(
            init_state=[params.init_mRNA], time_horizon=params.time_horizon, volume=params.volume, dt=params.dt
        )
        self.params = params

    @property
    def stoichiometry(self) -> np.ndarray:
        return np.array([[1], [-1]])

    def reaction_rates(self, state: Sequence[Quantity]) -> list[Quantity]:
        """No growth drift."""
        return [0 * self.params.alpha, self.params.delta * state[0]]

    def noise(self, rng: np.random.Generator, t, y):
        # we can set the rate lower, since only birth is via noise
        alpha = self.params.alpha

        dt = self.dt
        vol = self.volume

        p_jump = (alpha * dt * vol).magnitude
        if rng.random() < p_jump:
            return [1.0 / vol]
        return [0.0 / vol]

    def simulate(self, seed: int = 42) -> Timeseries:
        """Simulate a run."""
        return self.simulate_sde(seed=seed, noise_fun=self.noise)


class GeometricNoiseSystem(TranscriptionBCRN):
    """Stage 3: drift= -delta*x plus geometric jumps at alpha."""

    params: TranscriptionParameters

    def __init__(self, params: TranscriptionParameters) -> None:
        super().__init__(
            init_state=[params.init_mRNA], time_horizon=params.time_horizon, volume=params.volume, dt=params.dt
        )
        self.params = params

    @property
    def stoichiometry(self) -> np.ndarray:
        return np.array([[1], [-1]])

    def reaction_rates(self, state: Sequence[Quantity]) -> list[Quantity]:
        """No growth drift."""
        return [0 * self.params.alpha, self.params.delta * state[0]]

    def noise(self, rng: np.random.Generator, t, y):
        geom_p = self.params.geom_p

        alpha = self.params.alpha * self.params.geom_alpha_factor

        dt = self.dt
        vol = self.volume

        p_jump = (alpha * dt * vol).magnitude
        if rng.random() < p_jump:
            jump_size = 0
            while True:
                if rng.random() < geom_p:
                    break
                jump_size += 1
            return [jump_size / vol]
        return [0.0 / vol]

    def simulate(self, seed: int = 42) -> Timeseries:
        """Simulate a run."""
        return self.simulate_sde(seed=seed, noise_fun=self.noise)


class ParetoNoiseSystem(TranscriptionBCRN):
    """Stage 4: drift -delta*x + Pareto jumps at alpha."""

    params: TranscriptionParameters

    def __init__(self, params: TranscriptionParameters) -> None:
        super().__init__(
            init_state=[params.init_mRNA], time_horizon=params.time_horizon, volume=params.volume, dt=params.dt
        )
        self.params = params

    @property
    def stoichiometry(self) -> np.ndarray:
        return np.array([[1], [-1]])

    def reaction_rates(self, state: Sequence[Quantity]) -> list[Quantity]:
        """No growth drift."""
        return [0 * self.params.alpha, self.params.delta * state[0]]

    def noise(self, rng: np.random.Generator, t, y):
        alpha = self.params.alpha * self.params.pareto_alpha_factor
        x_m = self.params.x_m
        a = self.params.pareto_alpha

        dt = self.dt
        vol = self.volume

        p_jump = (alpha * dt * vol).magnitude
        if rng.random() < p_jump:
            u = rng.random()
            return [x_m / (u ** (1.0 / a))]
        return [0.0 * x_m]

    def simulate(self, seed: int = 42) -> Timeseries:
        """Simulate a run."""
        return self.simulate_sde(seed=seed, noise_fun=self.noise)


class GaussianNoiseSystem(TranscriptionBCRN):
    """Stage 5: drift + Gaussian noise."""

    params: TranscriptionParameters

    def __init__(self, params: TranscriptionParameters) -> None:
        super().__init__(
            init_state=[params.init_mRNA], time_horizon=params.time_horizon, volume=params.volume, dt=params.dt
        )
        self.params = params

    @property
    def stoichiometry(self) -> np.ndarray:
        return np.array([[1], [-1]])

    def reaction_rates(self, state: Sequence[Quantity]) -> list[Quantity]:
        """Uses all the drifts."""
        return [self.params.alpha, self.params.delta * state[0]]

    def noise(self, rng: np.random.Generator, t, y):
        dW = np.sqrt(self.params.dt) * rng.normal()  # noqa: N806
        return [self.params.sigma * dW]

    def simulate(self, seed: int = 42) -> Timeseries:
        """Simulate a run."""
        return self.simulate_sde(seed=seed, noise_fun=self.noise)


def value_at_time(time: Quantity, ts: Timeseries) -> list[Quantity]:
    """Return value at time within ts.

    Returns the last value if the time is larger than the Timeseries ts.
    """
    for i, t in enumerate(ts[0]):
        if t >= time:
            return ts[1][i]
    return ts[1][-1]


def plot_dynamics(systems: dict[str, TranscriptionBCRN], fname: str = "dynamics.pdf", height: float = 4) -> None:
    """Plot time-series (all 6 stages)."""

    def only_rna(ts_in: Timeseries) -> Timeseries:
        states = [[state[0]] for state in ts_in.states]
        times = ts_in.times
        return Timeseries(times=times, states=states)

    logger.info("simulating dynamics for all systems...")
    tss = [only_rna(sys.simulate()) for sys in systems.values()]
    plot_timeseries(tss=tss, labels=list(systems.keys()), figsize=(6, height), drawstyle="steps-post", lw=1.0)
    plt.savefig(fname)
    logger.info(f"wrote {fname}")


def plot_histogram(
    systems: dict[str, TranscriptionBCRN], n: int, at_time: Quantity, fname: str = "hist.pdf", hist_height: float = 1.1
) -> None:
    """Plot the histograms of RNA counts for all the given systems."""
    datasets: dict[str, list[float]] = {}

    for sys_name, sys in systems.items():
        logger.info(f"simulating {sys_name}...")  # noqa: G004
        datasets[sys_name] = []
        for i in range(n):
            ts = sys.simulate(seed=i)
            value = value_at_time(time=at_time, ts=ts)
            rna = value[0]  # assumes RNA is always the 1st component in state
            datasets[sys_name] += [float(rna.to("1/femtoliter").magnitude)]

    # fig, ax = plt.subplots(figsize=(8, 6))
    # for sys_name, rnas in datasets.items():
    #     ax.hist(rnas, bins=range(0, 40, 1), alpha=0.4, density=True, label=sys_name)

    keys = list(datasets.keys())
    plots_n = len(keys)

    _, axes = plt.subplots(nrows=plots_n, sharex=True, figsize=(4, hist_height * plots_n))

    if plots_n == 1:
        axes = [axes]  # Ensure axes is always a list

    for ax, key in zip(axes, keys, strict=False):
        ax.hist(datasets[key], bins=range(0, 30, 1), alpha=0.7, density=True)
        ax.set_ylabel("Density")
        ax.set_title(f"{key}")

    axes[-1].set_xlabel("mRNA (1/fL)")
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(fname)
    logger.info(f"wrote {fname}")


def example_onestate():
    u = UnitRegistry()

    init_mRNA = 0.0 / u.femtoliter  # noqa: N806
    volume = 1.0 * u.femtoliter
    time_horizon = 60.0 * u.minute
    dt = 0.1 * u.minute

    # Rate constants
    alpha = 1 / u.minute / u.femtoliter
    delta = 0.1 / u.minute

    on_to_off = 0.1 / u.minute  # / u.femtoliter
    off_to_on = 0.1 / u.minute  # / u.femtoliter

    # Noise parameters
    pareto_alpha = 1.8 * u.dimensionless
    pareto_alpha_factor = 0.5 * u.dimensionless
    geom_alpha_factor = 0.5 * u.dimensionless
    geom_p = 1 / 2.8 * u.dimensionless
    x_m = 1.0 / u.femtoliter
    sigma = 1.0 / u.femtoliter / (u.minute**0.5)

    params = TranscriptionParameters(
        alpha=alpha,
        delta=delta,
        on_to_off=on_to_off,
        off_to_on=off_to_on,
        init_mRNA=init_mRNA,
        volume=volume,
        time_horizon=time_horizon,
        dt=dt,
        sigma=sigma,
        pareto_alpha=pareto_alpha,
        geom_alpha_factor=geom_alpha_factor,
        pareto_alpha_factor=pareto_alpha_factor,
        x_m=x_m,
        geom_p=geom_p,
    )

    systems: dict[str, TranscriptionBCRN] = {}
    systems["Markov"] = OriginalSystem(params)
    systems["SDE (equiv. Markov)"] = NoDetSystem(params)
    systems["Jump+1 & const"] = OnlyDilutionDetSystem(params)
    systems["Geom"] = GeometricNoiseSystem(params)
    systems["Pareto"] = ParetoNoiseSystem(params)
    systems["Gauss"] = GaussianNoiseSystem(params)

    # plot time-series
    plot_dynamics(systems, fname="dynamics_onestate.pdf", height=6)
    # plot histograms
    plot_histogram(systems, n=500, at_time=time_horizon, fname="hist_onestate.pdf", hist_height=0.8)


def example_twostate():
    u = UnitRegistry()

    init_mRNA = 0.0 / u.femtoliter  # noqa: N806
    volume = 1.0 * u.femtoliter
    time_horizon = 60.0 * u.minute
    dt = 0.1 * u.minute

    # Rate constants
    alpha = 1 / u.minute / u.femtoliter
    delta = 0.1 / u.minute

    on_to_off = 0.1 / u.minute
    off_to_on = 0.1 / u.minute

    # Noise parameters
    pareto_alpha = 1.7 * u.dimensionless
    pareto_alpha_factor = 0.3 * u.dimensionless
    geom_alpha_factor = 0.3 * u.dimensionless
    geom_p = 1 / 3 * u.dimensionless
    x_m = 0.9 / u.femtoliter
    sigma = 1.0 / u.femtoliter / (u.minute**0.5)

    params = TranscriptionParameters(
        alpha=alpha,
        delta=delta,
        on_to_off=on_to_off,
        off_to_on=off_to_on,
        init_mRNA=init_mRNA,
        volume=volume,
        time_horizon=time_horizon,
        dt=dt,
        sigma=sigma,
        pareto_alpha=pareto_alpha,
        pareto_alpha_factor=pareto_alpha_factor,
        geom_alpha_factor=geom_alpha_factor,
        x_m=x_m,
        geom_p=geom_p,
    )

    systems: dict[str, TranscriptionBCRN] = {}
    systems["Markov (2 state)"] = TwoStateSystem(params)
    systems["Geom"] = GeometricNoiseSystem(params)
    systems["Pareto"] = ParetoNoiseSystem(params)

    gaussparams = params._replace(alpha=alpha * (off_to_on / (off_to_on + on_to_off)))

    systems["Gauss"] = GaussianNoiseSystem(gaussparams)

    # plot time-series
    plot_dynamics(systems, fname="dynamics_twostate.pdf", height=4)
    # plot histograms
    plot_histogram(systems, n=500, at_time=time_horizon, fname="hist_twostate.pdf", hist_height=0.8)


def main():
    example_onestate()
    example_twostate()
    plt.show()


if __name__ == "__main__":
    main()
