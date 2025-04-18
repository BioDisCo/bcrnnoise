from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from pint import Quantity, UnitRegistry

from bcrnnoise import BCRN


class VanillaParameters(NamedTuple):
    """Parameters for the vanilla (steadily on promoter) transcription model.

    The reactions are the following:
    0 -> mRNA [alpha]
    mRNA -> 0 [delta]

    Attributes:
        alpha: rate constant for transcription reaction (dimension: 1 / time / volume)
        delta: rate constant for mRNA degradation reaction (dimension: 1 / time)
        init_mRNA: initial concentration of mRNA (dimension: 1 / volume)
        volume: total volume of the system (dimension: volume)
        time_horizon: The maximum time of the time series in the system (dimension: time)
        dt: time delta used for Euleur-Maruyama iteration (dimension: time)
        sigma: amplitude of Gaussian noise (dimension: 1 / volume / sqrt(time))
        lambda_burst: rate of Pareto bursts (dimension: 1 / time)
        pareto_alpha: heavy-tail shape parameter (1 < alpha < 2) (dimension: dimensionless)
        x_m: minimal jump size for Pareto (dimension: 1 / volume)
    """

    alpha: Quantity
    delta: Quantity
    init_mRNA: Quantity
    volume: Quantity
    time_horizon: Quantity
    dt: Quantity
    sigma: Quantity
    lambda_burst: Quantity
    pareto_alpha: Quantity
    x_m: Quantity


class VanillaTimeseries(NamedTuple):
    """Time series for the vanilla (steadily on promoter) transcription model.

    Attributes:
        time: List of times (dimension: time).
        mRNA: List of mRNA concentrations at the above times (dimension: 1/ volume)
    """

    time: list[Quantity]
    mRNA: list[Quantity]


class VanillaSystem(BCRN):
    """Represents the vanilla (steadily on promoter) transcription model.

    The reactions are the following:
    0 -> mRNA [alpha]
    mRNA -> 0 [delta]
    """

    params: VanillaParameters

    def __init__(self, params: VanillaParameters) -> None:
        """Initialize the system."""
        self.params = params

    @property
    def time_horizon(self) -> Quantity:
        return self.params.time_horizon

    @property
    def volume(self) -> Quantity:
        """The total volume of the system (dimension: volume)."""
        return self.params.volume

    @property
    def init_state(self) -> list[Quantity]:
        """The initial system state (dimension of each entry: 1 / volume)."""
        return [self.params.init_mRNA]

    @property
    def dt(self) -> Quantity:
        """The time delta used for Euleur-Maruyama iteration (dimension: time)."""
        return self.params.dt

    @property
    def stoichiometry(self) -> np.ndarray:
        """The stoichiometry matrix of the system."""
        return np.array(
            [
                [1],
                [-1],
            ]
        )

    def reaction_rates(self, state: Sequence[Quantity]) -> list[Quantity]:
        """Calculate the rates of each reaction for a given system state.

        Args:
            state: the system state (dimension of each entry: 1 / volume)

        Returns:
            the reaction rates (dimension of each entry: 1 / volume / time)

        """
        return [self.params.alpha, self.params.delta * state[0]]

    def gaussian_noise_fun(self, rng: np.random.Generator, t, y):
        dW = np.sqrt(self.params.dt) * rng.normal()
        return [self.params.sigma * dW]

    def pareto_noise_fun(self, rng: np.random.Generator, t, y):
        if rng.random() < self.params.lambda_burst * self.params.dt:
            u = rng.random()
            return [self.params.x_m / (u ** (1.0 / self.params.pareto_alpha))]
        return [0.0 * self.params.init_mRNA]

    def plot_all(self) -> None:
        """Simulate and plot all time series."""
        self.plot_timeseries(
            tss=[
                self.simulate_ode(),
                self.simulate_markov_chain(),
                self.simulate_sde(noise_fun=self.gaussian_noise_fun),
                self.simulate_sde(noise_fun=self.pareto_noise_fun),
            ],
            labels=["ODE", "Markov chain", "SDE Gaussian", "SDE Pareto"],
        )


def main() -> None:
    u = UnitRegistry()

    alpha = 1.0 / u.minute / u.femtoliter
    delta = 0.1 / u.minute
    init_mRNA = 0.0 / u.femtoliter
    volume = 1.0 * u.femtoliter
    time_horizon = 2000.0 * u.minute
    dt = 0.01 * u.minute

    sigma = 5.0 / u.femtoliter / u.minute**0.5

    lambda_burst = 0.5 / u.minute
    pareto_alpha = 1.5 * u.dimensionless
    x_m = 1.0 / u.femtoliter

    params = VanillaParameters(
        alpha=alpha,
        delta=delta,
        init_mRNA=init_mRNA,
        volume=volume,
        time_horizon=time_horizon,
        dt=dt,
        sigma=sigma,
        lambda_burst=lambda_burst,
        pareto_alpha=pareto_alpha,
        x_m=x_m,
    )
    sys = VanillaSystem(params)
    sys.plot_all()


if __name__ == "__main__":
    main()
