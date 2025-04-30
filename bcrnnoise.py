"""bcrnnoise: simulate Bio-Chemical Reaction Networks with noise.

Supports simulation via ODEs, Markov chains, and SDEs with Gaussian and custom noise terms.

"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import NamedTuple, cast

import matplotlib.pyplot as plt
import numpy as np
from pint import Quantity, Unit
from solve_ivp_pint import solve_ivp


def dimensionalize(seq: Sequence[float], unit: Unit) -> list[Quantity]:
    """Apply unit to a sequence of numbers.

    Multiplies the unit with the numbers.

    Args:
        seq: the sequence of numbers
        unit: the unit to be applied

    Returns:
        the sequence of type pint.Quantity

    """
    return [val * unit for val in seq]


def get_unit(qty: Quantity | Sequence[Quantity]) -> Unit:
    """Extract a representative unit from a pint.Quantity or a sequence of pint.Quantity.

    Args:
        qty: the quantity or sequence of quantities

    Returns:
        the unit of the quantity, or that of the first quantity of the sequence

    Raises:
        TypeError: never

    """
    if not isinstance(qty, Quantity):
        qty = next(iter(qty))

    if isinstance(qty, Quantity):
        ureg = qty._REGISTRY  # noqa: SLF001
        return ureg.Unit(str(qty.units))
    raise TypeError


def undimensionalize(seq: Sequence[Quantity], unit: Unit | None = None) -> list[float]:
    """Convert the pint.Quantity of a sequence to a fixed unit.

    Args:
        seq: the sequence of quantities
        unit: If not None, the unit to be converted to. If None, the unit will be extracted via get_unit.

    Returns:
        the sequence of magnitudes after conversion

    """
    if unit is None:
        unit = get_unit(seq)

    return [qty.to(unit).magnitude for qty in seq]


class Timeseries(NamedTuple):
    """Time series for a BCRN model.

    Attributes:
        times: list of times (dimension: time)
        states: list of states at the above times (dimension of each entry: 1/ volume)

    """

    times: list[Quantity]
    states: list[list[Quantity]]


class BCRN(ABC):
    """Represents a BCRN system."""

    init_state: list[Quantity]
    time_horizon: Quantity
    volume: Quantity
    dt: Quantity

    def __init__(self, init_state: Sequence[Quantity], time_horizon: Quantity, volume: Quantity, dt: Quantity) -> None:
        """Initialize an abstract BCRN system.

        Args:
            init_state: The initial system state (dimension of each entry: 1 / volume).
            time_horizon: The time horizon of the system (dimension: time).
            volume: The total volume of the system (dimension: volume).
            dt: The time delta used for Euler-Maruyama iteration (dimension: time).
        """
        self.init_state = list(init_state)
        self.time_horizon = time_horizon
        self.volume = volume
        self.dt = dt

    @property
    @abstractmethod
    def stoichiometry(self) -> np.ndarray:
        """The stoichiometry matrix of the system."""

    @abstractmethod
    def reaction_rates(self, state: Sequence[Quantity]) -> list[Quantity]:
        """Calculate the rates of each reaction for a given system state.

        Args:
            state: the system state (dimension of each entry: 1 / volume)

        Returns:
            the reaction rates (dimension of each entry: 1 / volume / time)

        """

    def ivp_rhs(self, t: Quantity, y: Sequence[Quantity]) -> list[Quantity]:  # noqa: ARG002
        """Calculate the right-hand side of the ODE of the initial value problem.

        Args:
            t: the current time (dimension: time)
            y: the current state (dimension of each entry: 1 / volume)

        Returns:
            the time derivative of the state (dimension of each entry: 1 / volume / time)

        """
        stoichiometry_matrix = np.matrix(self.stoichiometry).T
        rates_matrix = np.matrix(self.reaction_rates(y), dtype=Quantity).T
        return cast("list[Quantity]", (stoichiometry_matrix * rates_matrix).tolist()[0])

    def simulate_ode(self) -> Timeseries:
        """Simulate the ODE kinetics of the BCRN.

        Returns:
            the simulated time series

        """
        sol = solve_ivp(
            fun=self.ivp_rhs,
            t_span=[0 * self.time_horizon, self.time_horizon],
            y0=self.init_state,
        )
        time_series_of_lists = [list(state) for state in zip(*sol.y, strict=False)]
        return Timeseries(times=sol.t, states=time_series_of_lists)

    # Gillespie's algorithm with units
    def gillespie(
        self, rng: np.random.Generator, initial_count_state: Sequence[int], max_time: Quantity
    ) -> list[tuple[Quantity, Sequence[int]]]:
        """Execute Gillespie's algorithm.

        Args:
            rng: the np.random.Generator used for random choices
            initial_count_state: the initial state of the system,
                expressed in counts per total volume (dimension: dimensionless)
            max_time: the time horizon of the simulation (dimension: time)

        Returns:
            a list of tuples (time, count_state) for each state transition,
                where time is the time of the transition (dimension: time) and
                count_state is a list of counts per total volume for each species
                (dimension of each entry: dimensionless)

        """
        # Convert the initial state to quantities with units
        count_state = initial_count_state
        time = cast("Quantity", 0.0 * max_time)
        history = [(time, count_state)]

        while time < max_time:
            concentration_state = [val / self.volume for val in count_state]
            rates = [rate * self.volume for rate in self.reaction_rates(concentration_state)]
            total_rate = sum(rates)

            if not isinstance(total_rate, Quantity):  # if rates is empty
                total_rate = 0.0 / self.volume / self.time_horizon

            if total_rate.magnitude == 0:
                break

            # Draw two random numbers
            r1, r2 = rng.random(2)

            # Time increment
            tau = (1 / total_rate) * np.log(1 / r1)
            time += tau

            # Which reaction occurs
            reaction_probabilities = [rate / total_rate for rate in rates]
            cumulative_probabilities = np.cumsum(reaction_probabilities)
            reaction_idx = np.searchsorted(cumulative_probabilities, r2)

            # Update the state according to the selected reaction
            count_state = [s + (self.stoichiometry[reaction_idx][i]) for i, s in enumerate(count_state)]

            history.append((time, count_state))

        return history

    def simulate_markov_chain(self, seed: int = 42) -> Timeseries:
        """Simulate the Markov chain kinetics of the BCRN.

        Args:
            seed: the seed for the random number generator used for random choices

        Returns:
            the simulated time series

        """
        initial_count_state = [int(val * self.volume) for val in self.init_state]

        rng = np.random.default_rng(seed=seed)

        history = self.gillespie(
            rng,
            initial_count_state=initial_count_state,
            max_time=self.time_horizon,
        )

        times = [item[0] for item in history]
        states = [[val / self.volume for val in item[1]] for item in history]

        return Timeseries(times=times, states=states)

    def simulate_sde(
        self, noise_fun: Callable[[np.random.Generator, Quantity, Sequence[Quantity]], list[Quantity]], seed: int = 42
    ) -> Timeseries:
        """Simulate the SDE kintetics of the BCRN with a given noise term using Euler-Maruyama discretization.

        Args:
            noise_fun: the function calculating the noise term, takes
                a np.random.Generator for its random choices,
                the current time (dimension: time), and
                the current state (dimension of each entry: 1 / volume), and
                returns a noise term for each species (dimension: 1 / volume)
            seed: the seed for the random number generator used for random choices

        Returns:
            the simulated time series

        """
        n_steps = int(np.floor(self.time_horizon / self.dt))
        times = [self.time_horizon * k / n_steps for k in range(n_steps + 1)]
        states: list[list[Quantity]] = [self.init_state] * (n_steps + 1)
        states[0] = self.init_state

        rng = np.random.default_rng(seed=seed)

        for k in range(n_steps):
            # Deterministic drift:
            drift: list[Quantity] = self.ivp_rhs(t=times[k], y=states[k])
            # Gaussian noise increment ~ N(0, sigma*sqrt(dt))
            state_new: list[Quantity] = cast(
                "list[Quantity]",
                [
                    states[k][i] + drift[i] * self.dt + noise_fun(rng, times[k], states[k])[i]
                    for i in range(len(self.init_state))
                ],
            )

            # Protein should remain nonnegative (if desired, we can clamp):
            for i in range(len(self.init_state)):
                if state_new[i].magnitude < 0:
                    state_new[i] *= 0.0
            states[k + 1] = state_new

        return Timeseries(times=times, states=states)

    def plot_timeseries(self, tss: list[Timeseries], labels: list[str]) -> None:
        """Plot a list of time series.

        Params:
            tss: the time series to be plotted
            labels: their respective labels

        """
        for i in range(len(self.init_state)):
            plt.figure(figsize=(6, 4))
            for k, ts in enumerate(tss):
                times = undimensionalize(ts.times, get_unit(self.time_horizon))
                vals = undimensionalize([s[i] for s in ts.states], get_unit(1 / self.volume))
                plt.plot(times, vals, label=labels[k])
            plt.xlabel(f"Time ({self.time_horizon.units})")
            plt.ylabel(f"concentration #{i} ({1 / self.volume.units})")  # TODO: useful names for concentrations
            plt.legend()
            plt.tight_layout()
        plt.show()
