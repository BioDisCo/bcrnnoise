"""Concrete BCRN subclasses used as test doubles across the test suite."""

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from pint import Quantity, UnitRegistry

from bcrnnoise import BCRN


class VanillaParams(NamedTuple):
    """Parameters for the 0 -> X [alpha], X -> 0 [delta] birth-death model."""

    alpha: Quantity
    delta: Quantity
    init_mrna: Quantity
    volume: Quantity
    time_horizon: Quantity
    dt: Quantity


class VanillaSystem(BCRN):
    """Birth-death transcription model: 0 -> X [alpha], X -> 0 [delta].

    Steady-state concentration: alpha / delta.
    """

    params: VanillaParams

    def __init__(self, params: VanillaParams) -> None:
        """Initialize with kinetic parameters and simulation settings."""
        super().__init__(
            init_state=[params.init_mrna],
            time_horizon=params.time_horizon,
            volume=params.volume,
            dt=params.dt,
        )
        self.params = params

    @property
    def stoichiometry(self) -> np.ndarray:
        """Stoichiometry matrix: production adds 1, degradation removes 1."""
        return np.array([[1], [-1]])

    def reaction_rates(self, state: Sequence[Quantity]) -> list[Quantity]:
        """Return [alpha, delta * X] as production and degradation rates."""
        return [self.params.alpha, self.params.delta * state[0]]


class PureDecaySystem(BCRN):
    """Single-species degradation model: X -> 0 [delta]. No production."""

    _delta: Quantity

    def __init__(
        self,
        delta: Quantity,
        init: Quantity,
        volume: Quantity,
        time_horizon: Quantity,
        dt: Quantity,
    ) -> None:
        """Initialize with degradation rate and initial concentration."""
        super().__init__(init_state=[init], time_horizon=time_horizon, volume=volume, dt=dt)
        self._delta = delta

    @property
    def stoichiometry(self) -> np.ndarray:
        """Stoichiometry matrix: degradation removes 1."""
        return np.array([[-1]])

    def reaction_rates(self, state: Sequence[Quantity]) -> list[Quantity]:
        """Return [delta * X] as the single degradation rate."""
        return [self._delta * state[0]]


def make_vanilla(ureg: UnitRegistry, *, time_horizon_min: float = 50.0) -> VanillaSystem:
    """Build a VanillaSystem with alpha=1/fL/min, delta=0.1/min, steady state = 10/fL."""
    return VanillaSystem(
        VanillaParams(
            alpha=1.0 / ureg.minute / ureg.femtoliter,
            delta=0.1 / ureg.minute,
            init_mrna=0.0 / ureg.femtoliter,
            volume=1.0 * ureg.femtoliter,
            time_horizon=time_horizon_min * ureg.minute,
            dt=0.1 * ureg.minute,
        )
    )


def make_pure_decay(ureg: UnitRegistry) -> PureDecaySystem:
    """Build a PureDecaySystem with delta=1/min, init=10/fL."""
    return PureDecaySystem(
        delta=1.0 / ureg.minute,
        init=10.0 / ureg.femtoliter,
        volume=1.0 * ureg.femtoliter,
        time_horizon=5.0 * ureg.minute,
        dt=0.01 * ureg.minute,
    )
