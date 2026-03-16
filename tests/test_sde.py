"""Tests for SDE simulation via simulate_sde (Euler-Maruyama)."""

import pytest
from helpers import PureDecaySystem, VanillaSystem, make_vanilla
from pint import UnitRegistry

from bcrnnoise import Timeseries


def test_simulate_sde_returns_timeseries(vanilla_system: VanillaSystem, ureg: UnitRegistry) -> None:
    def zero_noise(_rng, _t, _state):
        return [0.0 / ureg.femtoliter]

    assert isinstance(vanilla_system.simulate_sde(noise_fun=zero_noise), Timeseries)


def test_simulate_sde_n_steps(vanilla_system: VanillaSystem, ureg: UnitRegistry) -> None:
    # time_horizon=50 min, dt=0.1 min → 500 steps → 501 time points
    def zero_noise(_rng, _t, _state):
        return [0.0 / ureg.femtoliter]

    ts = vanilla_system.simulate_sde(noise_fun=zero_noise)
    assert len(ts.times) == 501
    assert len(ts.states) == 501


def test_simulate_sde_initial_state_preserved(vanilla_system: VanillaSystem, ureg: UnitRegistry) -> None:
    def zero_noise(_rng, _t, _state):
        return [0.0 / ureg.femtoliter]

    ts = vanilla_system.simulate_sde(noise_fun=zero_noise)
    assert ts.states[0][0].to(1 / ureg.femtoliter).magnitude == pytest.approx(0.0)


def test_simulate_sde_seed_reproducibility(vanilla_system: VanillaSystem, ureg: UnitRegistry) -> None:
    sigma = 0.5 / ureg.femtoliter

    def gaussian_noise(rng, _t, _state):
        return [sigma * rng.normal()]

    ts1 = vanilla_system.simulate_sde(noise_fun=gaussian_noise, seed=42)
    ts2 = vanilla_system.simulate_sde(noise_fun=gaussian_noise, seed=42)
    assert ts1.states[-1][0].to(1 / ureg.femtoliter).magnitude == pytest.approx(
        ts2.states[-1][0].to(1 / ureg.femtoliter).magnitude
    )


def test_simulate_sde_different_seeds_differ(vanilla_system: VanillaSystem, ureg: UnitRegistry) -> None:
    sigma = 0.5 / ureg.femtoliter

    def gaussian_noise(rng, _t, _state):
        return [sigma * rng.normal()]

    ts1 = vanilla_system.simulate_sde(noise_fun=gaussian_noise, seed=0)
    ts2 = vanilla_system.simulate_sde(noise_fun=gaussian_noise, seed=1)
    assert ts1.states[-1][0].to(1 / ureg.femtoliter).magnitude != pytest.approx(
        ts2.states[-1][0].to(1 / ureg.femtoliter).magnitude
    )


def test_simulate_sde_clamps_states_nonneg(pure_decay_system: PureDecaySystem, ureg: UnitRegistry) -> None:
    # Huge negative noise drives state below zero; clamping must keep it at >= 0
    def huge_neg_noise(_rng, _t, _state):
        return [-1000.0 / ureg.femtoliter]

    ts = pure_decay_system.simulate_sde(noise_fun=huge_neg_noise)
    for state in ts.states:
        assert state[0].to(1 / ureg.femtoliter).magnitude >= 0.0


def test_simulate_sde_zero_noise_converges_to_steady_state(ureg: UnitRegistry) -> None:
    # Zero-noise SDE = Euler ODE; alpha=1/fL/min, delta=0.1/min → steady state = 10/fL
    sys = make_vanilla(ureg, time_horizon_min=100.0)

    def zero_noise(_rng, _t, _state):
        return [0.0 / ureg.femtoliter]

    ts = sys.simulate_sde(noise_fun=zero_noise)
    final_conc = ts.states[-1][0].to(1 / ureg.femtoliter).magnitude
    assert final_conc == pytest.approx(10.0, abs=0.1)
