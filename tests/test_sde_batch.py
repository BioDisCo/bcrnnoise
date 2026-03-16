"""Tests for batch SDE simulation via simulate_sde_batch."""

import numpy as np
import pytest
from helpers import PureDecaySystem, VanillaSystem, make_vanilla
from pint import UnitRegistry

from bcrnnoise import BatchTimeseries, Timeseries


def test_simulate_sde_batch_returns_batch_timeseries(vanilla_system: VanillaSystem) -> None:
    def zero_noise(_rng, _t, states):
        return [s * 0.0 for s in states]

    assert isinstance(vanilla_system.simulate_sde_batch(noise_fun=zero_noise, n=3), BatchTimeseries)


def test_simulate_sde_batch_n_steps(vanilla_system: VanillaSystem) -> None:
    # time_horizon=50 min, dt=0.1 min → 500 steps → 501 time points
    def zero_noise(_rng, _t, states):
        return [s * 0.0 for s in states]

    batch = vanilla_system.simulate_sde_batch(noise_fun=zero_noise, n=3)
    assert len(batch.times) == 501
    assert len(batch.states) == 501


def test_simulate_sde_batch_n_attr(vanilla_system: VanillaSystem) -> None:
    def zero_noise(_rng, _t, states):
        return [s * 0.0 for s in states]

    batch = vanilla_system.simulate_sde_batch(noise_fun=zero_noise, n=7)
    assert batch.n == 7


def test_simulate_sde_batch_state_shape(vanilla_system: VanillaSystem) -> None:
    n = 11

    def zero_noise(_rng, _t, states):
        return [s * 0.0 for s in states]

    batch = vanilla_system.simulate_sde_batch(noise_fun=zero_noise, n=n)
    assert batch.states[0][0].magnitude.shape == (n,)


def test_simulate_sde_batch_initial_states(vanilla_system: VanillaSystem) -> None:
    def zero_noise(_rng, _t, states):
        return [s * 0.0 for s in states]

    batch = vanilla_system.simulate_sde_batch(noise_fun=zero_noise, n=5)
    np.testing.assert_allclose(batch.states[0][0].magnitude, 0.0)


def test_simulate_sde_batch_seed_reproducibility(vanilla_system: VanillaSystem, ureg: UnitRegistry) -> None:
    sigma = 0.5 / ureg.femtoliter

    def gaussian_noise(rng, _t, states):
        return [sigma * rng.normal(size=states[0].magnitude.shape[0])]

    b1 = vanilla_system.simulate_sde_batch(noise_fun=gaussian_noise, n=10, seed=42)
    b2 = vanilla_system.simulate_sde_batch(noise_fun=gaussian_noise, n=10, seed=42)
    np.testing.assert_array_equal(b1.states[-1][0].magnitude, b2.states[-1][0].magnitude)


def test_simulate_sde_batch_different_seeds_differ(vanilla_system: VanillaSystem, ureg: UnitRegistry) -> None:
    sigma = 0.5 / ureg.femtoliter

    def gaussian_noise(rng, _t, states):
        return [sigma * rng.normal(size=states[0].magnitude.shape[0])]

    b1 = vanilla_system.simulate_sde_batch(noise_fun=gaussian_noise, n=10, seed=0)
    b2 = vanilla_system.simulate_sde_batch(noise_fun=gaussian_noise, n=10, seed=1)
    assert not np.array_equal(b1.states[-1][0].magnitude, b2.states[-1][0].magnitude)


def test_simulate_sde_batch_clamps_states_nonneg(pure_decay_system: PureDecaySystem, ureg: UnitRegistry) -> None:
    def huge_neg_noise(_rng, _t, states):
        n = states[0].magnitude.shape[0]
        return [-1000.0 / ureg.femtoliter * np.ones(n)]

    batch = pure_decay_system.simulate_sde_batch(noise_fun=huge_neg_noise, n=5)
    for step in batch.states:
        assert np.all(step[0].magnitude >= 0.0)


def test_simulate_sde_batch_zero_noise_converges_to_steady_state(ureg: UnitRegistry) -> None:
    # Zero-noise batch = Euler ODE over N identical trajectories → all should reach 10/fL
    sys = make_vanilla(ureg, time_horizon_min=100.0)

    def zero_noise(_rng, _t, states):
        return [s * 0.0 for s in states]

    batch = sys.simulate_sde_batch(noise_fun=zero_noise, n=5)
    final_concs = batch.states[-1][0].magnitude
    assert final_concs == pytest.approx(np.full(5, 10.0), abs=0.1)


def test_simulate_sde_batch_trajectory_returns_timeseries(vanilla_system: VanillaSystem) -> None:
    def zero_noise(_rng, _t, states):
        return [s * 0.0 for s in states]

    batch = vanilla_system.simulate_sde_batch(noise_fun=zero_noise, n=4)
    ts = batch.trajectory(2)
    assert isinstance(ts, Timeseries)
    assert len(ts.states) == len(batch.states)
    assert np.ndim(ts.states[0][0].magnitude) == 0  # scalar per step
