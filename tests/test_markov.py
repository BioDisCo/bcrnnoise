"""Tests for Markov chain simulation via simulate_markov_chain and gillespie."""

import itertools

import numpy as np
import pytest
from helpers import PureDecaySystem, VanillaSystem
from pint import UnitRegistry

from bcrnnoise import Timeseries


def test_simulate_markov_returns_timeseries(vanilla_system: VanillaSystem) -> None:
    assert isinstance(vanilla_system.simulate_markov_chain(seed=0), Timeseries)


def test_simulate_markov_initial_state(vanilla_system: VanillaSystem, ureg: UnitRegistry) -> None:
    # init_mRNA=0/fL → count=0 → state[0]=0/fL
    ts = vanilla_system.simulate_markov_chain(seed=0)
    assert ts.states[0][0].to(1 / ureg.femtoliter).magnitude == pytest.approx(0.0)


def test_simulate_markov_times_nondecreasing(vanilla_system: VanillaSystem, ureg: UnitRegistry) -> None:
    ts = vanilla_system.simulate_markov_chain(seed=0)
    times_min = [t.to(ureg.minute).magnitude for t in ts.times]
    assert all(t1 <= t2 for t1, t2 in itertools.pairwise(times_min))


def test_simulate_markov_states_nonneg(vanilla_system: VanillaSystem, ureg: UnitRegistry) -> None:
    ts = vanilla_system.simulate_markov_chain(seed=0)
    for state in ts.states:
        assert state[0].to(1 / ureg.femtoliter).magnitude >= 0.0


def test_simulate_markov_seed_reproducibility(vanilla_system: VanillaSystem, ureg: UnitRegistry) -> None:
    ts1 = vanilla_system.simulate_markov_chain(seed=42)
    ts2 = vanilla_system.simulate_markov_chain(seed=42)
    assert len(ts1.times) == len(ts2.times)
    assert ts1.times[-1].to(ureg.minute).magnitude == pytest.approx(ts2.times[-1].to(ureg.minute).magnitude)


def test_simulate_markov_different_seeds_differ(vanilla_system: VanillaSystem) -> None:
    ts1 = vanilla_system.simulate_markov_chain(seed=0)
    ts2 = vanilla_system.simulate_markov_chain(seed=1)
    assert len(ts1.times) != len(ts2.times) or ts1.states != ts2.states


def test_gillespie_zero_count_stops_immediately(pure_decay_system: PureDecaySystem, ureg: UnitRegistry) -> None:
    # No production: with zero initial count, total_rate=0 → terminates after recording initial state
    rng = np.random.default_rng(seed=0)
    history = pure_decay_system.gillespie(rng, initial_count_state=[0], max_time=10.0 * ureg.minute)
    assert len(history) == 1
    assert history[0][1] == [0]


def test_gillespie_count_stays_nonneg(pure_decay_system: PureDecaySystem, ureg: UnitRegistry) -> None:
    rng = np.random.default_rng(seed=0)
    history = pure_decay_system.gillespie(rng, initial_count_state=[20], max_time=10.0 * ureg.minute)
    assert all(count >= 0 for _, state in history for count in state)
