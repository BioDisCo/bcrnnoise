"""Tests for ODE simulation and ivp_rhs."""

import pytest
from helpers import VanillaSystem, make_vanilla
from pint import UnitRegistry

from bcrnnoise import Timeseries


def test_simulate_ode_returns_timeseries(vanilla_system: VanillaSystem) -> None:
    assert isinstance(vanilla_system.simulate_ode(), Timeseries)


def test_simulate_ode_times_start_at_zero(vanilla_system: VanillaSystem, ureg: UnitRegistry) -> None:
    ts = vanilla_system.simulate_ode()
    assert ts.times[0].to(ureg.minute).magnitude == pytest.approx(0.0)


def test_simulate_ode_times_end_at_horizon(vanilla_system: VanillaSystem, ureg: UnitRegistry) -> None:
    ts = vanilla_system.simulate_ode()
    horizon = vanilla_system.time_horizon.to(ureg.minute).magnitude
    assert ts.times[-1].to(ureg.minute).magnitude == pytest.approx(horizon, rel=1e-3)


def test_simulate_ode_state_length(vanilla_system: VanillaSystem) -> None:
    ts = vanilla_system.simulate_ode()
    assert all(len(state) == 1 for state in ts.states)


def test_simulate_ode_states_nonneg(vanilla_system: VanillaSystem, ureg: UnitRegistry) -> None:
    ts = vanilla_system.simulate_ode()
    for state in ts.states:
        assert state[0].to(1 / ureg.femtoliter).magnitude >= 0.0


def test_simulate_ode_reaches_steady_state(ureg: UnitRegistry) -> None:
    # alpha=1/fL/min, delta=0.1/min → steady state = 10/fL; 5 time constants = 50 min
    ts = make_vanilla(ureg, time_horizon_min=50.0).simulate_ode()
    final_conc = ts.states[-1][0].to(1 / ureg.femtoliter).magnitude
    assert final_conc == pytest.approx(10.0, rel=0.01)


def test_simulate_ode_is_deterministic(vanilla_system: VanillaSystem, ureg: UnitRegistry) -> None:
    ts1 = vanilla_system.simulate_ode()
    ts2 = vanilla_system.simulate_ode()
    assert ts1.states[-1][0].to(1 / ureg.femtoliter).magnitude == pytest.approx(
        ts2.states[-1][0].to(1 / ureg.femtoliter).magnitude
    )


def test_ivp_rhs_at_zero_state(vanilla_system: VanillaSystem, ureg: UnitRegistry) -> None:
    # At X=0, only production fires: dy/dt = alpha = 1/fL/min
    dydt = vanilla_system.ivp_rhs(t=0 * ureg.minute, y=[0.0 / ureg.femtoliter])
    assert dydt[0].to(1 / ureg.femtoliter / ureg.minute).magnitude == pytest.approx(1.0)


def test_ivp_rhs_at_steady_state(vanilla_system: VanillaSystem, ureg: UnitRegistry) -> None:
    # At X = alpha/delta = 10/fL, production and degradation balance: dy/dt = 0
    dydt = vanilla_system.ivp_rhs(t=0 * ureg.minute, y=[10.0 / ureg.femtoliter])
    assert dydt[0].to(1 / ureg.femtoliter / ureg.minute).magnitude == pytest.approx(0.0, abs=1e-10)
