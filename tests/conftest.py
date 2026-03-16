"""Shared pytest fixtures for the bcrnnoise test suite."""

import pytest
from helpers import PureDecaySystem, VanillaSystem, make_pure_decay, make_vanilla
from pint import UnitRegistry


@pytest.fixture(scope="session")
def ureg() -> UnitRegistry:
    return UnitRegistry()


@pytest.fixture
def vanilla_system(ureg: UnitRegistry) -> VanillaSystem:
    return make_vanilla(ureg)


@pytest.fixture
def pure_decay_system(ureg: UnitRegistry) -> PureDecaySystem:
    return make_pure_decay(ureg)
