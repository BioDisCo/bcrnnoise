"""Tests for utility functions: dimensionalize, undimensionalize, get_unit."""

import pytest
from pint import UnitRegistry

from bcrnnoise import dimensionalize, get_unit, undimensionalize


def test_dimensionalize_applies_unit(ureg: UnitRegistry) -> None:
    result = dimensionalize([1.0, 2.0, 3.0], ureg.femtoliter**-1)
    assert len(result) == 3
    assert result[0].to(1 / ureg.femtoliter).magnitude == pytest.approx(1.0)
    assert result[2].to(1 / ureg.femtoliter).magnitude == pytest.approx(3.0)


def test_dimensionalize_empty(ureg: UnitRegistry) -> None:
    assert dimensionalize([], ureg.minute) == []


def test_undimensionalize_roundtrip(ureg: UnitRegistry) -> None:
    values = [1.0, 5.0, 10.0]
    unit = 1 / ureg.femtoliter
    result = undimensionalize(dimensionalize(values, unit), unit)
    assert result == pytest.approx(values)


def test_undimensionalize_unit_conversion(ureg: UnitRegistry) -> None:
    qty_list = dimensionalize([60.0], ureg.second)
    result = undimensionalize(qty_list, ureg.minute)
    assert result == pytest.approx([1.0])


def test_undimensionalize_infers_unit_when_none(ureg: UnitRegistry) -> None:
    qty_list = dimensionalize([3.0, 6.0], ureg.minute)
    result = undimensionalize(qty_list, None)
    assert result == pytest.approx([3.0, 6.0])


def test_get_unit_from_quantity(ureg: UnitRegistry) -> None:
    qty = 5.0 * ureg.femtoliter
    unit = get_unit(qty)
    assert str(unit) == str(ureg.femtoliter)


def test_get_unit_from_sequence(ureg: UnitRegistry) -> None:
    seq = dimensionalize([1.0, 2.0], ureg.minute)
    unit = get_unit(seq)
    assert str(unit) == str(ureg.minute)
