"""Unit tests for circadian and crew-error models."""

import pytest

from maritime_mesh.behaviour.human_factors import CircadianModel, ErrorProbabilityModel


def test_vigilance_nadir_and_peak() -> None:
    """Verify circadian vigilance reaches expected nadir and peak hours."""
    circadian = CircadianModel()
    assert circadian.vigilance(3.0) == pytest.approx(-1.0)
    assert circadian.vigilance(15.0) == pytest.approx(1.0)


def test_error_probability_increases_with_awake_hours() -> None:
    """Verify crew error probability grows with accumulated wakefulness."""
    model = ErrorProbabilityModel()
    low = model.compute(2.0, 9.0, 0.0)
    high = model.compute(18.0, 9.0, 0.0)
    assert high > low
