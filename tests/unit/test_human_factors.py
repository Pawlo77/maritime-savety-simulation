import pytest

from maritime_mesh.behaviour.human_factors import CircadianModel, ErrorProbabilityModel


def test_vigilance_nadir_and_peak() -> None:
    circadian = CircadianModel()
    assert circadian.vigilance(3.0) == pytest.approx(-1.0)
    assert circadian.vigilance(15.0) == pytest.approx(1.0)


def test_error_probability_increases_with_awake_hours() -> None:
    model = ErrorProbabilityModel()
    low = model.compute(2.0, 9.0, 0.0)
    high = model.compute(18.0, 9.0, 0.0)
    assert high > low
