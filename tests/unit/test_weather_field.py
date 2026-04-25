import numpy as np

from maritime_mesh.weather.weather_field import WeatherField


def test_hazard_at_is_bounded(seeded_rng: np.random.Generator) -> None:
    field = WeatherField(seeded_rng)
    value = field.hazard_at(10.0, 20.0)
    assert 0.0 <= value <= 1.0


def test_step_changes_grid(seeded_rng: np.random.Generator) -> None:
    field = WeatherField(seeded_rng)
    before = field.hazard_at(30.0, 30.0)
    field.step()
    after = field.hazard_at(30.0, 30.0)
    assert before != after


def test_inject_hazard_spike_persists(seeded_rng: np.random.Generator) -> None:
    field = WeatherField(seeded_rng)
    field.inject_hazard_spike(5, 5, 1.0)
    assert field.grid[5, 5].hazard() > 0.9
