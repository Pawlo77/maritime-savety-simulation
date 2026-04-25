"""Unit tests for weather field sampling and mutation behaviour."""

import numpy as np

from maritime_mesh.weather.weather_field import WeatherField


def test_hazard_at_is_bounded(seeded_rng: np.random.Generator) -> None:
    """Verify sampled hazards always remain in unit interval."""
    field = WeatherField(seeded_rng)
    value = field.hazard_at(10.0, 20.0)
    assert 0.0 <= value <= 1.0


def test_step_changes_grid(seeded_rng: np.random.Generator) -> None:
    """Verify stepping weather field changes hazard values over time."""
    field = WeatherField(seeded_rng)
    before = field.hazard_at(30.0, 30.0)
    field.step()
    after = field.hazard_at(30.0, 30.0)
    assert before != after


def test_inject_hazard_spike_persists(seeded_rng: np.random.Generator) -> None:
    """Verify injected hazard spikes are reflected in target cell."""
    field = WeatherField(seeded_rng)
    field.inject_hazard_spike(5, 5, 1.0)
    assert field.grid[5, 5].hazard() > 0.9


def test_reproducible_with_same_seed() -> None:
    """Verify weather evolution is deterministic for a fixed seed."""
    field_a = WeatherField(np.random.default_rng(123))
    field_b = WeatherField(np.random.default_rng(123))
    for _ in range(8):
        field_a.step()
        field_b.step()
    assert np.allclose(field_a._sea, field_b._sea)
    assert np.allclose(field_a._vis, field_b._vis)
    assert np.allclose(field_a._wind, field_b._wind)


def test_supports_regime_and_storm_systems(seeded_rng: np.random.Generator) -> None:
    """Verify storm systems appear and evolve under aggressive settings."""
    field = WeatherField(
        seeded_rng,
        weather_calm_to_storm_prob=1.0,
        weather_storm_spawn_rate=1.0,
        weather_max_systems=4,
        weather_storm_to_calm_prob=0.0,
    )
    for _ in range(3):
        field.step()
    assert field._regime == "storm"
    assert len(field._systems) > 0


def test_gradient_limit_reduces_extreme_adjacent_jump(seeded_rng: np.random.Generator) -> None:
    """Verify gradient limiter avoids unrealistic calm/extreme adjacency."""
    field = WeatherField(seeded_rng, weather_gradient_limit=0.05)
    field._sea[:, :] = 0.0
    field._sea[25, 25] = 1.0
    limited = field._limit_gradient(field._sea)
    neighbor_delta = abs(limited[25, 25] - limited[25, 26])
    assert neighbor_delta <= 0.05 + 1e-6
