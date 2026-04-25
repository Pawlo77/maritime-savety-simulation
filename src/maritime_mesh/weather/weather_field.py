"""Weather-field generation and sampling."""

import numpy as np

from maritime_mesh.constants import WEATHER_GRID_CELLS, WORLD_SIZE_NM
from maritime_mesh.weather.weather_cell import WeatherCell


class WeatherField:
    """Weather grid that evolves with smooth stochastic drift."""

    def __init__(self, rng: np.random.Generator, world_size_nm: float = WORLD_SIZE_NM) -> None:
        """Initialize field with seeded random generator."""
        self.rng = rng
        self.world_size_nm = world_size_nm
        self._sea = rng.uniform(0.0, 1.0, size=(WEATHER_GRID_CELLS, WEATHER_GRID_CELLS))
        self._vis = rng.uniform(0.0, 1.0, size=(WEATHER_GRID_CELLS, WEATHER_GRID_CELLS))
        self._wind = rng.uniform(0.0, 1.0, size=(WEATHER_GRID_CELLS, WEATHER_GRID_CELLS))
        self._advection_x = int(rng.integers(-1, 2))
        self._advection_y = int(rng.integers(-1, 2))
        self.grid = self._to_cells()

    def _smooth_field(self, field: np.ndarray) -> np.ndarray:
        """Apply local neighborhood smoothing for spatial coherence."""
        return (
            0.5 * field
            + 0.125 * np.roll(field, 1, axis=0)
            + 0.125 * np.roll(field, -1, axis=0)
            + 0.125 * np.roll(field, 1, axis=1)
            + 0.125 * np.roll(field, -1, axis=1)
        )

    def _evolve_channel(self, channel: np.ndarray, drift_scale: float) -> np.ndarray:
        """Advance one weather channel with advection, smoothing, and drift."""
        advected = np.roll(channel, shift=(self._advection_y, self._advection_x), axis=(0, 1))
        smoothed = self._smooth_field(advected)
        drifted = smoothed + self.rng.normal(0.0, drift_scale, channel.shape)
        return np.clip(drifted, 0.0, 1.0)

    def _to_cells(self) -> np.ndarray:
        """Materialize channel tensors into WeatherCell grid."""
        cells = np.empty((WEATHER_GRID_CELLS, WEATHER_GRID_CELLS), dtype=object)
        for i in range(WEATHER_GRID_CELLS):
            for j in range(WEATHER_GRID_CELLS):
                cells[i, j] = WeatherCell(
                    sea_state=float(self._sea[i, j]),
                    visibility=float(self._vis[i, j]),
                    wind_norm=float(self._wind[i, j]),
                )
        return cells

    def step(self) -> None:
        """Advance weather by adding small bounded random drift."""
        drift_scale = 0.03
        if self.rng.random() < 0.2:
            self._advection_x = int(np.clip(self._advection_x + self.rng.integers(-1, 2), -1, 1))
            self._advection_y = int(np.clip(self._advection_y + self.rng.integers(-1, 2), -1, 1))
        self._sea = self._evolve_channel(self._sea, drift_scale=drift_scale)
        self._vis = self._evolve_channel(self._vis, drift_scale=drift_scale)
        self._wind = self._evolve_channel(self._wind, drift_scale=drift_scale)
        self.grid = self._to_cells()

    def _coord_to_index(self, value_nm: float) -> int:
        """Convert world coordinate to weather-grid index."""
        clipped = min(max(value_nm, 0.0), self.world_size_nm)
        ratio = clipped / self.world_size_nm
        return min(WEATHER_GRID_CELLS - 1, int(ratio * WEATHER_GRID_CELLS))

    def hazard_at(self, x_nm: float, y_nm: float) -> float:
        """Read composite hazard at world position."""
        i = self._coord_to_index(y_nm)
        j = self._coord_to_index(x_nm)
        return self.grid[i, j].hazard()

    def inject_hazard_spike(self, cell_i: int, cell_j: int, value: float) -> None:
        """Override one cell channels to force target hazard-like severity."""
        forced = float(min(1.0, max(0.0, value)))
        self._sea[cell_i, cell_j] = forced
        self._vis[cell_i, cell_j] = 1.0 - forced
        self._wind[cell_i, cell_j] = forced
        self.grid[cell_i, cell_j] = WeatherCell(forced, 1.0 - forced, forced)
