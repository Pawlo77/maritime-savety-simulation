"""Weather-field generation and sampling."""

import numpy as np

from maritime_mesh.constants import WEATHER_GRID_CELLS, WORLD_SIZE_NM
from maritime_mesh.weather.weather_cell import WeatherCell


class WeatherField:
    """Weather grid that evolves with smooth stochastic drift."""

    def __init__(self, rng: np.random.Generator) -> None:
        """Initialize field with seeded random generator."""
        self.rng = rng
        self._sea = rng.uniform(0.0, 1.0, size=(WEATHER_GRID_CELLS, WEATHER_GRID_CELLS))
        self._vis = rng.uniform(0.0, 1.0, size=(WEATHER_GRID_CELLS, WEATHER_GRID_CELLS))
        self._wind = rng.uniform(0.0, 1.0, size=(WEATHER_GRID_CELLS, WEATHER_GRID_CELLS))
        self.grid = self._to_cells()

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
        self._sea = np.clip(self._sea + self.rng.normal(0.0, drift_scale, self._sea.shape), 0.0, 1.0)
        self._vis = np.clip(self._vis + self.rng.normal(0.0, drift_scale, self._vis.shape), 0.0, 1.0)
        self._wind = np.clip(
            self._wind + self.rng.normal(0.0, drift_scale, self._wind.shape), 0.0, 1.0
        )
        self.grid = self._to_cells()

    def _coord_to_index(self, value_nm: float) -> int:
        """Convert world coordinate to weather-grid index."""
        clipped = min(max(value_nm, 0.0), WORLD_SIZE_NM)
        ratio = clipped / WORLD_SIZE_NM
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
