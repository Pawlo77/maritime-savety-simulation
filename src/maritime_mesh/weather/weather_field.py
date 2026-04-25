"""Weather-field generation and sampling."""

from dataclasses import dataclass
from math import cos, radians, sin

import numpy as np

from maritime_mesh.constants import WEATHER_GRID_CELLS, WORLD_SIZE_NM
from maritime_mesh.weather.weather_cell import WeatherCell


@dataclass
class _StormSystem:
    center_x: float
    center_y: float
    radius_cells: float
    intensity: float
    phase: float
    drift_dx: float
    drift_dy: float


class WeatherField:
    """Tier-2 weather field with regimes, moving systems, and coupled channels."""

    def __init__(
        self,
        rng: np.random.Generator,
        world_size_nm: float = WORLD_SIZE_NM,
        weather_preset: str = "mixed",
        weather_unpredictability: float = 0.5,
        weather_calm_to_storm_prob: float = 0.03,
        weather_storm_to_calm_prob: float = 0.08,
        weather_storm_spawn_rate: float = 0.25,
        weather_max_systems: int = 3,
        weather_system_radius_min_cells: int = 4,
        weather_system_radius_max_cells: int = 10,
        weather_system_intensity_min: float = 0.45,
        weather_system_intensity_max: float = 0.95,
        weather_system_drift_speed_cells: float = 0.6,
        weather_system_drift_direction_deg: float = 35.0,
        weather_system_drift_jitter_deg: float = 18.0,
        weather_front_strength: float = 0.12,
        weather_background_persistence: float = 0.92,
        weather_channel_persistence_sea: float = 0.95,
        weather_channel_persistence_visibility: float = 0.90,
        weather_channel_persistence_wind: float = 0.93,
        weather_innovation_scale_sea: float = 0.02,
        weather_innovation_scale_visibility: float = 0.03,
        weather_innovation_scale_wind: float = 0.025,
        weather_shock_probability: float = 0.015,
        weather_shock_scale: float = 0.20,
        weather_coupling_sea_wind: float = 0.35,
        weather_coupling_sea_visibility: float = -0.20,
        weather_coupling_wind_visibility: float = -0.25,
        weather_gradient_limit: float = 0.12,
        weather_weight_sea_state: float = 0.4,
        weather_weight_visibility: float = 0.3,
        weather_weight_wind: float = 0.3,
    ) -> None:
        """Initialize weather state and tier-2 evolution parameters."""
        self.rng = rng
        self.world_size_nm = world_size_nm
        self.weather_preset = weather_preset
        self.weather_unpredictability = float(weather_unpredictability)
        self.weather_calm_to_storm_prob = float(weather_calm_to_storm_prob)
        self.weather_storm_to_calm_prob = float(weather_storm_to_calm_prob)
        self.weather_storm_spawn_rate = float(weather_storm_spawn_rate)
        self.weather_max_systems = int(weather_max_systems)
        self.weather_system_radius_min_cells = int(weather_system_radius_min_cells)
        self.weather_system_radius_max_cells = int(weather_system_radius_max_cells)
        self.weather_system_intensity_min = float(weather_system_intensity_min)
        self.weather_system_intensity_max = float(weather_system_intensity_max)
        self.weather_system_drift_speed_cells = float(weather_system_drift_speed_cells)
        self.weather_system_drift_direction_deg = float(weather_system_drift_direction_deg)
        self.weather_system_drift_jitter_deg = float(weather_system_drift_jitter_deg)
        self.weather_front_strength = float(weather_front_strength)
        self.weather_background_persistence = float(weather_background_persistence)
        self.weather_channel_persistence_sea = float(weather_channel_persistence_sea)
        self.weather_channel_persistence_visibility = float(weather_channel_persistence_visibility)
        self.weather_channel_persistence_wind = float(weather_channel_persistence_wind)
        self.weather_innovation_scale_sea = float(weather_innovation_scale_sea)
        self.weather_innovation_scale_visibility = float(weather_innovation_scale_visibility)
        self.weather_innovation_scale_wind = float(weather_innovation_scale_wind)
        self.weather_shock_probability = float(weather_shock_probability)
        self.weather_shock_scale = float(weather_shock_scale)
        self.weather_coupling_sea_wind = float(weather_coupling_sea_wind)
        self.weather_coupling_sea_visibility = float(weather_coupling_sea_visibility)
        self.weather_coupling_wind_visibility = float(weather_coupling_wind_visibility)
        self.weather_gradient_limit = float(weather_gradient_limit)
        self.weather_weight_sea_state = float(weather_weight_sea_state)
        self.weather_weight_visibility = float(weather_weight_visibility)
        self.weather_weight_wind = float(weather_weight_wind)
        self._regime = "calm"
        self._systems: list[_StormSystem] = []
        self._grid_x, self._grid_y = np.meshgrid(
            np.arange(WEATHER_GRID_CELLS, dtype=float),
            np.arange(WEATHER_GRID_CELLS, dtype=float),
        )
        self._sea = self.rng.uniform(0.15, 0.40, size=(WEATHER_GRID_CELLS, WEATHER_GRID_CELLS))
        self._vis = self.rng.uniform(0.60, 0.95, size=(WEATHER_GRID_CELLS, WEATHER_GRID_CELLS))
        self._wind = self.rng.uniform(0.15, 0.45, size=(WEATHER_GRID_CELLS, WEATHER_GRID_CELLS))
        self._advection_x = int(rng.integers(-1, 2))
        self._advection_y = int(rng.integers(-1, 2))
        self.grid = self._to_cells()

    def _smooth_field(self, field: np.ndarray) -> np.ndarray:
        """Apply local neighborhood smoothing for spatial coherence."""
        return (
            0.52 * field
            + 0.12 * np.roll(field, 1, axis=0)
            + 0.12 * np.roll(field, -1, axis=0)
            + 0.12 * np.roll(field, 1, axis=1)
            + 0.12 * np.roll(field, -1, axis=1)
        )

    def _limit_gradient(self, field: np.ndarray) -> np.ndarray:
        """Limit gradients to reduce unrealistic calm-extreme adjacency."""
        if self.weather_gradient_limit <= 0.0:
            return np.clip(field, 0.0, 1.0)
        lower = (
            np.maximum.reduce(
                [
                    np.roll(field, 1, axis=0),
                    np.roll(field, -1, axis=0),
                    np.roll(field, 1, axis=1),
                    np.roll(field, -1, axis=1),
                ]
            )
            - self.weather_gradient_limit
        )
        upper = (
            np.minimum.reduce(
                [
                    np.roll(field, 1, axis=0),
                    np.roll(field, -1, axis=0),
                    np.roll(field, 1, axis=1),
                    np.roll(field, -1, axis=1),
                ]
            )
            + self.weather_gradient_limit
        )
        limited = np.clip(field, lower, upper)
        return np.clip(limited, 0.0, 1.0)

    def _storm_kernel(self, system: _StormSystem) -> np.ndarray:
        """Generate circular Gaussian-like kernel for one storm system."""
        dx = self._grid_x - system.center_x
        dy = self._grid_y - system.center_y
        radius = max(1.0, system.radius_cells)
        squared = (dx * dx) + (dy * dy)
        radial = np.exp(-squared / (2.0 * radius * radius))
        return radial * system.intensity

    def _spawn_system(self, regime_boost: float) -> None:
        """Spawn a circular moving weather system if capacity allows."""
        if len(self._systems) >= self.weather_max_systems:
            return
        if self.rng.random() > self.weather_storm_spawn_rate * regime_boost:
            return
        direction_deg = self.weather_system_drift_direction_deg + self.rng.normal(
            0.0, self.weather_system_drift_jitter_deg
        )
        direction = radians(direction_deg)
        speed = max(0.0, self.weather_system_drift_speed_cells)
        self._systems.append(
            _StormSystem(
                center_x=float(self.rng.uniform(0.0, WEATHER_GRID_CELLS - 1)),
                center_y=float(self.rng.uniform(0.0, WEATHER_GRID_CELLS - 1)),
                radius_cells=float(
                    self.rng.uniform(
                        self.weather_system_radius_min_cells,
                        self.weather_system_radius_max_cells,
                    )
                ),
                intensity=float(
                    self.rng.uniform(
                        self.weather_system_intensity_min,
                        self.weather_system_intensity_max,
                    )
                ),
                phase=0.0,
                drift_dx=speed * cos(direction),
                drift_dy=speed * sin(direction),
            )
        )

    def _advance_systems(self) -> tuple[np.ndarray, np.ndarray]:
        """Advance storm systems and return sea/wind and visibility impacts."""
        sea_wind_overlay = np.zeros((WEATHER_GRID_CELLS, WEATHER_GRID_CELLS), dtype=float)
        vis_overlay = np.zeros((WEATHER_GRID_CELLS, WEATHER_GRID_CELLS), dtype=float)
        next_systems: list[_StormSystem] = []
        for system in self._systems:
            jitter_scale = 0.05 + (0.25 * self.weather_unpredictability)
            system.center_x = (system.center_x + system.drift_dx) % WEATHER_GRID_CELLS
            system.center_y = (system.center_y + system.drift_dy) % WEATHER_GRID_CELLS
            system.drift_dx += float(self.rng.normal(0.0, jitter_scale))
            system.drift_dy += float(self.rng.normal(0.0, jitter_scale))
            system.phase += 0.20 + (0.35 * self.weather_unpredictability)
            growth = 1.0 + (0.10 * np.sin(system.phase))
            system.radius_cells = float(
                np.clip(
                    system.radius_cells * growth,
                    self.weather_system_radius_min_cells,
                    self.weather_system_radius_max_cells * 1.25,
                )
            )
            system.intensity = float(
                np.clip(
                    system.intensity
                    + self.rng.normal(0.0, 0.03 + (0.04 * self.weather_unpredictability)),
                    0.10,
                    1.00,
                )
            )
            if system.intensity < 0.12:
                continue
            kernel = self._storm_kernel(system)
            sea_wind_overlay += kernel
            vis_overlay += kernel * (0.7 + 0.2 * self.weather_unpredictability)
            decay = 0.006 if self._regime == "storm" else 0.02
            system.intensity = max(0.0, system.intensity - decay)
            next_systems.append(system)
        self._systems = next_systems
        return np.clip(sea_wind_overlay, 0.0, 1.0), np.clip(vis_overlay, 0.0, 1.0)

    def _apply_front(self, base: np.ndarray, strength: float | None = None) -> np.ndarray:
        """Apply soft front structure to create coherent non-circular gradients."""
        direction = radians(self.weather_system_drift_direction_deg)
        plane = (self._grid_x * cos(direction)) + (self._grid_y * sin(direction))
        phase = float(self.rng.uniform(0.0, 2.0 * np.pi))
        front = 0.5 + (0.5 * np.sin((plane / 6.0) + phase))
        front_strength = self.weather_front_strength if strength is None else float(strength)
        return np.clip(base + (front_strength * (front - 0.5)), 0.0, 1.0)

    def _evolve_background(
        self, channel: np.ndarray, persistence: float, innovation_scale: float
    ) -> np.ndarray:
        """Advance one channel background with advection and bounded innovation."""
        advected = np.roll(channel, shift=(self._advection_y, self._advection_x), axis=(0, 1))
        smoothed = self._smooth_field(advected)
        innovation = self.rng.normal(
            0.0,
            innovation_scale * (0.3 + self.weather_unpredictability),
            size=channel.shape,
        )
        evolved = (persistence * smoothed) + ((1.0 - persistence) * 0.5) + innovation
        return np.clip(evolved, 0.0, 1.0)

    def _to_cells(self) -> np.ndarray:
        """Materialize channel tensors into WeatherCell grid."""
        cells = np.empty((WEATHER_GRID_CELLS, WEATHER_GRID_CELLS), dtype=object)
        for i in range(WEATHER_GRID_CELLS):
            for j in range(WEATHER_GRID_CELLS):
                cells[i, j] = WeatherCell(
                    sea_state=float(self._sea[i, j]),
                    visibility=float(self._vis[i, j]),
                    wind_norm=float(self._wind[i, j]),
                    weight_sea_state=self.weather_weight_sea_state,
                    weight_visibility=self.weather_weight_visibility,
                    weight_wind=self.weather_weight_wind,
                )
        return cells

    def step(self) -> None:
        """Advance weather using regime switching and moving circular systems."""
        is_calm_preset = self.weather_preset == "calm"
        if self.rng.random() < 0.2:
            self._advection_x = int(np.clip(self._advection_x + self.rng.integers(-1, 2), -1, 1))
            self._advection_y = int(np.clip(self._advection_y + self.rng.integers(-1, 2), -1, 1))
        calm_to_storm_prob = self.weather_calm_to_storm_prob
        storm_to_calm_prob = self.weather_storm_to_calm_prob
        if is_calm_preset:
            # Keep calm scenarios stable for early ticks and avoid rapid storm lock-in.
            calm_to_storm_prob *= 0.35
            storm_to_calm_prob = min(1.0, storm_to_calm_prob * 1.4)
        if self._regime == "calm" and self.rng.random() < calm_to_storm_prob:
            self._regime = "storm"
        elif self._regime == "storm" and self.rng.random() < storm_to_calm_prob:
            self._regime = "calm"

        if is_calm_preset:
            regime_boost = 0.75 if self._regime == "storm" else 0.20
            sea_overlay_scale = 0.14
            wind_overlay_scale = 0.12
            vis_overlay_scale = 0.10
            sea_anchor = 0.28
            wind_anchor = 0.25
            vis_anchor = 0.88
            coupling_scale = 0.35
            front_strength = self.weather_front_strength * 0.30
            shock_probability = self.weather_shock_probability * 0.20
        else:
            regime_boost = 1.35 if self._regime == "storm" else 0.65
            sea_overlay_scale = 0.45
            wind_overlay_scale = 0.40
            vis_overlay_scale = 0.35
            sea_anchor = 0.5
            wind_anchor = 0.5
            vis_anchor = 0.7
            coupling_scale = 1.0
            front_strength = self.weather_front_strength
            shock_probability = self.weather_shock_probability
        self._spawn_system(regime_boost=regime_boost)
        sea_wind_overlay, vis_overlay = self._advance_systems()

        self._sea = self._evolve_background(
            self._sea, self.weather_channel_persistence_sea, self.weather_innovation_scale_sea
        )
        self._vis = self._evolve_background(
            self._vis,
            self.weather_channel_persistence_visibility,
            self.weather_innovation_scale_visibility,
        )
        self._wind = self._evolve_background(
            self._wind, self.weather_channel_persistence_wind, self.weather_innovation_scale_wind
        )

        self._sea = np.clip(
            (self.weather_background_persistence * self._sea)
            + ((1.0 - self.weather_background_persistence) * sea_anchor)
            + (sea_overlay_scale * sea_wind_overlay),
            0.0,
            1.0,
        )
        self._wind = np.clip(
            (self.weather_background_persistence * self._wind)
            + ((1.0 - self.weather_background_persistence) * wind_anchor)
            + (wind_overlay_scale * sea_wind_overlay),
            0.0,
            1.0,
        )
        self._vis = np.clip(
            (self.weather_background_persistence * self._vis)
            + ((1.0 - self.weather_background_persistence) * vis_anchor)
            - (vis_overlay_scale * vis_overlay),
            0.0,
            1.0,
        )

        self._sea = np.clip(
            self._sea + (coupling_scale * self.weather_coupling_sea_wind * (self._wind - 0.5)),
            0.0,
            1.0,
        )
        self._sea = np.clip(
            self._sea + (coupling_scale * self.weather_coupling_sea_visibility * (0.5 - self._vis)),
            0.0,
            1.0,
        )
        self._wind = np.clip(
            self._wind
            + (coupling_scale * self.weather_coupling_wind_visibility * (0.5 - self._vis)),
            0.0,
            1.0,
        )
        self._sea = self._apply_front(self._sea, strength=front_strength)
        self._wind = self._apply_front(self._wind, strength=front_strength)
        self._vis = np.clip(
            1.0 - self._apply_front(1.0 - self._vis, strength=front_strength), 0.0, 1.0
        )

        if self.rng.random() < (shock_probability * (0.3 + self.weather_unpredictability)):
            shock = self.rng.normal(0.0, self.weather_shock_scale, size=self._sea.shape)
            self._sea = np.clip(self._sea + shock, 0.0, 1.0)
            self._wind = np.clip(self._wind + (0.8 * shock), 0.0, 1.0)
            self._vis = np.clip(self._vis - (0.6 * shock), 0.0, 1.0)

        self._sea = self._limit_gradient(self._smooth_field(self._sea))
        self._wind = self._limit_gradient(self._smooth_field(self._wind))
        self._vis = self._limit_gradient(self._smooth_field(self._vis))
        if is_calm_preset:
            # Guardrail: keep calm preset from drifting into sustained storm-like severity.
            self._sea = np.clip(self._sea, 0.0, 0.62)
            self._wind = np.clip(self._wind, 0.0, 0.62)
            self._vis = np.clip(self._vis, 0.45, 1.0)
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

    def channels_at(self, x_nm: float, y_nm: float) -> tuple[float, float, float]:
        """Read sea/visibility/wind channels at world position."""
        i = self._coord_to_index(y_nm)
        j = self._coord_to_index(x_nm)
        return (float(self._sea[i, j]), float(self._vis[i, j]), float(self._wind[i, j]))

    def inject_hazard_spike(self, cell_i: int, cell_j: int, value: float) -> None:
        """Override one cell channels to force target hazard-like severity."""
        forced = float(min(1.0, max(0.0, value)))
        self._sea[cell_i, cell_j] = forced
        self._vis[cell_i, cell_j] = 1.0 - forced
        self._wind[cell_i, cell_j] = forced
        self.grid[cell_i, cell_j] = WeatherCell(
            forced,
            1.0 - forced,
            forced,
            weight_sea_state=self.weather_weight_sea_state,
            weight_visibility=self.weather_weight_visibility,
            weight_wind=self.weather_weight_wind,
        )
