"""Configuration dataclasses for maritime weather mesh simulation."""

from dataclasses import dataclass, field
from pathlib import Path

from maritime_mesh.constants import (
    LAND_CLEARANCE_NM,
    LAND_PROFILE,
    RADIO_RANGE_FALLOFF,
    RADIO_WEATHER_INTERFERENCE,
    ROUTE_END_NEAR_SHORE_NM,
    ROUTE_END_OFFMAP_MARGIN_NM,
    ROUTE_START_NEAR_SHORE_NM,
    SHORE_BROADCAST_RADIUS_NM,
    SHORE_STATION_POSITION,
    VESSEL_RADIO_RANGE_NM,
    WORLD_SIZE_NM,
)
from maritime_mesh.enums import MethodCondition

WaypointTuple = tuple[float, float]
LaneDefinition = tuple[str, tuple[WaypointTuple, ...]]


@dataclass(frozen=True)
class ScenarioConfig:
    """Immutable parameters for a single simulation scenario."""

    name: str
    n_vessels: int
    shore_noise_std: float
    green_crew_fraction: float
    hazard_spike_value: float | None = None
    hazard_spike_tick: int | None = None
    noise_double_tick: int | None = None
    min_spawn_distance_nm: float = 0.0
    max_spawn_distance_nm: float = 100.0
    hypothesis_tags: tuple[str, ...] = ()
    calibration_target: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate scenario-level parameter bounds and invariants."""
        if not self.name.strip():
            raise ValueError("Scenario name must be non-empty.")
        if self.n_vessels <= 0:
            raise ValueError("Scenario n_vessels must be > 0.")
        if not 0.0 <= self.shore_noise_std <= 1.0:
            raise ValueError("Scenario shore_noise_std must be in [0.0, 1.0].")
        if not 0.0 <= self.green_crew_fraction <= 1.0:
            raise ValueError("Scenario green_crew_fraction must be in [0.0, 1.0].")
        if self.min_spawn_distance_nm < 0.0 or self.max_spawn_distance_nm < 0.0:
            raise ValueError("Spawn distance bounds must be >= 0.")
        if self.min_spawn_distance_nm > self.max_spawn_distance_nm:
            raise ValueError("min_spawn_distance_nm must be <= max_spawn_distance_nm.")
        if self.hazard_spike_tick is not None and self.hazard_spike_tick < 0:
            raise ValueError("hazard_spike_tick must be >= 0.")
        if self.noise_double_tick is not None and self.noise_double_tick < 0:
            raise ValueError("noise_double_tick must be >= 0.")
        if self.hazard_spike_value is not None and not 0.0 <= self.hazard_spike_value <= 1.0:
            raise ValueError("hazard_spike_value must be in [0.0, 1.0].")


@dataclass(frozen=True)
class SimulationConfig:
    """Top-level immutable simulation run configuration."""

    scenario: ScenarioConfig
    method: MethodCondition
    seed: int
    n_ticks: int
    mesh_enabled: bool
    evacuation_enabled: bool
    human_factors_enabled: bool
    output_dir: Path
    world_size_nm: float = WORLD_SIZE_NM
    shore_station_position: tuple[float, float] = SHORE_STATION_POSITION
    shore_station_positions: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    lane_definitions: tuple[LaneDefinition, ...] = field(default_factory=tuple)
    vessel_radio_range_nm: float = VESSEL_RADIO_RANGE_NM
    max_hop_count: int = 2
    shore_broadcast_radius_nm: float = SHORE_BROADCAST_RADIUS_NM
    radio_range_falloff: float = RADIO_RANGE_FALLOFF
    radio_weather_interference: float = RADIO_WEATHER_INTERFERENCE
    radio_packet_loss_rate: float = 0.02
    scheduler_mode: str = "phased"
    land_profile: str = LAND_PROFILE
    land_clearance_nm: float = LAND_CLEARANCE_NM
    route_start_near_shore_nm: float = ROUTE_START_NEAR_SHORE_NM
    route_end_near_shore_nm: float = ROUTE_END_NEAR_SHORE_NM
    route_end_offmap_margin_nm: float = ROUTE_END_OFFMAP_MARGIN_NM
    lane_endpoint_spawn_weights: tuple[tuple[str, tuple[float, float]], ...] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        """Validate simulation-level invariants used by the engine."""
        if self.n_ticks <= 0:
            raise ValueError("Simulation n_ticks must be > 0.")
        if self.world_size_nm <= 0.0:
            raise ValueError("world_size_nm must be > 0.")
        if self.vessel_radio_range_nm <= 0.0:
            raise ValueError("vessel_radio_range_nm must be > 0.")
        if self.max_hop_count < 0:
            raise ValueError("max_hop_count must be >= 0.")
        if self.shore_broadcast_radius_nm <= 0.0:
            raise ValueError("shore_broadcast_radius_nm must be > 0.")
        if self.radio_range_falloff <= 0.0:
            raise ValueError("radio_range_falloff must be > 0.")
        if self.radio_weather_interference < 0.0:
            raise ValueError("radio_weather_interference must be >= 0.")
        if not 0.0 <= self.radio_packet_loss_rate <= 1.0:
            raise ValueError("radio_packet_loss_rate must be in [0.0, 1.0].")
        if self.scheduler_mode not in {"legacy", "phased"}:
            raise ValueError("scheduler_mode must be either 'legacy' or 'phased'.")
        if self.land_profile not in {"legacy_rectangles", "natural_coast"}:
            raise ValueError("land_profile must be 'legacy_rectangles' or 'natural_coast'.")
        if self.land_clearance_nm < 0.0:
            raise ValueError("land_clearance_nm must be >= 0.")
        if self.route_start_near_shore_nm < 0.0 or self.route_end_near_shore_nm < 0.0:
            raise ValueError("Route-to-shore thresholds must be >= 0.")
        if self.route_end_offmap_margin_nm < 0.0:
            raise ValueError("route_end_offmap_margin_nm must be >= 0.")
        if (
            self.shore_station_positions
            and self.shore_station_position in self.shore_station_positions
        ):
            # Allowed but redundant: kept intentionally silent.
            pass
        stations = self.shore_station_positions or (self.shore_station_position,)
        for x_nm, y_nm in stations:
            if not 0.0 <= x_nm <= self.world_size_nm or not 0.0 <= y_nm <= self.world_size_nm:
                raise ValueError("Shore station positions must lie within world bounds.")
        for lane_name, waypoints in self.lane_definitions:
            if not lane_name.strip():
                raise ValueError("Lane names must be non-empty.")
            if len(waypoints) < 2:
                raise ValueError("Each lane definition must include at least two waypoints.")
            for x_nm, y_nm in waypoints:
                if not 0.0 <= x_nm <= self.world_size_nm or not 0.0 <= y_nm <= self.world_size_nm:
                    raise ValueError("Lane waypoints must lie within world bounds.")
        for lane_name, weights in self.lane_endpoint_spawn_weights:
            if not lane_name.strip():
                raise ValueError("Spawn weight lane names must be non-empty.")
            if len(weights) != 2:
                raise ValueError("Each lane endpoint weight must contain start and end weights.")
            if weights[0] < 0.0 or weights[1] < 0.0:
                raise ValueError("Lane endpoint spawn weights must be >= 0.")
