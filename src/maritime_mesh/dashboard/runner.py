"""Experiment run helpers for GUI pages."""

from pathlib import Path

import pandas as pd

from maritime_mesh.enums import MethodCondition
from maritime_mesh.experiment import scenarios
from maritime_mesh.experiment.runner import ExperimentRunner
from maritime_mesh.world.land import WorldLand

LaneDefinitions = tuple[tuple[str, tuple[tuple[float, float], ...]], ...]
ShoreStations = tuple[tuple[float, float], ...]


def parse_lane_definitions(
    raw_text: str,
) -> tuple[tuple[str, tuple[tuple[float, float], ...]], ...]:
    """Parse lane definitions from multiline text."""
    lane_definitions = []
    example = "lane_alpha: 10,20; 25,40; 70,85"
    for line_number, raw_line in enumerate(raw_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(
                "Invalid lane format. Each line must look like "
                f"`{example}` (missing `:` on line {line_number})."
            )
        lane_name, waypoints_part = line.split(":", maxsplit=1)
        lane_name = lane_name.strip()
        if not lane_name:
            raise ValueError(
                f"Lane name cannot be empty on line {line_number}. Expected format: `{example}`."
            )
        waypoints = []
        for point in waypoints_part.split(";"):
            cleaned = point.strip()
            if not cleaned:
                continue
            if "," not in cleaned:
                raise ValueError(
                    f"Invalid waypoint `{cleaned}` on line {line_number}. "
                    "Use `x,y` format (example: `12.5,40`)."
                )
            x_str, y_str = cleaned.split(",", maxsplit=1)
            try:
                waypoints.append((float(x_str), float(y_str)))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric waypoint `{cleaned}` on line {line_number}. "
                    "Use numbers like `12,40` or `12.5,40.25`."
                ) from exc
        if len(waypoints) < 2:
            raise ValueError(
                f"Lane `{lane_name}` on line {line_number} needs at least two waypoints."
            )
        lane_definitions.append((lane_name, tuple(waypoints)))
    return tuple(lane_definitions)


def run_from_gui(
    output_dir: Path,
    selected_scenario_names: list[str],
    selected_methods: list[MethodCondition],
    n_seeds: int,
    n_ticks: int,
    n_vessels: int,
    world_size_nm: float,
    green_crew_fraction: float,
    shore_noise_std: float,
    shore_position: tuple[float, float],
    shore_positions: ShoreStations | None,
    lane_definitions: LaneDefinitions,
    min_spawn_distance_nm: float = 0.0,
    max_spawn_distance_nm: float | None = None,
    vessel_radio_range_nm: float | None = None,
    max_hop_count: int | None = None,
    shore_broadcast_radius_nm: float | None = None,
    radio_range_falloff: float | None = None,
    radio_weather_interference: float | None = None,
    radio_packet_loss_rate: float | None = None,
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
    weather_channel_persistence_visibility: float = 0.9,
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
    land_profile: str = "natural_coast",
    land_clearance_nm: float = 0.0,
    route_start_near_shore_nm: float = 45.0,
    route_end_near_shore_nm: float = 60.0,
    route_end_offmap_margin_nm: float = 12.0,
    lane_endpoint_spawn_weights: tuple[tuple[str, tuple[float, float]], ...] = (),
    max_workers: int = 1,
) -> pd.DataFrame:
    """Run experiment matrix from GUI controls."""
    land = WorldLand.default_for_world_size(world_size_nm, profile=land_profile)
    effective_shores = shore_positions or (shore_position,)
    for position in effective_shores:
        if not land.is_land(position):
            raise ValueError(
                f"Shore station {position} must be on land. "
                "Move station coordinates onto the green land area in the preview."
            )
    for lane_name, waypoints in lane_definitions:
        start_point = waypoints[0]
        end_point = waypoints[-1]
        near_start_shore = any(
            ((start_point[0] - shore[0]) ** 2 + (start_point[1] - shore[1]) ** 2) ** 0.5
            <= route_start_near_shore_nm
            for shore in effective_shores
        )
        if not near_start_shore:
            raise ValueError(
                f"Lane '{lane_name}' must start near shore "
                f"(<= {route_start_near_shore_nm:.1f} nm). "
                "Move the first waypoint closer to a shore station."
            )
        near_end_shore = any(
            ((end_point[0] - shore[0]) ** 2 + (end_point[1] - shore[1]) ** 2) ** 0.5
            <= route_end_near_shore_nm
            for shore in effective_shores
        )
        near_boundary = (
            end_point[0] <= route_end_offmap_margin_nm
            or end_point[1] <= route_end_offmap_margin_nm
            or end_point[0] >= world_size_nm - route_end_offmap_margin_nm
            or end_point[1] >= world_size_nm - route_end_offmap_margin_nm
        )
        if not (near_end_shore or near_boundary):
            raise ValueError(
                f"Lane '{lane_name}' must end near shore or map boundary. "
                "Move the final waypoint near a shore station or close to map edge."
            )
        for idx in range(len(waypoints) - 1):
            if land.segment_intersects_land(
                waypoints[idx],
                waypoints[idx + 1],
                clearance_nm=land_clearance_nm,
            ):
                raise ValueError(
                    f"Lane '{lane_name}' intersects land. "
                    "Add an offshore waypoint to route around the shoreline."
                )

    scenario_lookup = {
        "scenario_1_calm_passage": scenarios.scenario_1_calm_passage,
        "scenario_2_storm_corridor": scenarios.scenario_2_storm_corridor,
        "scenario_3_blind_shore": scenarios.scenario_3_blind_shore,
        "scenario_4_deep_water_rescue": scenarios.scenario_4_deep_water_rescue,
    }
    scenario_factories = [scenario_lookup[name] for name in selected_scenario_names]
    max_spawn = world_size_nm if max_spawn_distance_nm is None else max_spawn_distance_nm
    runner = ExperimentRunner(
        n_seeds=n_seeds,
        max_workers=max_workers,
        output_dir=output_dir,
        scenario_factories=scenario_factories,
        methods=selected_methods,
        simulation_overrides={
            "n_ticks": n_ticks,
            "world_size_nm": world_size_nm,
            "shore_station_position": shore_position,
            "shore_station_positions": effective_shores,
            "lane_definitions": lane_definitions,
            "vessel_radio_range_nm": vessel_radio_range_nm,
            "max_hop_count": max_hop_count,
            "shore_broadcast_radius_nm": shore_broadcast_radius_nm,
            "radio_range_falloff": radio_range_falloff,
            "radio_weather_interference": radio_weather_interference,
            "radio_packet_loss_rate": radio_packet_loss_rate,
            "land_profile": land_profile,
            "land_clearance_nm": land_clearance_nm,
            "route_start_near_shore_nm": route_start_near_shore_nm,
            "route_end_near_shore_nm": route_end_near_shore_nm,
            "route_end_offmap_margin_nm": route_end_offmap_margin_nm,
            "lane_endpoint_spawn_weights": lane_endpoint_spawn_weights,
            "weather_preset": weather_preset,
            "weather_unpredictability": weather_unpredictability,
            "weather_calm_to_storm_prob": weather_calm_to_storm_prob,
            "weather_storm_to_calm_prob": weather_storm_to_calm_prob,
            "weather_storm_spawn_rate": weather_storm_spawn_rate,
            "weather_max_systems": weather_max_systems,
            "weather_system_radius_min_cells": weather_system_radius_min_cells,
            "weather_system_radius_max_cells": weather_system_radius_max_cells,
            "weather_system_intensity_min": weather_system_intensity_min,
            "weather_system_intensity_max": weather_system_intensity_max,
            "weather_system_drift_speed_cells": weather_system_drift_speed_cells,
            "weather_system_drift_direction_deg": weather_system_drift_direction_deg,
            "weather_system_drift_jitter_deg": weather_system_drift_jitter_deg,
            "weather_front_strength": weather_front_strength,
            "weather_background_persistence": weather_background_persistence,
            "weather_channel_persistence_sea": weather_channel_persistence_sea,
            "weather_channel_persistence_visibility": weather_channel_persistence_visibility,
            "weather_channel_persistence_wind": weather_channel_persistence_wind,
            "weather_innovation_scale_sea": weather_innovation_scale_sea,
            "weather_innovation_scale_visibility": weather_innovation_scale_visibility,
            "weather_innovation_scale_wind": weather_innovation_scale_wind,
            "weather_shock_probability": weather_shock_probability,
            "weather_shock_scale": weather_shock_scale,
            "weather_coupling_sea_wind": weather_coupling_sea_wind,
            "weather_coupling_sea_visibility": weather_coupling_sea_visibility,
            "weather_coupling_wind_visibility": weather_coupling_wind_visibility,
            "weather_gradient_limit": weather_gradient_limit,
            "weather_weight_sea_state": weather_weight_sea_state,
            "weather_weight_visibility": weather_weight_visibility,
            "weather_weight_wind": weather_weight_wind,
        },
        scenario_overrides={
            "n_vessels": n_vessels,
            "green_crew_fraction": green_crew_fraction,
            "shore_noise_std": shore_noise_std,
            "min_spawn_distance_nm": min_spawn_distance_nm,
            "max_spawn_distance_nm": max_spawn,
        },
    )
    return runner.run_all()
