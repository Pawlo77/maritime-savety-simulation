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
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lane_name, waypoints_part = line.split(":", maxsplit=1)
        waypoints = []
        for point in waypoints_part.split(";"):
            x_str, y_str = point.strip().split(",", maxsplit=1)
            waypoints.append((float(x_str), float(y_str)))
        lane_definitions.append((lane_name.strip(), tuple(waypoints)))
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
) -> pd.DataFrame:
    """Run experiment matrix from GUI controls."""
    land = WorldLand.default_for_world_size(world_size_nm)
    effective_shores = shore_positions or (shore_position,)
    for position in effective_shores:
        if not land.is_land(position):
            raise ValueError(f"Shore station {position} must be on land.")
    for lane_name, waypoints in lane_definitions:
        for idx in range(len(waypoints) - 1):
            if land.segment_intersects_land(waypoints[idx], waypoints[idx + 1]):
                raise ValueError(f"Lane '{lane_name}' intersects land.")

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
