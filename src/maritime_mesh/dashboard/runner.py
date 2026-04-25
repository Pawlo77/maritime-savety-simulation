"""Experiment run helpers for GUI pages."""

from pathlib import Path

import pandas as pd

from maritime_mesh.enums import MethodCondition
from maritime_mesh.experiment import scenarios
from maritime_mesh.experiment.runner import ExperimentRunner

LaneDefinitions = tuple[tuple[str, tuple[tuple[float, float], ...]], ...]


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
    lane_definitions: LaneDefinitions,
    min_spawn_distance_nm: float = 0.0,
    max_spawn_distance_nm: float | None = None,
) -> pd.DataFrame:
    """Run experiment matrix from GUI controls."""
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
            "lane_definitions": lane_definitions,
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
