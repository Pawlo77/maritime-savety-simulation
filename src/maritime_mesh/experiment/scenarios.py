"""Scenario factory functions."""

from pathlib import Path

from maritime_mesh.config import ScenarioConfig, SimulationConfig
from maritime_mesh.enums import MethodCondition


def _make_config(
    scenario: ScenarioConfig,
    method: MethodCondition,
    seed: int,
) -> SimulationConfig:
    """Build simulation config with method-specific ablations."""
    return SimulationConfig(
        scenario=scenario,
        method=method,
        seed=seed,
        n_ticks=120,
        mesh_enabled=method == MethodCondition.PROPOSED,
        evacuation_enabled=method != MethodCondition.BASELINE_A,
        human_factors_enabled=method != MethodCondition.BASELINE_A,
        output_dir=Path("outputs/maritime_mesh"),
    )


def scenario_1_calm_passage(method: MethodCondition, seed: int) -> SimulationConfig:
    """Factory for scenario 1 calibration case."""
    scenario = ScenarioConfig(
        name="scenario_1_calm_passage",
        n_vessels=20,
        shore_noise_std=0.05,
        green_crew_fraction=0.1,
        hypothesis_tags=("calibration",),
        calibration_target={"collision_per_1k_hrs": 0.5},
    )
    return _make_config(scenario=scenario, method=method, seed=seed)


def scenario_2_storm_corridor(method: MethodCondition, seed: int) -> SimulationConfig:
    """Factory for scenario 2 storm corridor."""
    scenario = ScenarioConfig(
        name="scenario_2_storm_corridor",
        n_vessels=25,
        shore_noise_std=0.18,
        green_crew_fraction=0.3,
        hypothesis_tags=("H1",),
    )
    return _make_config(scenario=scenario, method=method, seed=seed)


def scenario_3_blind_shore(method: MethodCondition, seed: int) -> SimulationConfig:
    """Factory for scenario 3 blind shore stress case."""
    scenario = ScenarioConfig(
        name="scenario_3_blind_shore",
        n_vessels=25,
        shore_noise_std=0.18,
        green_crew_fraction=0.3,
        noise_double_tick=40,
        min_spawn_distance_nm=40.0,
        max_spawn_distance_nm=70.0,
        hypothesis_tags=("H3",),
    )
    return _make_config(scenario=scenario, method=method, seed=seed)


def scenario_4_deep_water_rescue(method: MethodCondition, seed: int) -> SimulationConfig:
    """Factory for scenario 4 deep-water rescue case."""
    scenario = ScenarioConfig(
        name="scenario_4_deep_water_rescue",
        n_vessels=15,
        shore_noise_std=0.18,
        green_crew_fraction=0.25,
        hazard_spike_tick=30,
        hazard_spike_value=0.9,
        min_spawn_distance_nm=60.0,
        max_spawn_distance_nm=100.0,
        hypothesis_tags=("H2", "H3"),
    )
    return _make_config(scenario=scenario, method=method, seed=seed)
