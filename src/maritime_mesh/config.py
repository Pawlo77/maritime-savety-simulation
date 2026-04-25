"""Configuration dataclasses for maritime weather mesh simulation."""

from dataclasses import dataclass, field
from pathlib import Path

from maritime_mesh.enums import MethodCondition


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
