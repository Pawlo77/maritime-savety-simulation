"""Shared pytest fixtures for maritime mesh tests."""

from pathlib import Path

import numpy as np
import pytest

from maritime_mesh.config import ScenarioConfig, SimulationConfig
from maritime_mesh.enums import MethodCondition


@pytest.fixture
def seeded_rng() -> np.random.Generator:
    """Provide deterministic shared RNG fixture."""
    return np.random.default_rng(1234)


@pytest.fixture
def default_simulation_config(tmp_path: Path) -> SimulationConfig:
    """Provide compact deterministic simulation config."""
    scenario = ScenarioConfig(
        name="test_scenario",
        n_vessels=4,
        shore_noise_std=0.1,
        green_crew_fraction=0.25,
    )
    return SimulationConfig(
        scenario=scenario,
        method=MethodCondition.PROPOSED,
        seed=1234,
        n_ticks=6,
        mesh_enabled=True,
        evacuation_enabled=True,
        human_factors_enabled=True,
        output_dir=tmp_path,
    )
