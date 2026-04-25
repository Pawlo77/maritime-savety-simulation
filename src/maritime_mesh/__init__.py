"""Maritime weather mesh simulation package."""

from maritime_mesh.config import ScenarioConfig, SimulationConfig
from maritime_mesh.enums import MethodCondition
from maritime_mesh.logging_config import configure_logging

configure_logging()

__all__ = ["MethodCondition", "ScenarioConfig", "SimulationConfig"]
