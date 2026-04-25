"""Integration tests for scenario calibration constraints."""

from maritime_mesh.enums import MethodCondition
from maritime_mesh.experiment.scenarios import scenario_1_calm_passage
from maritime_mesh.model.maritime_model import MaritimeModel


def test_scenario_1_baseline_a_collision_threshold() -> None:
    """Validate scenario 1 produces a non-negative collision KPI."""
    config = scenario_1_calm_passage(method=MethodCondition.BASELINE_A, seed=7)
    model = MaritimeModel(config)
    result = model.run()
    assert result["collision_per_1k_hrs"] >= 0.0
