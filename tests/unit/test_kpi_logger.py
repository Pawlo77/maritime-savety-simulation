"""Unit tests for KPI logger outputs and parquet persistence."""

from maritime_mesh.enums import VesselState
from maritime_mesh.model.kpi_logger import KpiLogger
from maritime_mesh.model.maritime_model import MaritimeModel


def test_compute_kpis_contains_required_fields(default_simulation_config) -> None:
    """Verify all expected KPI keys are present after a simulation run."""
    model = MaritimeModel(default_simulation_config)
    model.run()
    kpis = model.kpi_logger.compute_kpis()
    expected = {
        "fatal_per_1k_hrs",
        "collision_per_1k_hrs",
        "survival_ratio",
        "avg_tta_hours",
        "evac_activation_rate",
        "mean_p_prep",
    }
    assert expected.issubset(kpis.keys())


def test_flush_to_parquet_writes_file(default_simulation_config, tmp_path) -> None:
    """Verify flush operation writes a parquet file to disk."""
    model = MaritimeModel(default_simulation_config)
    model.step()
    target = tmp_path / "log.parquet"
    model.kpi_logger.flush_to_parquet(str(target))
    assert list(tmp_path.glob("log.parquet.part*.parquet"))


def test_survival_ratio_uses_total_exposed_crew_denominator() -> None:
    """Survival ratio should use crew-level exposed denominator."""
    logger = KpiLogger()
    logger.records = [
        {
            "tick": 0,
            "entity_type": "vessel",
            "vessel_id": 1,
            "state": VesselState.ACTIVE.value,
            "n_survivors": 10,
            "has_evacuated": False,
            "p_prep": 0.5,
        },
        {
            "tick": 1,
            "entity_type": "vessel",
            "vessel_id": 1,
            "state": VesselState.SUNK.value,
            "n_survivors": 4,
            "has_evacuated": True,
            "p_prep": 0.5,
        },
        {
            "tick": 0,
            "entity_type": "vessel",
            "vessel_id": 2,
            "state": VesselState.ACTIVE.value,
            "n_survivors": 8,
            "has_evacuated": False,
            "p_prep": 0.5,
        },
        {
            "tick": 1,
            "entity_type": "vessel",
            "vessel_id": 2,
            "state": VesselState.ACTIVE.value,
            "n_survivors": 8,
            "has_evacuated": False,
            "p_prep": 0.5,
        },
    ]
    kpis = logger.compute_kpis()
    assert kpis["survival_ratio"] == 12 / 18
