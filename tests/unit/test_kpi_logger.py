"""Unit tests for KPI logger outputs and parquet persistence."""

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
    assert target.exists()
