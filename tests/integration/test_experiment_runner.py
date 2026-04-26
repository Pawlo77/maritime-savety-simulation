"""Integration tests for experiment runner matrix execution."""

from maritime_mesh.experiment.runner import ExperimentRunner


def test_run_all_matrix_shape(tmp_path) -> None:
    """Assert matrix size equals scenario x method x seed combinations."""
    runner = ExperimentRunner(n_seeds=2, output_dir=tmp_path)
    runner.run_single = lambda config: {"dummy_kpi": float(config.seed)}
    df = runner.run_all()
    assert len(df) == 4 * 3 * 2


def test_run_all_writes_run_manifest(tmp_path) -> None:
    """Each run should persist a manifest for reproducibility."""
    runner = ExperimentRunner(n_seeds=1, output_dir=tmp_path)
    runner.run_single = lambda config: (
        runner._write_run_manifest(runner._apply_overrides(config)) or {"dummy_kpi": 0.0}
    )
    runner.run_all()
    manifests = list(tmp_path.glob("*.manifest.json"))
    assert manifests


def test_run_all_supports_parallel_execution(tmp_path) -> None:
    """Runner should execute independent experiments concurrently when configured."""
    runner = ExperimentRunner(
        n_seeds=2,
        output_dir=tmp_path,
        max_workers=4,
        parallel_backend="thread",
    )
    runner.run_single = lambda config: {"dummy_kpi": float(config.seed)}
    df = runner.run_all()
    assert len(df) == 4 * 3 * 2
    assert set(df["dummy_kpi"]) == {0.0, 1.0}
    assert (tmp_path / "summary.csv").exists()
