"""Integration tests for experiment runner matrix execution."""

from maritime_mesh.experiment.runner import ExperimentRunner


def test_run_all_matrix_shape(tmp_path) -> None:
    """Assert matrix size equals scenario x method x seed combinations."""
    runner = ExperimentRunner(n_seeds=2, output_dir=tmp_path)
    df = runner.run_all()
    assert len(df) == 4 * 3 * 2


def test_run_all_writes_run_manifest(tmp_path) -> None:
    """Each run should persist a manifest for reproducibility."""
    runner = ExperimentRunner(n_seeds=1, output_dir=tmp_path)
    runner.run_all()
    manifests = list(tmp_path.glob("*.manifest.json"))
    assert manifests
