from maritime_mesh.experiment.runner import ExperimentRunner


def test_run_all_matrix_shape(tmp_path) -> None:
    runner = ExperimentRunner(n_seeds=2, output_dir=tmp_path)
    df = runner.run_all()
    assert len(df) == 4 * 3 * 2
