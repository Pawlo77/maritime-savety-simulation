# maritime_mesh

Reproducible multi-agent maritime weather mesh simulation package.

## Quick Start

```bash
make clean
make install
make test
make gui

# Run the full experiment suite
uv run python -c "from maritime_mesh.experiment.runner import ExperimentRunner; ExperimentRunner(n_seeds=1).run_all()"
```


## Project Structure

- `src/maritime_mesh/` - simulation package
- `tests/unit/` - module-level unit tests
- `tests/integration/` - scenario and runner integration tests
- `outputs/maritime_mesh/` - generated summaries and per-run parquet logs
