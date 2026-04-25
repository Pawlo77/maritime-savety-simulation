# maritime_mesh

Reproducible multi-agent maritime weather mesh simulation package.

## Quick Start

```bash
uv sync
uv run pytest -q
uv run python -c "from maritime_mesh.experiment.runner import ExperimentRunner; ExperimentRunner(n_seeds=1).run_all()"
uv run streamlit run src/maritime_mesh/dashboard/app.py
```

## Project Structure

- `src/maritime_mesh/` - simulation package
- `tests/unit/` - module-level unit tests
- `tests/integration/` - scenario and runner integration tests
- `outputs/maritime_mesh/` - generated summaries and per-run parquet logs
