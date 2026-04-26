# Maritime Safety Mesh Simulation

Reproducible multi-agent maritime weather mesh simulation package.

## Quick Start

```bash
make clean
make install
make test
make gui

# Run the full experiment suite
uv run python -c "from maritime_mesh.experiment.runner import ExperimentRunner; ExperimentRunner(n_seeds=30, max_workers=8, process_max_tasks_per_child=1).run_all()"
```

## First Successful Run Checklist

1. Install and verify:
   - `make clean`
   - `make install`
   - `make test`
2. Generate baseline experiment outputs:
   - `uv run python -c "from maritime_mesh.experiment.runner import ExperimentRunner; ExperimentRunner(n_seeds=30, max_workers=8, process_max_tasks_per_child=1).run_all()"`
3. Confirm artifacts in your output directory (default: `outputs/maritime_mesh/`):
   - `summary.csv` (required for Results/Hypothesis pages)
   - `*.parquet` run logs (required for Map Playback)
   - `*.manifest.json` run manifests (run metadata and schema checks)
4. Launch dashboard:
   - `make gui`
5. In the dashboard:
   - verify the selected output directory points to the same run artifacts,
   - open `Results` to confirm KPI tables/charts are visible.

## CLI and GUI Run Parity

- The CLI runner (`maritime_mesh.experiment.runner.ExperimentRunner`) and dashboard Run page write the same artifact contract (`summary.csv`, parquet logs, manifests).
- Prefer CLI for scripted/reproducible batch runs and CI workflows.
- Prefer GUI for iterative setup tuning (lanes, shore stations, weather/comms settings) and visual preview before launch.


## Project Structure

- `src/maritime_mesh/` - simulation package
- `tests/unit/` - module-level unit tests
- `tests/integration/` - scenario and runner integration tests
- `outputs/maritime_mesh/` - generated summaries and per-run parquet logs
