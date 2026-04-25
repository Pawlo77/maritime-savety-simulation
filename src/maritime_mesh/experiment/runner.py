"""Experiment execution driver."""

import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import pandas as pd

from maritime_mesh.config import SimulationConfig
from maritime_mesh.enums import MethodCondition
from maritime_mesh.experiment import scenarios
from maritime_mesh.model.maritime_model import MaritimeModel


class ExperimentRunner:
    """Run the full scenario-method-seed matrix and collect KPIs."""

    def __init__(
        self,
        n_seeds: int = 30,
        output_dir: Path | None = None,
        scenario_factories: list | None = None,
        methods: list[MethodCondition] | None = None,
        simulation_overrides: dict | None = None,
        scenario_overrides: dict | None = None,
        max_workers: int = 1,
        parallel_backend: str = "process",
    ) -> None:
        """Initialize matrix dimensions and factories."""
        self.scenario_factories = scenario_factories or [
            scenarios.scenario_1_calm_passage,
            scenarios.scenario_2_storm_corridor,
            scenarios.scenario_3_blind_shore,
            scenarios.scenario_4_deep_water_rescue,
        ]
        self.methods = methods or [
            MethodCondition.BASELINE_A,
            MethodCondition.BASELINE_B,
            MethodCondition.PROPOSED,
        ]
        self.n_seeds = n_seeds
        self.output_dir = output_dir or Path("outputs/maritime_mesh")
        self.simulation_overrides = simulation_overrides or {}
        self.scenario_overrides = scenario_overrides or {}
        self.max_workers = max(1, int(max_workers))
        if parallel_backend not in {"process", "thread"}:
            raise ValueError("parallel_backend must be 'process' or 'thread'.")
        self.parallel_backend = parallel_backend

    def _apply_overrides(self, config: SimulationConfig) -> SimulationConfig:
        """Apply GUI/runtime overrides to scenario and simulation configs."""
        scenario_override_keys = {
            key: value
            for key, value in self.scenario_overrides.items()
            if hasattr(config.scenario, key) and value is not None
        }
        sim_override_keys = {
            key: value
            for key, value in self.simulation_overrides.items()
            if hasattr(config, key) and value is not None
        }
        scenario = replace(config.scenario, **scenario_override_keys)
        return replace(config, scenario=scenario, output_dir=self.output_dir, **sim_override_keys)

    def run_single(self, config: SimulationConfig) -> dict[str, float]:
        """Run one simulation and return KPI dictionary."""
        effective_config = self._apply_overrides(config)
        model = MaritimeModel(effective_config)
        kpis = model.run()
        self._write_run_manifest(effective_config)
        parquet_path = self.output_dir / (
            f"{effective_config.scenario.name}_{effective_config.method.value}_"
            f"{effective_config.seed}.parquet"
        )
        model.kpi_logger.flush_to_parquet(str(parquet_path))
        return kpis

    def _write_run_manifest(self, config: SimulationConfig) -> None:
        """Persist effective run configuration for reproducibility."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.output_dir / (
            f"{config.scenario.name}_{config.method.value}_{config.seed}.manifest.json"
        )

        def _serialize(value):
            if isinstance(value, Path):
                return str(value)
            if hasattr(value, "value"):
                return value.value
            if isinstance(value, tuple):
                return [_serialize(item) for item in value]
            if isinstance(value, list):
                return [_serialize(item) for item in value]
            if isinstance(value, dict):
                return {str(key): _serialize(item) for key, item in value.items()}
            if hasattr(value, "__dict__"):
                return {key: _serialize(item) for key, item in value.__dict__.items()}
            return value

        payload = {
            "engine": "maritime_mesh",
            "schema_version": 1,
            "scenario": config.scenario.name,
            "method": config.method.value,
            "seed": config.seed,
            "effective_config": _serialize(config),
        }
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def run_all(self) -> pd.DataFrame:
        """Run all combinations and return KPI dataframe."""
        configs = []
        for scenario_factory in self.scenario_factories:
            for method in self.methods:
                for seed in range(self.n_seeds):
                    configs.append(scenario_factory(method=method, seed=seed))

        rows = []
        if self.max_workers == 1 or len(configs) <= 1:
            for config in configs:
                kpis = self.run_single(config=config)
                rows.append(
                    {
                        "scenario": config.scenario.name,
                        "method": config.method.value,
                        "seed": config.seed,
                        **kpis,
                    }
                )
        else:
            executor_cls = (
                ProcessPoolExecutor if self.parallel_backend == "process" else ThreadPoolExecutor
            )
            with executor_cls(max_workers=self.max_workers) as executor:
                future_to_config = {
                    executor.submit(self.run_single, config): config for config in configs
                }
                for future in as_completed(future_to_config):
                    config = future_to_config[future]
                    kpis = future.result()
                    rows.append(
                        {
                            "scenario": config.scenario.name,
                            "method": config.method.value,
                            "seed": config.seed,
                            **kpis,
                        }
                    )

        rows.sort(key=lambda row: (row["scenario"], row["method"], row["seed"]))
        result = pd.DataFrame(rows)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(self.output_dir / "summary.csv", index=False)
        return result
