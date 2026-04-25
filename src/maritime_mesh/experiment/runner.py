"""Experiment execution driver."""

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import replace
from inspect import signature
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from maritime_mesh.config import SimulationConfig
from maritime_mesh.enums import MethodCondition
from maritime_mesh.experiment import scenarios
from maritime_mesh.logging_config import configure_logging
from maritime_mesh.model.maritime_model import MaritimeModel

LOGGER = logging.getLogger(__name__)


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
        configure_logging()
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
        LOGGER.info(
            "ExperimentRunner initialized (seeds=%s, workers=%s, backend=%s, output_dir=%s)",
            self.n_seeds,
            self.max_workers,
            self.parallel_backend,
            self.output_dir,
        )

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

    def run_single(
        self,
        config: SimulationConfig,
        *,
        show_run_progress: bool = False,
        progress_slots: int = 1,
    ) -> dict[str, float]:
        """Run one simulation and return KPI dictionary."""
        effective_config = self._apply_overrides(config)
        LOGGER.info(
            "Run started: scenario=%s method=%s seed=%s ticks=%s",
            effective_config.scenario.name,
            effective_config.method.value,
            effective_config.seed,
            effective_config.n_ticks,
        )
        model = MaritimeModel(effective_config)
        if show_run_progress:
            # Reserve progress row 0 for the global bar in the parent process.
            run_position = 1 + (os.getpid() % max(1, int(progress_slots)))
            run_desc = (
                f"run {effective_config.scenario.name}/"
                f"{effective_config.method.value}/s{effective_config.seed}"
            )
            for _ in tqdm(
                range(effective_config.n_ticks),
                desc=run_desc,
                leave=False,
                position=run_position,
                dynamic_ncols=True,
            ):
                model.step()
            kpis = model.kpi_logger.compute_kpis()
        else:
            kpis = model.run()
        self._write_run_manifest(effective_config)
        parquet_path = self.output_dir / (
            f"{effective_config.scenario.name}_{effective_config.method.value}_"
            f"{effective_config.seed}.parquet"
        )
        model.kpi_logger.flush_to_parquet(str(parquet_path))
        LOGGER.info(
            "Run finished: scenario=%s method=%s seed=%s parquet=%s",
            effective_config.scenario.name,
            effective_config.method.value,
            effective_config.seed,
            parquet_path,
        )
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
        LOGGER.info(
            "Experiment matrix prepared with %s runs (%s scenarios x %s methods x %s seeds)",
            len(configs),
            len(self.scenario_factories),
            len(self.methods),
            self.n_seeds,
        )

        rows = []
        run_single_params = signature(self.run_single).parameters
        supports_show_run_progress = "show_run_progress" in run_single_params
        supports_progress_slots = "progress_slots" in run_single_params
        if self.max_workers == 1 or len(configs) <= 1:
            for config in tqdm(configs, desc="all runs", dynamic_ncols=True):
                run_kwargs: dict[str, object] = {}
                if supports_show_run_progress:
                    run_kwargs["show_run_progress"] = False
                if supports_progress_slots:
                    run_kwargs["progress_slots"] = self.max_workers
                kpis = self.run_single(config=config, **run_kwargs)
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
            show_child_progress = self.parallel_backend == "process"
            with executor_cls(max_workers=self.max_workers) as executor:
                run_kwargs = {}
                if supports_show_run_progress:
                    run_kwargs["show_run_progress"] = show_child_progress
                if supports_progress_slots:
                    run_kwargs["progress_slots"] = self.max_workers
                future_to_config = {
                    executor.submit(
                        self.run_single,
                        config,
                        **run_kwargs,
                    ): config
                    for config in configs
                }
                with tqdm(total=len(configs), desc="all runs", dynamic_ncols=True) as all_bar:
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
                        all_bar.update(1)

        rows.sort(key=lambda row: (row["scenario"], row["method"], row["seed"]))
        result = pd.DataFrame(rows)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(self.output_dir / "summary.csv", index=False)
        LOGGER.info(
            "All runs completed; summary written to %s with %s rows",
            self.output_dir / "summary.csv",
            len(result),
        )
        return result
