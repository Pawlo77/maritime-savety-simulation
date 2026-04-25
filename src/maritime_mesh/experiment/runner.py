"""Experiment execution driver."""

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
    ) -> None:
        """Initialize matrix dimensions and factories."""
        self.scenario_factories = [
            scenarios.scenario_1_calm_passage,
            scenarios.scenario_2_storm_corridor,
            scenarios.scenario_3_blind_shore,
            scenarios.scenario_4_deep_water_rescue,
        ]
        self.methods = [MethodCondition.BASELINE_A, MethodCondition.BASELINE_B, MethodCondition.PROPOSED]
        self.n_seeds = n_seeds
        self.output_dir = output_dir or Path("outputs/maritime_mesh")

    def run_single(self, config: SimulationConfig) -> dict[str, float]:
        """Run one simulation and return KPI dictionary."""
        model = MaritimeModel(config)
        kpis = model.run()
        parquet_path = self.output_dir / f"{config.scenario.name}_{config.method.value}_{config.seed}.parquet"
        model.kpi_logger.flush_to_parquet(str(parquet_path))
        return kpis

    def run_all(self) -> pd.DataFrame:
        """Run all combinations and return KPI dataframe."""
        rows = []
        for scenario_factory in self.scenario_factories:
            for method in self.methods:
                for seed in range(self.n_seeds):
                    config = scenario_factory(method=method, seed=seed)
                    kpis = self.run_single(config=config)
                    rows.append(
                        {
                            "scenario": config.scenario.name,
                            "method": method.value,
                            "seed": seed,
                            **kpis,
                        }
                    )
        result = pd.DataFrame(rows)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(self.output_dir / "summary.csv", index=False)
        return result
