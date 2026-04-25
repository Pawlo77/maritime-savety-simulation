"""Statistical analysis utilities for experiment results."""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from maritime_mesh.enums import MethodCondition


def _cliffs_delta(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """Compute Cliff's delta effect size."""
    wins = 0
    losses = 0
    for value_a in sample_a:
        wins += int(np.sum(value_a > sample_b))
        losses += int(np.sum(value_a < sample_b))
    return (wins - losses) / float(len(sample_a) * len(sample_b))


class StatisticalAnalyser:
    """Run non-parametric hypothesis tests on KPI outcomes."""

    def __init__(self, results_df: pd.DataFrame, rng_seed: int = 42) -> None:
        """Store result table and bootstrap RNG."""
        self.results_df = results_df
        self.rng = np.random.default_rng(rng_seed)

    def _bootstrap_ci(self, values: np.ndarray, n_bootstrap: int = 1000) -> tuple[float, float]:
        """Bootstrap confidence interval for sample mean."""
        means = []
        for _ in range(n_bootstrap):
            sample = self.rng.choice(values, size=len(values), replace=True)
            means.append(float(np.mean(sample)))
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    def test_hypothesis(
        self,
        kpi: str,
        condition_a: MethodCondition,
        condition_b: MethodCondition,
        scenario_name: str,
    ) -> dict[str, float]:
        """Run Mann-Whitney U, Cliff's delta, and bootstrap CI for delta series."""
        filtered = self.results_df[self.results_df["scenario"] == scenario_name]
        sample_a = filtered[filtered["method"] == condition_a.value][kpi].to_numpy()
        sample_b = filtered[filtered["method"] == condition_b.value][kpi].to_numpy()
        _, p_value = mannwhitneyu(sample_a, sample_b, alternative="two-sided")
        delta = _cliffs_delta(sample_a, sample_b)
        ci_lower, ci_upper = self._bootstrap_ci(sample_a - sample_b)
        confirmed = bool(p_value < 0.05 and (kpi == "survival_ratio" or abs(delta) > 0.2))
        return {
            "p_value": float(p_value),
            "cliffs_delta": float(delta),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "confirmed": float(confirmed),
        }

    def full_report(self) -> pd.DataFrame:
        """Produce compact hypothesis report across scenario-specific KPIs."""
        comparisons = [
            (
                "scenario_2_storm_corridor",
                "fatal_per_1k_hrs",
                MethodCondition.PROPOSED,
                MethodCondition.BASELINE_A,
                "H1",
            ),
            (
                "scenario_4_deep_water_rescue",
                "survival_ratio",
                MethodCondition.PROPOSED,
                MethodCondition.BASELINE_A,
                "H2",
            ),
            (
                "scenario_3_blind_shore",
                "fatal_per_1k_hrs",
                MethodCondition.PROPOSED,
                MethodCondition.BASELINE_B,
                "H3",
            ),
        ]
        rows = []
        for scenario_name, kpi, cond_a, cond_b, hypothesis in comparisons:
            outcome = self.test_hypothesis(kpi, cond_a, cond_b, scenario_name)
            rows.append(
                {
                    "hypothesis": hypothesis,
                    "scenario": scenario_name,
                    "kpi": kpi,
                    "condition_a": cond_a.value,
                    "condition_b": cond_b.value,
                    **outcome,
                }
            )
        return pd.DataFrame(rows)
