"""Statistical analysis utilities for experiment results."""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from maritime_mesh.enums import MethodCondition

KPI_HIGHER_IS_BETTER = {
    "fatal_per_1k_hrs": False,
    "collision_per_1k_hrs": False,
    "survival_ratio": True,
    "avg_tta_hours": False,
    "evac_activation_rate": True,
    "mean_p_prep": True,
}


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
        min_improvement_pct: float | None = None,
        kpi_higher_is_better: bool | None = None,
    ) -> dict[str, float]:
        """Run Mann-Whitney U, Cliff's delta, and bootstrap CI for delta series."""
        filtered = self.results_df[self.results_df["scenario"] == scenario_name]
        sample_a = filtered[filtered["method"] == condition_a.value][kpi].to_numpy()
        sample_b = filtered[filtered["method"] == condition_b.value][kpi].to_numpy()
        if len(sample_a) == 0 or len(sample_b) == 0:
            return {
                "p_value": float("nan"),
                "cliffs_delta": float("nan"),
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "improvement_pct": float("nan"),
                "threshold_met": float("nan"),
                "confirmed": 0.0,
            }
        _, p_value = mannwhitneyu(sample_a, sample_b, alternative="two-sided")
        delta = _cliffs_delta(sample_a, sample_b)
        ci_lower, ci_upper = self._bootstrap_ci(sample_a - sample_b)
        mean_a = float(np.mean(sample_a))
        mean_b = float(np.mean(sample_b))
        higher_is_better = (
            bool(kpi_higher_is_better)
            if kpi_higher_is_better is not None
            else KPI_HIGHER_IS_BETTER.get(kpi, False)
        )
        if np.isclose(mean_b, 0.0):
            improvement_pct = float("nan")
        else:
            direction = (mean_a - mean_b) if higher_is_better else (mean_b - mean_a)
            improvement_pct = (direction / abs(mean_b)) * 100.0
        threshold_met = (
            float(improvement_pct >= float(min_improvement_pct))
            if min_improvement_pct is not None and np.isfinite(improvement_pct)
            else 1.0
        )
        significance_met = bool(p_value < 0.05 and (kpi == "survival_ratio" or abs(delta) > 0.2))
        confirmed = bool(significance_met and threshold_met >= 1.0)
        return {
            "p_value": float(p_value),
            "cliffs_delta": float(delta),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "improvement_pct": float(improvement_pct),
            "threshold_met": float(threshold_met),
            "confirmed": float(confirmed),
        }

    @staticmethod
    def _holm_adjust(p_values: list[float]) -> list[float]:
        """Holm-Bonferroni correction for multiple comparisons."""
        indexed = sorted(enumerate(p_values), key=lambda item: item[1])
        adjusted = [float("nan")] * len(p_values)
        running_max = 0.0
        m = len(p_values)
        for rank, (index, value) in enumerate(indexed):
            corrected = min(1.0, (m - rank) * value)
            running_max = max(running_max, corrected)
            adjusted[index] = running_max
        return adjusted

    def evaluate_comparisons(
        self,
        comparisons: list[dict],
        apply_holm_correction: bool = True,
    ) -> pd.DataFrame:
        """Evaluate arbitrary comparison list and return report dataframe."""
        rows = []
        for comparison in comparisons:
            cond_a = MethodCondition(comparison["condition_a"])
            cond_b = MethodCondition(comparison["condition_b"])
            outcome = self.test_hypothesis(
                kpi=comparison["kpi"],
                condition_a=cond_a,
                condition_b=cond_b,
                scenario_name=comparison["scenario"],
                min_improvement_pct=comparison.get("min_improvement_pct"),
                kpi_higher_is_better=comparison.get("kpi_higher_is_better"),
            )
            rows.append(
                {
                    "hypothesis": comparison.get("hypothesis", ""),
                    "scenario": comparison["scenario"],
                    "kpi": comparison["kpi"],
                    "condition_a": cond_a.value,
                    "condition_b": cond_b.value,
                    "min_improvement_pct": comparison.get("min_improvement_pct"),
                    **outcome,
                }
            )
        report = pd.DataFrame(rows)
        if report.empty:
            return report
        if apply_holm_correction:
            p_values = report["p_value"].fillna(1.0).astype(float).tolist()
            report["p_value_corrected"] = self._holm_adjust(p_values)
            report["confirmed"] = (
                (report["p_value_corrected"] < 0.05)
                & ((report["kpi"] == "survival_ratio") | (report["cliffs_delta"].abs() > 0.2))
                & report["threshold_met"].fillna(0.0).ge(1.0)
            ).astype(float)
        return report

    def full_report(
        self,
        comparisons: list[dict] | None = None,
        apply_holm_correction: bool = True,
    ) -> pd.DataFrame:
        """Produce compact hypothesis report across scenario-specific KPIs."""
        default_comparisons = [
            {
                "scenario": "scenario_2_storm_corridor",
                "kpi": "fatal_per_1k_hrs",
                "condition_a": MethodCondition.PROPOSED.value,
                "condition_b": MethodCondition.BASELINE_A.value,
                "hypothesis": "H1",
                "min_improvement_pct": 20.0,
                "kpi_higher_is_better": False,
            },
            {
                "scenario": "scenario_4_deep_water_rescue",
                "kpi": "survival_ratio",
                "condition_a": MethodCondition.PROPOSED.value,
                "condition_b": MethodCondition.BASELINE_A.value,
                "hypothesis": "H2",
                "min_improvement_pct": 15.0,
                "kpi_higher_is_better": True,
            },
            {
                "scenario": "scenario_3_blind_shore",
                "kpi": "fatal_per_1k_hrs",
                "condition_a": MethodCondition.PROPOSED.value,
                "condition_b": MethodCondition.BASELINE_B.value,
                "hypothesis": "H3",
                "kpi_higher_is_better": False,
            },
        ]
        return self.evaluate_comparisons(
            comparisons=comparisons or default_comparisons,
            apply_holm_correction=apply_holm_correction,
        )
