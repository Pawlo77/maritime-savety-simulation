"""Unit tests for configurable hypothesis analysis."""

import pandas as pd

from maritime_mesh.experiment.analysis import StatisticalAnalyser


def test_evaluate_comparisons_returns_corrected_p_values() -> None:
    """Configured comparisons should include Holm-corrected p-values."""
    results = pd.DataFrame(
        {
            "scenario": ["s1"] * 8,
            "method": ["proposed"] * 4 + ["baseline_a"] * 4,
            "seed": [0, 1, 2, 3, 0, 1, 2, 3],
            "survival_ratio": [0.9, 0.92, 0.91, 0.93, 0.5, 0.55, 0.53, 0.52],
            "fatal_per_1k_hrs": [1.0, 1.1, 1.2, 1.0, 3.0, 2.9, 3.1, 2.8],
        }
    )
    analyser = StatisticalAnalyser(results_df=results, rng_seed=1)
    comparisons = [
        {
            "scenario": "s1",
            "kpi": "survival_ratio",
            "condition_a": "proposed",
            "condition_b": "baseline_a",
            "hypothesis": "Hx",
        }
    ]
    report = analyser.evaluate_comparisons(comparisons=comparisons, apply_holm_correction=True)
    assert not report.empty
    assert "p_value_corrected" in report.columns
    assert report.iloc[0]["condition_a"] == "proposed"


def test_evaluate_comparisons_handles_missing_samples() -> None:
    """Missing method data should produce a safe empty-statistics row."""
    results = pd.DataFrame(
        {
            "scenario": ["s1", "s1"],
            "method": ["proposed", "proposed"],
            "seed": [0, 1],
            "survival_ratio": [0.8, 0.85],
        }
    )
    analyser = StatisticalAnalyser(results_df=results)
    report = analyser.evaluate_comparisons(
        comparisons=[
            {
                "scenario": "s1",
                "kpi": "survival_ratio",
                "condition_a": "proposed",
                "condition_b": "baseline_a",
            }
        ]
    )
    assert report.iloc[0]["confirmed"] == 0.0
