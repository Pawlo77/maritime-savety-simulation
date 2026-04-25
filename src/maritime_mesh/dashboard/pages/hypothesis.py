"""Hypothesis testing page."""

from pathlib import Path

import streamlit as st

from maritime_mesh.dashboard.constants import KPI_COLUMNS
from maritime_mesh.dashboard.data_access import load_summary
from maritime_mesh.enums import MethodCondition
from maritime_mesh.experiment.analysis import StatisticalAnalyser


def render(output_dir: Path) -> None:
    """Render hypothesis-testing table."""
    try:
        results = load_summary(output_dir)
    except ValueError as exc:
        st.error(str(exc))
        return
    if results.empty:
        st.warning("No summary.csv found. Run experiments first.")
        return
    st.markdown("### Hypothesis Tests")
    st.markdown(
        "Configure scenario/KPI/method comparisons and run non-parametric tests "
        "with Holm-corrected p-values."
    )
    analyser = StatisticalAnalyser(results_df=results)
    scenarios = sorted(results["scenario"].unique())
    methods = [
        condition.value
        for condition in MethodCondition
        if condition.value in set(results["method"])
    ]
    kpis = [kpi for kpi in KPI_COLUMNS if kpi in set(results.columns)]
    col_a, col_b = st.columns(2)
    with col_a:
        selected_scenarios = st.multiselect("Scenarios", scenarios, default=scenarios)
        selected_kpis = st.multiselect("KPIs", kpis, default=kpis[: min(2, len(kpis))])
    with col_b:
        method_a = st.selectbox("Condition A", methods, index=methods.index("proposed"))
        default_b = "baseline_a" if "baseline_a" in methods else methods[0]
        method_b = st.selectbox("Condition B", methods, index=methods.index(default_b))
    if method_a == method_b:
        st.info("Condition A and Condition B must differ.")
        return
    comparisons = [
        {
            "hypothesis": f"{scenario}:{kpi}:{method_a}_vs_{method_b}",
            "scenario": scenario,
            "kpi": kpi,
            "condition_a": method_a,
            "condition_b": method_b,
        }
        for scenario in selected_scenarios
        for kpi in selected_kpis
    ]
    report = analyser.evaluate_comparisons(comparisons=comparisons, apply_holm_correction=True)
    if report.empty:
        st.info("No valid comparisons for current filters.")
        return
    st.dataframe(report, use_container_width=True, hide_index=True)
