"""Hypothesis testing page."""

from pathlib import Path

import streamlit as st

from maritime_mesh.dashboard.constants import (
    KPI_COLUMNS,
    display_method_name,
    display_scenario_name,
)
from maritime_mesh.dashboard.data_access import load_summary
from maritime_mesh.dashboard.ui import info_panel, page_intro
from maritime_mesh.enums import MethodCondition
from maritime_mesh.experiment.analysis import StatisticalAnalyser


def render(output_dir: Path) -> None:
    """Render hypothesis-testing table."""
    try:
        with st.spinner("Loading summary results..."):
            results = load_summary(output_dir)
    except ValueError as exc:
        st.error(str(exc))
        return
    if results.empty:
        st.warning(
            "No summary.csv found in the selected output directory. "
            "Run experiments first to generate comparison data."
        )
        return
    page_intro(
        "Hypothesis Tests",
        "Run non-parametric pairwise method comparisons across selected scenarios and KPIs.",
    )
    info_panel(
        "Interpretation Guide",
        (
            "Lower corrected p-values indicate stronger evidence. "
            "Confirmed comparisons usually mean corrected p-value < 0.05."
        ),
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
        selected_scenarios = st.multiselect(
            "Scenarios",
            scenarios,
            default=scenarios,
            format_func=display_scenario_name,
        )
        selected_kpis = st.multiselect("KPIs", kpis, default=kpis[: min(2, len(kpis))])
    with col_b:
        method_a = st.selectbox(
            "Condition A",
            methods,
            index=methods.index("proposed"),
            format_func=display_method_name,
        )
        default_b = "baseline_a" if "baseline_a" in methods else methods[0]
        method_b = st.selectbox(
            "Condition B",
            methods,
            index=methods.index(default_b),
            format_func=display_method_name,
        )
    if st.button("Reset hypothesis filters", use_container_width=False):
        st.rerun()
    if not selected_scenarios:
        st.info("Select at least one scenario to run comparisons.")
        return
    if not selected_kpis:
        st.info("Select at least one KPI to run comparisons.")
        return
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
