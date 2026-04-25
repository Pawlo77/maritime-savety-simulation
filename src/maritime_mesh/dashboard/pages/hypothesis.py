"""Hypothesis testing page."""

from pathlib import Path

import streamlit as st

from maritime_mesh.dashboard.constants import (
    KPI_COLUMNS,
    display_method_name,
    display_scenario_name,
)
from maritime_mesh.dashboard.data_access import load_summary
from maritime_mesh.dashboard.ui import info_panel, page_intro, render_dataframe
from maritime_mesh.enums import MethodCondition
from maritime_mesh.experiment.analysis import StatisticalAnalyser

_HYPOTHESIS_THRESHOLDS: dict[tuple[str, str, str, str], tuple[float, bool]] = {
    ("scenario_2_storm_corridor", "fatal_per_1k_hrs", "proposed", "baseline_a"): (20.0, False),
    ("scenario_4_deep_water_rescue", "survival_ratio", "proposed", "baseline_a"): (15.0, True),
}


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
        info_panel(
            "Next Best Action",
            (
                "1) Confirm output directory points to fresh artifacts. "
                "2) Execute experiment matrix from Run. "
                "3) Return when summary.csv is available."
            ),
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
            "Confirmed comparisons require corrected p-value < 0.05 and, "
            "for non-survival KPIs, |Cliff's delta| > 0.2. "
            "When a minimum improvement threshold is configured, it must also be met."
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
            key="hypothesis_scenarios",
        )
        selected_kpis = st.multiselect(
            "KPIs",
            kpis,
            default=kpis[: min(2, len(kpis))],
            key="hypothesis_kpis",
        )
    with col_b:
        method_a = st.selectbox(
            "Condition A",
            methods,
            index=methods.index("proposed"),
            format_func=display_method_name,
            key="hypothesis_method_a",
        )
        default_b = "baseline_a" if "baseline_a" in methods else methods[0]
        method_b = st.selectbox(
            "Condition B",
            methods,
            index=methods.index(default_b),
            format_func=display_method_name,
            key="hypothesis_method_b",
        )
    if st.button("Reset hypothesis filters", width="content"):
        for key in (
            "hypothesis_scenarios",
            "hypothesis_kpis",
            "hypothesis_method_a",
            "hypothesis_method_b",
        ):
            st.session_state.pop(key, None)
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
    st.markdown(
        (
            "<span class='mm-badge mm-badge-success'>"
            f"Scenarios: {len(selected_scenarios)}</span>"
            "<span class='mm-badge mm-badge-success'>"
            f"KPIs: {', '.join(selected_kpis)}</span>"
            "<span class='mm-badge mm-badge-warning'>"
            f"Pair: {display_method_name(method_a)} vs {display_method_name(method_b)}</span>"
        ),
        unsafe_allow_html=True,
    )
    comparisons = []
    for scenario in selected_scenarios:
        for kpi in selected_kpis:
            comparison = {
                "hypothesis": f"{scenario}:{kpi}:{method_a}_vs_{method_b}",
                "scenario": scenario,
                "kpi": kpi,
                "condition_a": method_a,
                "condition_b": method_b,
            }
            threshold = _HYPOTHESIS_THRESHOLDS.get((scenario, kpi, method_a, method_b))
            if threshold is not None:
                comparison["min_improvement_pct"] = threshold[0]
                comparison["kpi_higher_is_better"] = threshold[1]
            comparisons.append(comparison)
    report = analyser.evaluate_comparisons(comparisons=comparisons, apply_holm_correction=True)
    if report.empty:
        st.info("No valid comparisons for current filters.")
        info_panel(
            "Next Best Action",
            "Choose scenarios/KPIs where both selected methods have completed runs.",
        )
        return
    render_dataframe(report, width="stretch", hide_index=True)
