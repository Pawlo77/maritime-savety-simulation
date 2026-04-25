"""Results exploration page."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from maritime_mesh.dashboard.constants import (
    KPI_COLUMNS,
    KPI_DESCRIPTIONS,
    KPI_HIGHER_IS_BETTER,
    display_method_name,
    display_scenario_name,
)
from maritime_mesh.dashboard.data_access import load_summary
from maritime_mesh.dashboard.ui import (
    apply_plotly_theme,
    info_panel,
    page_intro,
    render_dataframe,
    section_intro,
)
from maritime_mesh.experiment.analysis import StatisticalAnalyser


def _render_kpi_cards(filtered: pd.DataFrame) -> None:
    """Render compact KPI cards for quick orientation."""
    available = [column for column in KPI_COLUMNS if column in filtered.columns]
    if not available:
        return
    means = filtered[available].mean(numeric_only=True)
    stds = filtered[available].std(numeric_only=True).fillna(0.0)
    cols = st.columns(min(3, len(available)))
    for index, kpi in enumerate(available):
        with cols[index % len(cols)]:
            st.markdown(
                (
                    "<div class='mm-card'>"
                    f"<div class='mm-card-label'>{kpi}</div>"
                    f"<div class='mm-card-value'>{means[kpi]:.4f}</div>"
                    f"<div class='mm-muted'>std: {stds[kpi]:.4f}</div>"
                    f"<div class='mm-muted'>{KPI_DESCRIPTIONS.get(kpi, '')}</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )


def _format_confidence_table(
    filtered: pd.DataFrame,
    kpi: str,
    baseline_method: str,
) -> pd.DataFrame:
    """Build method ranking table with uncertainty and baseline deltas."""
    grouped = (
        filtered.groupby("method")[kpi]
        .agg(["count", "mean", "std"])
        .rename(columns={"count": "n", "mean": "mean", "std": "std"})
        .reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["ci95_halfwidth"] = np.where(
        grouped["n"] > 1,
        1.96 * grouped["std"] / np.sqrt(grouped["n"]),
        0.0,
    )
    grouped["ci95_low"] = grouped["mean"] - grouped["ci95_halfwidth"]
    grouped["ci95_high"] = grouped["mean"] + grouped["ci95_halfwidth"]
    baseline_mean = grouped.loc[grouped["method"] == baseline_method, "mean"]
    baseline = float(baseline_mean.iloc[0]) if not baseline_mean.empty else float("nan")
    grouped["delta_vs_baseline"] = grouped["mean"] - baseline
    grouped["delta_pct_vs_baseline"] = np.where(
        abs(baseline) > 1e-9,
        (grouped["delta_vs_baseline"] / baseline) * 100.0,
        np.nan,
    )
    ascending = not KPI_HIGHER_IS_BETTER.get(kpi, True)
    grouped = grouped.sort_values("mean", ascending=ascending).reset_index(drop=True)
    grouped.insert(0, "rank", grouped.index + 1)
    return grouped


def _significance_badges(
    filtered: pd.DataFrame,
    scenario: str,
    kpi: str,
    baseline_method: str,
) -> pd.DataFrame:
    """Compute method-vs-baseline significance markers."""
    methods = [
        method for method in sorted(filtered["method"].unique()) if method != baseline_method
    ]
    comparisons = [
        {
            "hypothesis": f"{method} vs {baseline_method}",
            "scenario": scenario,
            "kpi": kpi,
            "condition_a": method,
            "condition_b": baseline_method,
        }
        for method in methods
    ]
    if not comparisons:
        return pd.DataFrame()
    analyser = StatisticalAnalyser(results_df=filtered)
    report = analyser.evaluate_comparisons(comparisons=comparisons, apply_holm_correction=True)
    report["significant"] = report["confirmed"].astype(bool)
    return report[["condition_a", "condition_b", "p_value", "p_value_corrected", "significant"]]


def _render_seed_outliers(filtered: pd.DataFrame, kpi: str) -> None:
    """Show top/bottom seeds for quick run-level diagnosis."""
    section_intro(
        "Seed Outlier Drill-Down",
        "Review extreme seeds by method to diagnose instability and edge-case behavior.",
    )
    top_n = st.slider("Top/Bottom seeds per method", min_value=1, max_value=10, value=3)
    ascending = not KPI_HIGHER_IS_BETTER.get(kpi, True)
    ranked = filtered.sort_values(kpi, ascending=ascending)
    top = ranked.groupby("method", as_index=False).head(top_n).assign(bucket="Top")
    bottom = ranked.groupby("method", as_index=False).tail(top_n).assign(bucket="Bottom")
    outliers = pd.concat([top, bottom], ignore_index=True).sort_values(["method", "bucket", "seed"])
    render_dataframe(
        outliers[["method", "seed", "scenario", kpi, "bucket"]],
        width="stretch",
        hide_index=True,
    )


def render(output_dir: Path) -> None:
    """Render results page."""
    try:
        with st.spinner("Loading summary results..."):
            results = load_summary(output_dir)
    except ValueError as exc:
        st.error(str(exc))
        return
    if results.empty:
        st.warning(
            "No summary.csv found in the selected output directory. "
            "Run `Run Experiment Matrix` in the Run page first."
        )
        info_panel(
            "Next Best Action",
            (
                "1) Confirm Output directory points to your latest run artifacts. "
                "2) Launch at least one scenario-method-seed matrix from Run. "
                "3) Return here after summary.csv is generated."
            ),
        )
        return
    page_intro(
        "Results Overview",
        (
            "Compare methods and scenarios, inspect KPI distributions, "
            "and validate significance against a baseline."
        ),
    )
    info_panel(
        "How To Read",
        (
            "Use Scenario view and method filters first, then focus on one KPI "
            "for ranking, significance, and outlier diagnosis."
        ),
    )
    scenario_mode = st.radio(
        "Scenario view",
        ["Single scenario", "Side-by-side comparison"],
        horizontal=True,
        key="results_scenario_mode",
    )
    scenarios = sorted(results["scenario"].unique())
    if scenario_mode == "Single scenario":
        selected_scenarios = [
            st.selectbox(
                "Scenario",
                scenarios,
                format_func=display_scenario_name,
                key="results_single_scenario",
            )
        ]
    else:
        selected_scenarios = st.multiselect(
            "Scenarios",
            scenarios,
            default=scenarios[: min(2, len(scenarios))],
            format_func=display_scenario_name,
            key="results_multi_scenarios",
        )
        if not selected_scenarios:
            st.info("Select at least one scenario.")
            return

    method_filter = st.multiselect(
        "Methods to display",
        sorted(results["method"].unique()),
        default=sorted(results["method"].unique()),
        format_func=display_method_name,
        key="results_method_filter",
    )
    filtered = results[
        (results["scenario"].isin(selected_scenarios)) & (results["method"].isin(method_filter))
    ]
    if filtered.empty:
        st.warning("No rows match current scenario/method filters.")
        info_panel(
            "Next Best Action",
            ("Reset filters, then re-apply gradually: scenario first, methods second, KPIs last."),
        )
        return
    baseline_method = st.selectbox(
        "Baseline method for deltas/significance",
        sorted(filtered["method"].unique()),
        format_func=display_method_name,
        key="results_baseline_method",
    )
    available_kpis = [column for column in KPI_COLUMNS if column in filtered.columns]
    selected_kpis = st.multiselect(
        "KPIs to visualize",
        available_kpis,
        default=["survival_ratio"] if "survival_ratio" in available_kpis else available_kpis[:1],
        key="results_selected_kpis",
    )
    default_focus_kpi = (
        "survival_ratio"
        if "survival_ratio" in selected_kpis
        else selected_kpis[0]
        if selected_kpis
        else None
    )
    focus_kpi = (
        st.selectbox(
            "Focus KPI",
            selected_kpis,
            index=selected_kpis.index(default_focus_kpi),
            key="results_focus_kpi",
        )
        if selected_kpis
        else None
    )
    if st.button("Reset result filters", width="content"):
        st.session_state["results_scenario_mode"] = "Single scenario"
        st.session_state["results_method_filter"] = sorted(results["method"].unique())
        st.session_state["results_selected_kpis"] = (
            ["survival_ratio"] if "survival_ratio" in available_kpis else available_kpis[:1]
        )
        st.rerun()
    active_scenarios_label = ", ".join(display_scenario_name(name) for name in selected_scenarios)
    active_methods_label = ", ".join(
        display_method_name(name) for name in sorted(filtered["method"].unique())
    )
    focus_label = focus_kpi or "none"
    st.markdown(
        (
            "<span class='mm-badge mm-badge-success'>"
            f"Active scenarios: {active_scenarios_label}</span>"
            "<span class='mm-badge mm-badge-success'>"
            f"Active methods: {active_methods_label}</span>"
            "<span class='mm-badge mm-badge-warning'>"
            f"Baseline: {display_method_name(baseline_method)}</span>"
            "<span class='mm-badge mm-badge-warning'>"
            f"Focus KPI: {focus_label}</span>"
        ),
        unsafe_allow_html=True,
    )
    jump_a, jump_b = st.columns(2)
    with jump_a:
        if st.button("Open this context in Map Playback", width="stretch"):
            st.session_state["playback_scenario"] = selected_scenarios[0]
            st.session_state["playback_method_filter"] = sorted(filtered["method"].unique())
            st.session_state["playback_method"] = baseline_method
            if hasattr(st, "switch_page"):
                st.switch_page("map-playback")
            else:
                st.success("Playback context prepared. Open Map Playback tab.")
    with jump_b:
        if st.button("Open this context in Hypothesis", width="stretch"):
            st.session_state["hypothesis_scenarios"] = selected_scenarios
            st.session_state["hypothesis_method_a"] = baseline_method
            if hasattr(st, "switch_page"):
                st.switch_page("hypothesis")
            else:
                st.success("Hypothesis context prepared. Open Hypothesis tab.")

    tab_overview, tab_explorer, tab_ranking, tab_significance, tab_outliers, tab_means = st.tabs(
        [
            "Results Overview",
            "KPI Explorer",
            "Method Ranking and Uncertainty",
            "Significance vs Baseline",
            "Seed Outlier Drill-Down",
            "Method Mean/Std Table",
        ]
    )

    with tab_overview:
        section_intro(
            "Snapshot",
            (
                "Top cards show average KPI values for active filters; "
                "use table below for per-seed details."
            ),
        )
        _render_kpi_cards(filtered)
        render_dataframe(filtered, width="stretch", hide_index=True)

    with tab_explorer:
        if not selected_kpis:
            st.info("Select at least one KPI.")
        else:
            section_intro(
                "Distribution Explorer",
                "Boxplots show spread/outliers; bars show method means to support quick ranking.",
            )
            tab_single, tab_multi = st.tabs(["Single KPI detail", "Compare multiple KPIs"])
            with tab_single:
                kpi = st.selectbox("KPI", selected_kpis, key="explorer_single_kpi")
                col_a, col_b = st.columns(2)
                with col_a:
                    fig_box = px.box(
                        filtered,
                        x="method" if scenario_mode == "Single scenario" else "scenario",
                        y=kpi,
                        points="all",
                        hover_data=["seed", "scenario"],
                    )
                    fig_box.update_traces(
                        marker={"color": "#35b8e7", "line": {"color": "#8fdfff", "width": 0.6}},
                        line={"color": "#6fd6ff", "width": 2},
                    )
                    fig_box.update_layout(
                        showlegend=False,
                        title=f"{kpi}: distribution by {'method' if scenario_mode == 'Single scenario' else 'scenario'}",  # noqa: E501
                    )
                    apply_plotly_theme(fig_box, height=420)
                    st.plotly_chart(fig_box, width="stretch")
                with col_b:
                    stats = (
                        filtered.groupby("method", as_index=False)[kpi]
                        .agg(["mean", "std"])
                        .reset_index()
                        .rename(columns={"mean": "kpi_mean", "std": "kpi_std"})
                    )
                    stats["kpi_std"] = stats["kpi_std"].fillna(0.0)
                    stats = stats.sort_values(
                        "kpi_mean", ascending=not KPI_HIGHER_IS_BETTER.get(kpi, True)
                    )
                    fig_mean = px.bar(
                        stats,
                        x="method",
                        y="kpi_mean",
                        error_y="kpi_std",
                        text_auto=".3f",
                        barmode="group",
                        labels={"kpi_mean": f"{kpi} mean", "kpi_std": f"{kpi} std"},
                    )
                    fig_mean.update_traces(marker_color="#35b8e7")
                    fig_mean.update_layout(showlegend=False, title=f"{kpi}: mean with std")
                    apply_plotly_theme(fig_mean, height=420)
                    st.plotly_chart(fig_mean, width="stretch")
            with tab_multi:
                long_df = filtered.melt(
                    id_vars=["scenario", "method", "seed"],
                    value_vars=selected_kpis,
                    var_name="kpi",
                    value_name="value",
                )
                stats_long = (
                    long_df.groupby(["kpi", "method"], as_index=False)["value"]
                    .agg(["mean", "std"])
                    .reset_index()
                    .rename(columns={"mean": "value_mean", "std": "value_std"})
                )
                stats_long["value_std"] = stats_long["value_std"].fillna(0.0)
                fig_multi = px.bar(
                    stats_long,
                    x="method",
                    y="value_mean",
                    error_y="value_std",
                    facet_col="kpi",
                    facet_col_wrap=2,
                    barmode="group",
                    text_auto=".3f",
                    labels={"value_mean": "mean", "value_std": "std"},
                )
                fig_multi.update_traces(marker_color="#35b8e7")
                fig_multi.update_layout(showlegend=False, title="KPI means with std by method")
                apply_plotly_theme(fig_multi, height=700)
                st.plotly_chart(fig_multi, width="stretch")

    with tab_ranking:
        if not focus_kpi:
            st.info("Select at least one KPI.")
        else:
            section_intro(
                "Method Ranking And Uncertainty",
                (
                    "CI95 bounds quantify estimate uncertainty; delta columns "
                    "compare each method to the baseline."
                ),
            )
            ranking_table = _format_confidence_table(
                filtered=filtered,
                kpi=focus_kpi,
                baseline_method=baseline_method,
            )
            render_dataframe(
                ranking_table[
                    [
                        "rank",
                        "method",
                        "n",
                        "mean",
                        "std",
                        "ci95_low",
                        "ci95_high",
                        "delta_vs_baseline",
                        "delta_pct_vs_baseline",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )

    with tab_significance:
        if scenario_mode != "Single scenario":
            st.info("Significance view is available in Single scenario mode.")
        elif not focus_kpi:
            st.info("Select at least one KPI.")
        else:
            section_intro(
                "Significance Vs Baseline",
                (
                    "Corrected p-values (< 0.05) indicate statistically significant "
                    "differences after Holm correction."
                ),
            )
            significance = _significance_badges(
                filtered=filtered,
                scenario=selected_scenarios[0],
                kpi=focus_kpi,
                baseline_method=baseline_method,
            )
            if significance.empty:
                st.info("No non-baseline methods available for significance testing.")
            else:
                render_dataframe(significance, width="stretch", hide_index=True)

    with tab_outliers:
        if not focus_kpi:
            st.info("Select at least one KPI.")
        else:
            section_intro(
                "Seed-Level Diagnostics",
                (
                    "Investigate best/worst seeds to detect instability, "
                    "anomalous runs, or scenario-specific failure modes."
                ),
            )
            _render_seed_outliers(filtered=filtered, kpi=focus_kpi)

    with tab_means:
        section_intro(
            "Method Mean/Std Table",
            "Compact scenario-method aggregation (mean and std) for reporting and export.",
        )
        stats_table = (
            filtered.groupby(["scenario", "method"], as_index=False)[available_kpis]
            .agg(["mean", "std"])
            .reset_index()
        )
        stats_table.columns = [
            (
                f"{column[0]}_{column[1]}"
                if isinstance(column, tuple) and column[1]
                else column[0]
                if isinstance(column, tuple)
                else column
            )
            for column in stats_table.columns
        ]
        render_dataframe(stats_table, width="stretch", hide_index=True)
