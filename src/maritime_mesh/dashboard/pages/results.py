"""Results exploration page."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from maritime_mesh.dashboard.constants import KPI_COLUMNS, KPI_DESCRIPTIONS, KPI_HIGHER_IS_BETTER
from maritime_mesh.dashboard.data_access import load_summary
from maritime_mesh.experiment.analysis import StatisticalAnalyser


def _render_kpi_cards(filtered: pd.DataFrame) -> None:
    """Render compact KPI cards for quick orientation."""
    available = [column for column in KPI_COLUMNS if column in filtered.columns]
    if not available:
        return
    means = filtered[available].mean(numeric_only=True)
    cols = st.columns(min(3, len(available)))
    for index, kpi in enumerate(available):
        with cols[index % len(cols)]:
            st.markdown(
                (
                    "<div class='mm-card'>"
                    f"<div><b>{kpi}</b></div>"
                    f"<div style='font-size:1.2rem'>{means[kpi]:.4f}</div>"
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
    st.markdown("### Seed Outlier Drill-Down")
    top_n = st.slider("Top/Bottom seeds per method", min_value=1, max_value=10, value=3)
    ascending = not KPI_HIGHER_IS_BETTER.get(kpi, True)
    ranked = filtered.sort_values(kpi, ascending=ascending)
    top = ranked.groupby("method", as_index=False).head(top_n).assign(bucket="Top")
    bottom = ranked.groupby("method", as_index=False).tail(top_n).assign(bucket="Bottom")
    outliers = pd.concat([top, bottom], ignore_index=True).sort_values(["method", "bucket", "seed"])
    st.dataframe(
        outliers[["method", "seed", "scenario", kpi, "bucket"]],
        use_container_width=True,
        hide_index=True,
    )


def render(output_dir: Path) -> None:
    """Render results page."""
    try:
        results = load_summary(output_dir)
    except ValueError as exc:
        st.error(str(exc))
        return
    if results.empty:
        st.warning("No summary.csv found. Run experiments first.")
        return
    st.markdown("### Results Overview")
    scenario_mode = st.radio(
        "Scenario view",
        ["Single scenario", "Side-by-side comparison"],
        horizontal=True,
    )
    scenarios = sorted(results["scenario"].unique())
    if scenario_mode == "Single scenario":
        selected_scenarios = [st.selectbox("Scenario", scenarios)]
    else:
        selected_scenarios = st.multiselect(
            "Scenarios",
            scenarios,
            default=scenarios[: min(2, len(scenarios))],
        )
        if not selected_scenarios:
            st.info("Select at least one scenario.")
            return

    method_filter = st.multiselect(
        "Methods to display",
        sorted(results["method"].unique()),
        default=sorted(results["method"].unique()),
    )
    filtered = results[
        (results["scenario"].isin(selected_scenarios)) & (results["method"].isin(method_filter))
    ]
    if filtered.empty:
        st.warning("No rows match current scenario/method filters.")
        return
    baseline_method = st.selectbox(
        "Baseline method for deltas/significance",
        sorted(filtered["method"].unique()),
    )
    available_kpis = [column for column in KPI_COLUMNS if column in filtered.columns]
    selected_kpis = st.multiselect(
        "KPIs to visualize",
        available_kpis,
        default=["survival_ratio"] if "survival_ratio" in available_kpis else available_kpis[:1],
    )
    default_focus_kpi = (
        "survival_ratio"
        if "survival_ratio" in selected_kpis
        else selected_kpis[0]
        if selected_kpis
        else None
    )
    focus_kpi = (
        st.selectbox("Focus KPI", selected_kpis, index=selected_kpis.index(default_focus_kpi))
        if selected_kpis
        else None
    )

    tab_overview, tab_explorer, tab_ranking, tab_significance, tab_outliers, tab_means = st.tabs(
        [
            "Results Overview",
            "KPI Explorer",
            "Method Ranking and Uncertainty",
            "Significance vs Baseline",
            "Seed Outlier Drill-Down",
            "Method Means Table",
        ]
    )

    with tab_overview:
        _render_kpi_cards(filtered)
        st.dataframe(filtered, use_container_width=True, hide_index=True)

    with tab_explorer:
        if not selected_kpis:
            st.info("Select at least one KPI.")
        else:
            tab_single, tab_multi = st.tabs(["Single KPI detail", "Compare multiple KPIs"])
            with tab_single:
                kpi = st.selectbox("KPI", selected_kpis, key="explorer_single_kpi")
                col_a, col_b = st.columns(2)
                with col_a:
                    fig_box = px.box(
                        filtered,
                        x="method" if scenario_mode == "Single scenario" else "scenario",
                        y=kpi,
                        color="method",
                        points="all",
                        hover_data=["seed", "scenario"],
                        template="plotly_white",
                    )
                    fig_box.update_layout(showlegend=False, height=420)
                    st.plotly_chart(fig_box, use_container_width=True)
                with col_b:
                    means = filtered.groupby("method", as_index=False)[kpi].mean()
                    means = means.sort_values(
                        kpi, ascending=not KPI_HIGHER_IS_BETTER.get(kpi, True)
                    )
                    fig_mean = px.bar(
                        means,
                        x="method",
                        y=kpi,
                        color="method",
                        template="plotly_white",
                        text_auto=".3f",
                        barmode="group",
                    )
                    fig_mean.update_layout(showlegend=False, height=420)
                    st.plotly_chart(fig_mean, use_container_width=True)
            with tab_multi:
                long_df = filtered.melt(
                    id_vars=["scenario", "method", "seed"],
                    value_vars=selected_kpis,
                    var_name="kpi",
                    value_name="value",
                )
                means_long = long_df.groupby(["kpi", "method"], as_index=False)["value"].mean()
                fig_multi = px.bar(
                    means_long,
                    x="method",
                    y="value",
                    color="method",
                    facet_col="kpi",
                    facet_col_wrap=2,
                    template="plotly_white",
                    barmode="group",
                    text_auto=".3f",
                )
                fig_multi.update_layout(showlegend=False, height=700)
                st.plotly_chart(fig_multi, use_container_width=True)

    with tab_ranking:
        if not focus_kpi:
            st.info("Select at least one KPI.")
        else:
            ranking_table = _format_confidence_table(
                filtered=filtered,
                kpi=focus_kpi,
                baseline_method=baseline_method,
            )
            st.dataframe(
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
                use_container_width=True,
                hide_index=True,
            )

    with tab_significance:
        if scenario_mode != "Single scenario":
            st.info("Significance view is available in Single scenario mode.")
        elif not focus_kpi:
            st.info("Select at least one KPI.")
        else:
            significance = _significance_badges(
                filtered=filtered,
                scenario=selected_scenarios[0],
                kpi=focus_kpi,
                baseline_method=baseline_method,
            )
            if significance.empty:
                st.info("No non-baseline methods available for significance testing.")
            else:
                st.dataframe(significance, use_container_width=True, hide_index=True)

    with tab_outliers:
        if not focus_kpi:
            st.info("Select at least one KPI.")
        else:
            _render_seed_outliers(filtered=filtered, kpi=focus_kpi)

    with tab_means:
        means_table = filtered.groupby(["scenario", "method"], as_index=False)[
            available_kpis
        ].mean()
        st.dataframe(means_table, use_container_width=True, hide_index=True)
