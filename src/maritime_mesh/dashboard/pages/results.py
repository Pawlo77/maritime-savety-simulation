"""Results exploration page."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from maritime_mesh.dashboard.constants import KPI_COLUMNS, KPI_DESCRIPTIONS
from maritime_mesh.dashboard.data_access import load_summary


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


def render(output_dir: Path) -> None:
    """Render results page."""
    results = load_summary(output_dir)
    if results.empty:
        st.warning("No summary.csv found. Run experiments first.")
        return
    st.markdown("### Results Overview")
    scenario = st.selectbox("Scenario", sorted(results["scenario"].unique()))
    method_filter = st.multiselect(
        "Methods to display",
        sorted(results["method"].unique()),
        default=sorted(results["method"].unique()),
    )
    filtered = results[(results["scenario"] == scenario) & (results["method"].isin(method_filter))]
    if filtered.empty:
        st.warning("No rows match current scenario/method filters.")
        return
    _render_kpi_cards(filtered)
    st.dataframe(filtered, use_container_width=True)

    st.markdown("### KPI Explorer")
    available_kpis = [column for column in KPI_COLUMNS if column in filtered.columns]
    selected_kpis = st.multiselect(
        "KPIs to visualize",
        available_kpis,
        default=["survival_ratio"] if "survival_ratio" in available_kpis else available_kpis[:1],
    )
    mode = st.radio("Chart mode", ["Single KPI detail", "Compare multiple KPIs"], horizontal=True)
    if not selected_kpis:
        st.info("Select at least one KPI.")
    elif mode == "Single KPI detail":
        kpi = st.selectbox("KPI", selected_kpis)
        col_a, col_b = st.columns(2)
        with col_a:
            fig_box = px.box(
                filtered,
                x="method",
                y=kpi,
                color="method",
                points="all",
                hover_data=["seed", "scenario"],
                template="plotly_white",
            )
            fig_box.update_layout(showlegend=False, height=420)
            st.plotly_chart(fig_box, use_container_width=True)
        with col_b:
            means = (
                filtered.groupby("method", as_index=False)[kpi]
                .mean()
                .sort_values(kpi, ascending=False)
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
    else:
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

    st.markdown("### Method Means Table")
    means_table = filtered.groupby("method", as_index=False)[available_kpis].mean()
    st.dataframe(means_table, use_container_width=True)
