"""Map playback page."""

from pathlib import Path

import plotly.express as px
import streamlit as st

from maritime_mesh.dashboard.constants import KPI_COLUMNS, KPI_HIGHER_IS_BETTER
from maritime_mesh.dashboard.data_access import can_render_map, load_run_log, load_summary
from maritime_mesh.dashboard.map_view import make_timeline_map


def render(output_dir: Path) -> None:
    """Render run playback map controls."""
    try:
        results = load_summary(output_dir)
    except ValueError as exc:
        st.error(str(exc))
        return
    if results.empty:
        st.warning("No summary.csv found. Run experiments first.")
        return
    st.markdown("### Simulation Playback")
    scenario = st.selectbox("Scenario", sorted(results["scenario"].unique()))
    method_filter = st.multiselect(
        "Methods to display",
        sorted(results["method"].unique()),
        default=sorted(results["method"].unique()),
    )
    filtered = results[(results["scenario"] == scenario) & (results["method"].isin(method_filter))]
    if filtered.empty:
        st.info("No rows match current scenario/method filters.")
        return
    selected_method = st.selectbox("Method for playback", sorted(filtered["method"].unique()))
    method_subset = filtered[filtered["method"] == selected_method]
    seed_candidates = sorted(method_subset["seed"].unique())
    shortcut_mode = st.radio(
        "Seed selection",
        ["Manual seed", "Top seed by KPI", "Bottom seed by KPI"],
        horizontal=True,
    )
    selected_seed: int
    if shortcut_mode == "Manual seed":
        selected_seed = int(st.selectbox("Seed for playback", seed_candidates))
    else:
        shortcut_kpi = st.selectbox(
            "KPI for seed shortcut", [k for k in KPI_COLUMNS if k in method_subset]
        )
        choose_max = shortcut_mode == "Top seed by KPI"
        preferred = method_subset.sort_values(
            shortcut_kpi,
            ascending=not choose_max
            if KPI_HIGHER_IS_BETTER.get(shortcut_kpi, True)
            else choose_max,
        ).iloc[0]
        selected_seed = int(preferred["seed"])
        st.caption(
            f"Using seed {selected_seed} selected from {shortcut_mode.lower()} "
            f"({shortcut_kpi}={preferred[shortcut_kpi]:.4f})."
        )
    try:
        run_df = load_run_log(
            output_dir=output_dir,
            scenario=scenario,
            method=selected_method,
            seed=int(selected_seed),
        )
    except ValueError as exc:
        st.error(str(exc))
        return
    if run_df.empty:
        st.info("Per-run parquet not found for this selection.")
    elif not can_render_map(run_df=run_df):
        st.info(
            "This run log does not contain map-position columns (`x_nm`, `y_nm`). "
            "Re-run experiments with the latest logger to enable playback."
        )
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            show_links = st.toggle("Show communication links", value=False)
        with col_b:
            show_markers = st.toggle("Show event markers", value=True)
        st.plotly_chart(
            make_timeline_map(
                run_df=run_df,
                show_communication_links=show_links,
                show_event_markers=show_markers,
            ),
            use_container_width=True,
        )
        if "entity_type" in run_df.columns:
            event_series = (
                run_df[run_df["entity_type"].isin(["communication_link", "intervention_event"])]
                .groupby(["tick", "entity_type"], as_index=False)
                .size()
                .rename(columns={"size": "count"})
            )
            if not event_series.empty:
                event_fig = px.line(
                    event_series,
                    x="tick",
                    y="count",
                    color="entity_type",
                    template="plotly_white",
                    markers=True,
                    title="Communication and intervention events per tick",
                )
                st.plotly_chart(event_fig, use_container_width=True)
