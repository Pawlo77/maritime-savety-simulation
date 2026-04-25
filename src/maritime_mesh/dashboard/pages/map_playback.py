"""Map playback page."""

from pathlib import Path

import plotly.express as px
import streamlit as st

from maritime_mesh.dashboard.constants import (
    KPI_COLUMNS,
    KPI_HIGHER_IS_BETTER,
    display_method_name,
    display_scenario_name,
)
from maritime_mesh.dashboard.data_access import can_render_map, load_run_log, load_summary
from maritime_mesh.dashboard.map_view import make_timeline_map
from maritime_mesh.dashboard.ui import apply_plotly_theme, info_panel, page_intro, section_intro


def render(output_dir: Path) -> None:
    """Render run playback map controls."""
    try:
        with st.spinner("Loading run summaries..."):
            results = load_summary(output_dir)
    except ValueError as exc:
        st.error(str(exc))
        return
    if results.empty:
        st.warning(
            "No summary.csv found in the selected output directory. "
            "Run at least one experiment first."
        )
        return
    page_intro(
        "Simulation Playback",
        (
            "Replay a specific scenario/method/seed run and inspect vessel movement, "
            "communication links, and events over time."
        ),
    )
    info_panel(
        "Map Glossary",
        (
            "Preparedness score is vessel readiness; risk blend score is fused "
            "hazard estimate; shore signal indicates coastal message reception "
            "at current tick. Weather layer selector controls the heatmap metric."
        ),
    )
    scenario = st.selectbox(
        "Scenario",
        sorted(results["scenario"].unique()),
        format_func=display_scenario_name,
        key="playback_scenario",
    )
    method_filter = st.multiselect(
        "Methods to display",
        sorted(results["method"].unique()),
        default=sorted(results["method"].unique()),
        format_func=display_method_name,
        key="playback_method_filter",
    )
    if st.button("Reset playback filters", width="content"):
        st.session_state["playback_method_filter"] = sorted(results["method"].unique())
        st.session_state["playback_seed_mode"] = "Manual seed"
        st.rerun()
    filtered = results[(results["scenario"] == scenario) & (results["method"].isin(method_filter))]
    if filtered.empty:
        st.info("No rows match current scenario/method filters.")
        return
    selected_method = st.selectbox(
        "Method for playback",
        sorted(filtered["method"].unique()),
        format_func=display_method_name,
        key="playback_method",
    )
    method_subset = filtered[filtered["method"] == selected_method]
    seed_candidates = sorted(method_subset["seed"].unique())
    section_intro(
        "Seed Selection",
        (
            "Manual mode picks an exact seed; shortcut modes auto-pick "
            "highest/lowest seed for the selected KPI."
        ),
    )
    shortcut_mode = st.radio(
        "Seed selection",
        ["Manual seed", "Top seed by KPI", "Bottom seed by KPI"],
        horizontal=True,
        key="playback_seed_mode",
    )
    selected_seed: int
    if shortcut_mode == "Manual seed":
        selected_seed = int(st.selectbox("Seed for playback", seed_candidates, key="playback_seed"))
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
        with st.spinner("Loading selected run log..."):
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
            show_links = st.toggle("Show active communication links", value=True)
        with col_b:
            show_markers = st.toggle("Show event markers", value=True)
        weather_layer = st.selectbox(
            "Weather heatmap layer",
            options=["Hazard", "Sea state", "Wind", "Low visibility"],
            index=0,
            help="Choose which weather component to visualize in the map background.",
        )
        linger_ticks = st.slider(
            "Hide collided vessels after this many ticks",
            min_value=0,
            max_value=30,
            value=6,
            help=(
                "After a vessel's first collision, keep it visible only for this many ticks. "
                "Collision X markers remain visible."
            ),
        )
        st.plotly_chart(
            make_timeline_map(
                run_df=run_df,
                show_communication_links=show_links,
                show_event_markers=show_markers,
                collision_vessel_linger_ticks=int(linger_ticks),
                weather_layer=weather_layer,
            ),
            width="stretch",
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
                    markers=True,
                    title="Communication and intervention events per tick",
                )
                apply_plotly_theme(event_fig, height=340)
                st.plotly_chart(event_fig, width="stretch")
