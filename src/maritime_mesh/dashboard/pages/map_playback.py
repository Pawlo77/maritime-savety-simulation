"""Map playback page."""

from pathlib import Path

import streamlit as st

from maritime_mesh.dashboard.data_access import can_render_map, load_run_log, load_summary
from maritime_mesh.dashboard.map_view import make_timeline_map


def render(output_dir: Path) -> None:
    """Render run playback map controls."""
    results = load_summary(output_dir)
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
    seed_candidates = sorted(filtered[filtered["method"] == selected_method]["seed"].unique())
    selected_seed = st.selectbox("Seed for playback", seed_candidates)
    run_df = load_run_log(
        output_dir=output_dir,
        scenario=scenario,
        method=selected_method,
        seed=int(selected_seed),
    )
    if run_df.empty:
        st.info("Per-run parquet not found for this selection.")
    elif not can_render_map(run_df=run_df):
        st.info(
            "This run log does not contain map-position columns (`x_nm`, `y_nm`). "
            "Re-run experiments with the latest logger to enable playback."
        )
    else:
        st.plotly_chart(make_timeline_map(run_df=run_df), use_container_width=True)
