"""Map playback page."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from maritime_mesh.dashboard.constants import (
    KPI_COLUMNS,
    KPI_HIGHER_IS_BETTER,
    display_method_name,
    display_scenario_name,
)
from maritime_mesh.dashboard.data_access import (
    can_render_map,
    load_run_log,
    load_run_manifest,
    load_summary,
)
from maritime_mesh.dashboard.map_view import make_timeline_map
from maritime_mesh.dashboard.ui import apply_plotly_theme, info_panel, page_intro


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
        info_panel(
            "Next Best Action",
            (
                "1) Verify output directory selection in the sidebar. "
                "2) Run at least one scenario/method/seed matrix. "
                "3) Ensure per-run parquet logs are generated."
            ),
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
    with st.sidebar:
        st.markdown("### Playback Filters")
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
        info_panel(
            "Next Best Action",
            "Reset playback filters and start with one scenario plus all methods.",
        )
        return
    with st.sidebar:
        selected_method = st.selectbox(
            "Method for playback",
            sorted(filtered["method"].unique()),
            format_func=display_method_name,
            key="playback_method",
        )
    method_subset = filtered[filtered["method"] == selected_method]
    seed_candidates = sorted(method_subset["seed"].unique())
    with st.sidebar:
        st.markdown("### Seed Selection")
        st.caption(
            "Manual mode picks an exact seed; shortcut modes auto-pick "
            "highest/lowest seed for the selected KPI."
        )
        shortcut_mode = st.radio(
            "Seed selection",
            ["Manual seed", "Top seed by KPI", "Bottom seed by KPI"],
            horizontal=True,
            key="playback_seed_mode",
        )
    selected_seed: int
    if shortcut_mode == "Manual seed":
        with st.sidebar:
            selected_seed = int(
                st.selectbox("Seed for playback", seed_candidates, key="playback_seed")
            )
    else:
        with st.sidebar:
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
        manifest = load_run_manifest(
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
        effective_config = (
            manifest.get("effective_config", {}) if isinstance(manifest, dict) else {}
        )
        raw_shore_positions = effective_config.get("shore_station_positions")
        if not raw_shore_positions:
            default_shore = effective_config.get("shore_station_position")
            raw_shore_positions = [default_shore] if default_shore else []
        configured_shore_positions = [
            (float(position[0]), float(position[1]))
            for position in raw_shore_positions
            if isinstance(position, list | tuple) and len(position) == 2
        ]
        raw_lane_defs = effective_config.get("lane_definitions") or []
        configured_lane_definitions: list[tuple[str, list[tuple[float, float]]]] = []
        for lane in raw_lane_defs:
            if not isinstance(lane, list | tuple) or len(lane) != 2:
                continue
            lane_name, points = lane
            parsed_points = [
                (float(point[0]), float(point[1]))
                for point in points
                if isinstance(point, list | tuple) and len(point) == 2
            ]
            configured_lane_definitions.append((str(lane_name), parsed_points))
        st.markdown(
            (
                "<span class='mm-badge mm-badge-success'>"
                f"Scenario: {display_scenario_name(scenario)}</span>"
                "<span class='mm-badge mm-badge-success'>"
                f"Method: {display_method_name(selected_method)}</span>"
                "<span class='mm-badge mm-badge-warning'>"
                f"Seed mode: {shortcut_mode}</span>"
                "<span class='mm-badge mm-badge-warning'>"
                f"Seed: {selected_seed}</span>"
                "<span class='mm-badge mm-badge-warning'>"
                f"Configured shores: {len(configured_shore_positions)} "
                "| Live shores: {int(live_station_count)}</span>"
                "<span class='mm-badge mm-badge-warning'>"
                f"Configured lanes: {len(configured_lane_definitions)} "
                "| SOS vessels: {int(sos_count)} "
                "| Rescue assets: {int(rescue_count)}</span>"
                "<span class='mm-badge mm-badge-warning'>"
                "Configured n_vessels: "
                f"{effective_config.get('scenario', {}).get('n_vessels', 'n/a')}</span>"
            ),
            unsafe_allow_html=True,
        )
        with st.sidebar:
            st.markdown("### Playback Layers")
            show_links = st.toggle("Show active communication links", value=True)
            show_markers = st.toggle("Show event markers", value=True)
            show_rescue_assets = st.toggle("Show rescue assets", value=False)
            show_sos_vessels = st.toggle("Show SOS vessels", value=False)
            show_configured_overlays = st.toggle(
                "Show configured shores/routes",
                value=False,
                help="Show manifest-defined shore stations and lane paths.",
            )
            show_land = st.toggle("Show land", value=True)
        run_ticks = (
            int(run_df["tick"].max()) + 1 if "tick" in run_df.columns and not run_df.empty else 0
        )
        default_stride = 1
        if run_ticks > 1200:
            default_stride = 8
        elif run_ticks > 600:
            default_stride = 4
        elif run_ticks > 250:
            default_stride = 2
        with st.sidebar:
            st.markdown("### Playback Speed")
            tick_stride = st.slider(
                "Playback speedup (render every Nth tick)",
                min_value=1,
                max_value=12,
                value=default_stride,
                help=(
                    "Higher values load much faster by reducing animation frames. "
                    "Raw run data remains unchanged."
                ),
            )
            playback_speed_label = st.select_slider(
                "Playback speed",
                options=["Very slow", "Slow", "Normal", "Fast", "Very fast"],
                value="Normal",
                help="Controls animation tempo (frame duration).",
            )
        speed_to_ms = {
            "Very slow": (1000, 120),
            "Slow": (700, 100),
            "Normal": (420, 80),
            "Fast": (220, 50),
            "Very fast": (120, 20),
        }
        frame_duration_ms, transition_duration_ms = speed_to_ms[playback_speed_label]
        vessel_radio_range_nm = float(effective_config.get("vessel_radio_range_nm", 0.0) or 0.0)
        shore_broadcast_radius_nm = float(
            effective_config.get("shore_broadcast_radius_nm", 0.0) or 0.0
        )
        min_spawn_distance_nm = float(effective_config.get("min_spawn_distance_nm", 0.0) or 0.0)
        max_spawn_distance_nm = float(effective_config.get("max_spawn_distance_nm", 0.0) or 0.0)
        with st.sidebar:
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
        map_tab, telemetry_tab = st.tabs(["Map", "Telemetry"])
        with map_tab:
            st.plotly_chart(
                make_timeline_map(
                    run_df=run_df,
                    show_communication_links=show_links,
                    show_event_markers=show_markers,
                    show_rescue_assets=show_rescue_assets,
                    show_sos_vessels=show_sos_vessels,
                    show_configured_overlays=show_configured_overlays,
                    show_land=show_land,
                    collision_vessel_linger_ticks=int(linger_ticks),
                    weather_layer=weather_layer,
                    tick_stride=int(tick_stride),
                    frame_duration_ms=int(frame_duration_ms),
                    transition_duration_ms=int(transition_duration_ms),
                    configured_shore_positions=configured_shore_positions,
                    configured_lane_definitions=configured_lane_definitions,
                    vessel_radio_range_nm=vessel_radio_range_nm,
                    shore_broadcast_radius_nm=shore_broadcast_radius_nm,
                    min_spawn_distance_nm=min_spawn_distance_nm,
                    max_spawn_distance_nm=max_spawn_distance_nm,
                ),
                width="stretch",
            )
    with telemetry_tab:
        vessel_series = run_df[run_df["entity_type"] == "vessel"].copy()
        if not vessel_series.empty and {"vessel_id", "tick", "n_survivors"}.issubset(
            vessel_series.columns
        ):
            vessel_series["vessel_id"] = vessel_series["vessel_id"].astype("int64", errors="ignore")
            vessel_series["n_survivors"] = pd.to_numeric(
                vessel_series["n_survivors"], errors="coerce"
            )
            vessel_series = vessel_series.dropna(subset=["tick", "vessel_id", "n_survivors"])
            vessel_series = vessel_series.sort_values(["vessel_id", "tick"])
            ticks = (
                vessel_series[["tick"]].drop_duplicates().sort_values("tick").reset_index(drop=True)
            )
            initial_by_vessel = (
                vessel_series.groupby("vessel_id", as_index=False)
                .head(1)[["vessel_id", "n_survivors"]]
                .rename(columns={"n_survivors": "initial_survivors"})
                .set_index("vessel_id")
            )
            cumulative_people = ticks.copy()
            cumulative_people["dead_raw"] = 0.0
            cumulative_people["rescued_raw"] = 0.0
            if "state" in vessel_series.columns:
                terminal = vessel_series[
                    vessel_series["state"].astype(str).str.lower().isin(["sunk", "rescued"])
                ][["tick", "vessel_id", "state", "n_survivors"]].copy()
                if not terminal.empty:
                    terminal = (
                        terminal.sort_values(["vessel_id", "tick"])
                        .groupby("vessel_id", as_index=False)
                        .head(1)
                    )
                    terminal = terminal.join(initial_by_vessel, on="vessel_id", how="left")
                    terminal["initial_survivors"] = terminal["initial_survivors"].fillna(
                        terminal["n_survivors"]
                    )
                    terminal["fatalities"] = (
                        terminal["initial_survivors"] - terminal["n_survivors"]
                    ).clip(lower=0.0)
                    dead_events = terminal[terminal["state"].astype(str).str.lower() == "sunk"]
                    if not dead_events.empty:
                        dead_by_tick = dead_events.groupby("tick", as_index=False)[
                            "fatalities"
                        ].sum()
                        cumulative_people = cumulative_people.merge(
                            dead_by_tick.rename(columns={"fatalities": "dead_raw"}),
                            on="tick",
                            how="left",
                            suffixes=("", "_evt"),
                        )
                        if "dead_raw_evt" in cumulative_people.columns:
                            cumulative_people["dead_raw"] = cumulative_people[
                                "dead_raw_evt"
                            ].fillna(0.0)
                            cumulative_people = cumulative_people.drop(columns=["dead_raw_evt"])
                    rescued_events = terminal[
                        terminal["state"].astype(str).str.lower() == "rescued"
                    ]
                    if not rescued_events.empty:
                        rescued_by_tick = rescued_events.groupby("tick", as_index=False)[
                            "n_survivors"
                        ].sum()
                        cumulative_people = cumulative_people.merge(
                            rescued_by_tick.rename(columns={"n_survivors": "rescued_raw"}),
                            on="tick",
                            how="left",
                            suffixes=("", "_evt"),
                        )
                        if "rescued_raw_evt" in cumulative_people.columns:
                            cumulative_people["rescued_raw"] = cumulative_people[
                                "rescued_raw_evt"
                            ].fillna(0.0)
                            cumulative_people = cumulative_people.drop(columns=["rescued_raw_evt"])
            cumulative_people["dead_raw"] = pd.to_numeric(
                cumulative_people["dead_raw"], errors="coerce"
            ).fillna(0.0)
            cumulative_people["rescued_raw"] = pd.to_numeric(
                cumulative_people["rescued_raw"], errors="coerce"
            ).fillna(0.0)
            cumulative_people["dead_cumulative"] = cumulative_people["dead_raw"].cumsum()
            cumulative_people["rescued_cumulative"] = cumulative_people["rescued_raw"].cumsum()
            cumulative_fig = px.line(
                cumulative_people.melt(
                    id_vars=["tick"],
                    value_vars=["dead_cumulative", "rescued_cumulative"],
                    var_name="population",
                    value_name="count",
                ),
                x="tick",
                y="count",
                color="population",
                markers=False,
                title="Cumulative people outcomes: dead vs rescued survivors",
            )
            apply_plotly_theme(cumulative_fig, height=300)
            cumulative_fig.update_layout(
                margin={"l": 24, "r": 24, "t": 96, "b": 24},
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.0,
                    "xanchor": "left",
                    "x": 0.0,
                },
            )
            st.plotly_chart(cumulative_fig, width="stretch")
        if "entity_type" in run_df.columns:
            communication_series = (
                run_df[run_df["entity_type"] == "communication_link"]
                .groupby("tick", as_index=False)
                .size()
                .rename(columns={"size": "count"})
                .assign(metric="communication_link")
            )
            intervention_rows = run_df[run_df["entity_type"] == "intervention_event"].copy()
            intervention_series = pd.DataFrame(columns=["tick", "count", "metric"])
            if not intervention_rows.empty:
                if "event_kind" in intervention_rows.columns:
                    intervention_series = (
                        intervention_rows.groupby(["tick", "event_kind"], as_index=False)
                        .size()
                        .rename(columns={"size": "count", "event_kind": "metric"})
                    )
                else:
                    intervention_series = (
                        intervention_rows.groupby("tick", as_index=False)
                        .size()
                        .rename(columns={"size": "count"})
                        .assign(metric="intervention_event")
                    )
            event_series = pd.concat([communication_series, intervention_series], ignore_index=True)
            full_tick_index = (
                run_df[["tick"]].dropna().drop_duplicates().sort_values("tick")
                if "tick" in run_df.columns
                else None
            )
            if full_tick_index is not None and not full_tick_index.empty:
                if event_series.empty:
                    event_series = full_tick_index.assign(metric="communication_link", count=0)
                else:
                    base_ticks = full_tick_index.assign(_k=1)
                    entity_types = event_series[["metric"]].drop_duplicates().assign(_k=1)
                    dense = base_ticks.merge(entity_types, on="_k", how="inner").drop(columns="_k")
                    event_series = (
                        dense.merge(event_series, on=["tick", "metric"], how="left")
                        .fillna({"count": 0})
                        .sort_values(["tick", "metric"])
                    )
                event_fig = px.line(
                    event_series,
                    x="tick",
                    y="count",
                    color="metric",
                    markers=True,
                    title="Communication and intervention events per tick",
                )
                apply_plotly_theme(event_fig, height=340)
                event_fig.update_layout(
                    margin={"l": 24, "r": 24, "t": 96, "b": 24},
                    legend={
                        "orientation": "h",
                        "yanchor": "bottom",
                        "y": 1.0,
                        "xanchor": "left",
                        "x": 0.0,
                    },
                )
                st.plotly_chart(event_fig, width="stretch")

        if not vessel_series.empty and "sos_reason" in vessel_series.columns:
            sos_rows = vessel_series.copy()
            if "sos_sent" in sos_rows.columns:
                sos_raw = sos_rows["sos_sent"].infer_objects(copy=False)
                if str(sos_raw.dtype) in {"bool", "boolean"}:
                    sos_mask = sos_raw.astype("boolean").fillna(False)
                else:
                    sos_mask = (
                        sos_raw.astype(str)
                        .str.strip()
                        .str.lower()
                        .map(
                            {
                                "true": True,
                                "1": True,
                                "yes": True,
                                "y": True,
                                "false": False,
                                "0": False,
                                "no": False,
                                "n": False,
                            }
                        )
                        .astype("boolean")
                        .fillna(False)
                    )
                sos_rows = sos_rows[sos_mask]
            sos_rows["sos_reason"] = sos_rows["sos_reason"].astype(str).str.strip()
            sos_rows = sos_rows[sos_rows["sos_reason"] != ""]
            if not sos_rows.empty:
                if {"vessel_id", "tick"}.issubset(sos_rows.columns):
                    sos_rows = (
                        sos_rows.sort_values(["vessel_id", "tick"])
                        .groupby("vessel_id", as_index=False)
                        .head(1)
                    )

                def _reason_bucket(raw: str) -> str:
                    text = raw.strip().lower()
                    if text.startswith("policy_trigger"):
                        return "policy_trigger"
                    if text.startswith("land_collision"):
                        return "land_collision"
                    if text.startswith("collision"):
                        return "collision"
                    if text.startswith("distress"):
                        return "distress"
                    return text if text else "unknown"

                sos_rows["reason_bucket"] = sos_rows["sos_reason"].map(_reason_bucket)
                reason_counts = (
                    sos_rows.groupby("reason_bucket", as_index=False)
                    .size()
                    .rename(columns={"size": "count"})
                    .sort_values("count", ascending=False)
                )
                if not reason_counts.empty:
                    reason_fig = px.pie(
                        reason_counts,
                        names="reason_bucket",
                        values="count",
                        title="SOS calls by reason",
                    )
                    apply_plotly_theme(reason_fig, height=320)
                    reason_fig.update_layout(
                        margin={"l": 24, "r": 24, "t": 96, "b": 24},
                        legend={
                            "orientation": "h",
                            "yanchor": "bottom",
                            "y": 1.0,
                            "xanchor": "left",
                            "x": 0.0,
                            "title": {"text": ""},
                        },
                    )
                    reason_fig.update_traces(textinfo="percent+label")
                    st.plotly_chart(reason_fig, width="stretch")
                policy_rows = sos_rows[sos_rows["reason_bucket"] == "policy_trigger"].copy()
                if not policy_rows.empty:
                    parsed = (
                        policy_rows["sos_reason"]
                        .str.extract(
                            r"p_raw=(?P<p_raw>[-+]?\d*\.?\d+),"
                            r"\s*p_eff=(?P<p_eff>[-+]?\d*\.?\d+),"
                            r"\s*hazard=(?P<hazard>[-+]?\d*\.?\d+),"
                            r"\s*err=(?P<err>[-+]?\d*\.?\d+),"
                            r"\s*prep=(?P<prep>[-+]?\d*\.?\d+)"
                        )
                        .add_prefix("diag_")
                    )
                    policy_rows = pd.concat([policy_rows.reset_index(drop=True), parsed], axis=1)
                    for metric_column in (
                        "diag_p_raw",
                        "diag_p_eff",
                        "diag_hazard",
                        "diag_err",
                        "diag_prep",
                    ):
                        policy_rows[metric_column] = pd.to_numeric(
                            policy_rows[metric_column], errors="coerce"
                        )
                    policy_rows = policy_rows.dropna(
                        subset=["diag_p_raw", "diag_p_eff", "diag_hazard", "diag_err", "diag_prep"]
                    )
                    if not policy_rows.empty:
                        st.caption(
                            "Policy-trigger diagnostics from SOS logs: "
                            "`p_raw` (base probability), `p_eff` (after early-tick damping), "
                            "`hazard`, forecast `err`, and crew `prep`."
                        )
                        diag_left, diag_right = st.columns(2)
                        with diag_left:
                            prob_fig = px.histogram(
                                policy_rows,
                                x="diag_p_eff",
                                nbins=16,
                                title="Policy-trigger effective probability distribution",
                            )
                            apply_plotly_theme(prob_fig, height=300)
                            prob_fig.update_layout(
                                margin={"l": 24, "r": 24, "t": 96, "b": 24},
                                showlegend=False,
                            )
                            st.plotly_chart(prob_fig, width="stretch")
                        with diag_right:
                            driver_fig = px.scatter(
                                policy_rows,
                                x="diag_hazard",
                                y="diag_p_eff",
                                color="diag_prep",
                                size="diag_err",
                                hover_data=["vessel_id", "tick", "diag_p_raw"],
                                title="Policy trigger drivers: hazard vs effective probability",
                            )
                            apply_plotly_theme(driver_fig, height=300)
                            driver_fig.update_layout(
                                margin={"l": 24, "r": 24, "t": 96, "b": 24},
                                legend={"title": {"text": "prep"}},
                            )
                            st.plotly_chart(driver_fig, width="stretch")

        tick_index = (
            run_df[["tick"]].dropna().drop_duplicates().sort_values("tick")
            if "tick" in run_df.columns
            else pd.DataFrame(columns=["tick"])
        )
        vessel_metrics = tick_index.copy()
        vessel_rows = run_df[run_df["entity_type"] == "vessel"].copy()
        if not vessel_rows.empty:
            vessel_metrics = vessel_metrics.merge(
                vessel_rows.groupby("tick", as_index=False)["vessel_id"]
                .nunique()
                .rename(columns={"vessel_id": "vessels_total"}),
                on="tick",
                how="left",
            )
            if "state" in vessel_rows.columns:
                for state in ("active", "evac", "rescued", "sunk"):
                    state_counts = (
                        vessel_rows[vessel_rows["state"].astype(str).str.lower() == state]
                        .groupby("tick", as_index=False)["vessel_id"]
                        .nunique()
                        .rename(columns={"vessel_id": f"vessels_{state}"})
                    )
                    vessel_metrics = vessel_metrics.merge(state_counts, on="tick", how="left")
            if "sos_sent" in vessel_rows.columns:
                sos_counts = (
                    vessel_rows[vessel_rows["sos_sent"]]
                    .groupby("tick", as_index=False)["vessel_id"]
                    .nunique()
                    .rename(columns={"vessel_id": "vessels_sos"})
                )
                vessel_metrics = vessel_metrics.merge(sos_counts, on="tick", how="left")
        for column in vessel_metrics.columns:
            if column != "tick":
                vessel_metrics[column] = pd.to_numeric(
                    vessel_metrics[column], errors="coerce"
                ).fillna(0.0)

        rescue_metrics = tick_index.copy()
        rescue_rows = run_df[run_df["entity_type"] == "rescue_asset"].copy()
        if not rescue_rows.empty:
            rescue_metrics = rescue_metrics.merge(
                rescue_rows.groupby("tick", as_index=False)["entity_id"]
                .nunique()
                .rename(columns={"entity_id": "rescue_assets_total"}),
                on="tick",
                how="left",
            )
            if "is_idle" in rescue_rows.columns:
                idle_raw = rescue_rows["is_idle"].infer_objects(copy=False)
                if str(idle_raw.dtype) in {"bool", "boolean"}:
                    idle_flags = idle_raw.astype("boolean")
                else:
                    # Handle object/string values that can appear in older parquet artifacts.
                    idle_flags = (
                        idle_raw.astype(str)
                        .str.strip()
                        .str.lower()
                        .map(
                            {
                                "true": True,
                                "1": True,
                                "yes": True,
                                "y": True,
                                "false": False,
                                "0": False,
                                "no": False,
                                "n": False,
                            }
                        )
                        .astype("boolean")
                    )
                active_rescue = (
                    rescue_rows[~idle_flags.fillna(False)]
                    .groupby("tick", as_index=False)["entity_id"]
                    .nunique()
                    .rename(columns={"entity_id": "rescue_assets_active"})
                )
                rescue_metrics = rescue_metrics.merge(active_rescue, on="tick", how="left")
            else:
                # Backward-compatible heuristic for legacy artifacts without explicit `is_idle`.
                active_mask = pd.Series(False, index=rescue_rows.index, dtype="boolean")
                if "mobilisation_ticks_remaining" in rescue_rows.columns:
                    mobilisation = pd.to_numeric(
                        rescue_rows["mobilisation_ticks_remaining"], errors="coerce"
                    ).fillna(0.0)
                    active_mask = active_mask | (mobilisation > 0.0)
                if {"x_nm", "y_nm", "target_x_nm", "target_y_nm"}.issubset(rescue_rows.columns):
                    distance_to_target = (
                        (
                            pd.to_numeric(rescue_rows["x_nm"], errors="coerce")
                            - pd.to_numeric(rescue_rows["target_x_nm"], errors="coerce")
                        )
                        ** 2
                        + (
                            pd.to_numeric(rescue_rows["y_nm"], errors="coerce")
                            - pd.to_numeric(rescue_rows["target_y_nm"], errors="coerce")
                        )
                        ** 2
                    ) ** 0.5
                    # If asset is still meaningfully away from target, keep it active.
                    active_mask = active_mask | (distance_to_target > 0.4)
                active_rescue = (
                    rescue_rows[active_mask.fillna(False)]
                    .groupby("tick", as_index=False)["entity_id"]
                    .nunique()
                    .rename(columns={"entity_id": "rescue_assets_active"})
                )
                rescue_metrics = rescue_metrics.merge(active_rescue, on="tick", how="left")
        for column in rescue_metrics.columns:
            if column != "tick":
                rescue_metrics[column] = pd.to_numeric(
                    rescue_metrics[column], errors="coerce"
                ).fillna(0.0)

        comm_metrics = tick_index.copy()
        comm_rows = run_df[run_df["entity_type"] == "communication_link"].copy()
        if not comm_rows.empty:
            comm_metrics = comm_metrics.merge(
                comm_rows.groupby("tick", as_index=False)
                .size()
                .rename(columns={"size": "communication_links"}),
                on="tick",
                how="left",
            )
        event_rows = run_df[run_df["entity_type"] == "intervention_event"].copy()
        if not event_rows.empty and "event_kind" in event_rows.columns:
            collision_counts = (
                event_rows[event_rows["event_kind"].astype(str) == "collision"]
                .groupby("tick", as_index=False)
                .size()
                .rename(columns={"size": "collision_events"})
            )
            land_collision_counts = (
                event_rows[event_rows["event_kind"].astype(str) == "land_collision"]
                .groupby("tick", as_index=False)
                .size()
                .rename(columns={"size": "land_collision_events"})
            )
            comm_metrics = comm_metrics.merge(collision_counts, on="tick", how="left")
            comm_metrics = comm_metrics.merge(land_collision_counts, on="tick", how="left")
            despawn_counts = (
                event_rows[event_rows["event_kind"].astype(str) == "despawned"]
                .groupby("tick", as_index=False)
                .size()
                .rename(columns={"size": "vessels_despawned"})
            )
            comm_metrics = comm_metrics.merge(despawn_counts, on="tick", how="left")
        for column in comm_metrics.columns:
            if column != "tick":
                comm_metrics[column] = pd.to_numeric(comm_metrics[column], errors="coerce").fillna(
                    0.0
                )
        if "vessels_despawned" in comm_metrics.columns:
            comm_metrics["vessels_despawned_cumsum"] = comm_metrics["vessels_despawned"].cumsum()

        col_left, col_right = st.columns(2)
        with col_left:
            vessel_cols = [
                column
                for column in (
                    "vessels_total",
                    "vessels_active",
                    "vessels_evac",
                    "vessels_sos",
                    "vessels_rescued",
                    "vessels_sunk",
                )
                if column in vessel_metrics.columns
            ]
            if vessel_cols:
                vessel_fig = px.line(
                    vessel_metrics.melt(
                        id_vars=["tick"],
                        value_vars=vessel_cols,
                        var_name="metric",
                        value_name="count",
                    ),
                    x="tick",
                    y="count",
                    color="metric",
                    title="Vessel counts and states over time",
                )
                apply_plotly_theme(vessel_fig, height=320)
                vessel_fig.update_layout(
                    margin={"l": 24, "r": 24, "t": 110, "b": 24},
                    legend={
                        "orientation": "h",
                        "yanchor": "bottom",
                        "y": 1.02,
                        "xanchor": "left",
                        "x": 0.0,
                        "title": {"text": ""},
                    },
                )
                st.plotly_chart(vessel_fig, width="stretch")
        with col_right:
            rescue_cols = [
                column
                for column in ("rescue_assets_total", "rescue_assets_active")
                if column in rescue_metrics.columns
            ]
            if rescue_cols:
                rescue_fig = px.line(
                    rescue_metrics.melt(
                        id_vars=["tick"],
                        value_vars=rescue_cols,
                        var_name="metric",
                        value_name="count",
                    ),
                    x="tick",
                    y="count",
                    color="metric",
                    title="Rescue asset activity over time",
                )
                apply_plotly_theme(rescue_fig, height=320)
                rescue_fig.update_layout(
                    margin={"l": 24, "r": 24, "t": 110, "b": 24},
                    legend={
                        "orientation": "h",
                        "yanchor": "bottom",
                        "y": 1.02,
                        "xanchor": "left",
                        "x": 0.0,
                        "title": {"text": ""},
                    },
                )
                st.plotly_chart(rescue_fig, width="stretch")

        cumulative_outcomes = tick_index.copy()
        cumulative_outcomes["vessels_evac_raw"] = 0.0
        cumulative_outcomes["vessels_rescued_raw"] = 0.0
        cumulative_outcomes["vessels_destination_raw"] = 0.0
        if not vessel_rows.empty and {"vessel_id", "tick", "state"}.issubset(vessel_rows.columns):
            first_evac = (
                vessel_rows[vessel_rows["state"].astype(str).str.lower() == "evac"]
                .sort_values(["vessel_id", "tick"])
                .groupby("vessel_id", as_index=False)
                .head(1)
            )
            if not first_evac.empty:
                evac_by_tick = (
                    first_evac.groupby("tick", as_index=False)["vessel_id"]
                    .nunique()
                    .rename(columns={"vessel_id": "vessels_evac_raw"})
                )
                cumulative_outcomes = cumulative_outcomes.merge(
                    evac_by_tick, on="tick", how="left", suffixes=("", "_evt")
                )
                if "vessels_evac_raw_evt" in cumulative_outcomes.columns:
                    cumulative_outcomes["vessels_evac_raw"] = cumulative_outcomes[
                        "vessels_evac_raw_evt"
                    ].fillna(0.0)
                    cumulative_outcomes = cumulative_outcomes.drop(columns=["vessels_evac_raw_evt"])

            first_rescued = (
                vessel_rows[vessel_rows["state"].astype(str).str.lower() == "rescued"]
                .sort_values(["vessel_id", "tick"])
                .groupby("vessel_id", as_index=False)
                .head(1)
            )
            if not first_rescued.empty:
                rescued_by_tick = (
                    first_rescued.groupby("tick", as_index=False)["vessel_id"]
                    .nunique()
                    .rename(columns={"vessel_id": "vessels_rescued_raw"})
                )
                cumulative_outcomes = cumulative_outcomes.merge(
                    rescued_by_tick, on="tick", how="left", suffixes=("", "_evt")
                )
                if "vessels_rescued_raw_evt" in cumulative_outcomes.columns:
                    cumulative_outcomes["vessels_rescued_raw"] = cumulative_outcomes[
                        "vessels_rescued_raw_evt"
                    ].fillna(0.0)
                    cumulative_outcomes = cumulative_outcomes.drop(
                        columns=["vessels_rescued_raw_evt"]
                    )

        if "vessels_despawned" in comm_metrics.columns:
            destination_by_tick = comm_metrics[["tick", "vessels_despawned"]].rename(
                columns={"vessels_despawned": "vessels_destination_raw"}
            )
            cumulative_outcomes = cumulative_outcomes.merge(
                destination_by_tick, on="tick", how="left", suffixes=("", "_evt")
            )
            if "vessels_destination_raw_evt" in cumulative_outcomes.columns:
                cumulative_outcomes["vessels_destination_raw"] = cumulative_outcomes[
                    "vessels_destination_raw_evt"
                ].fillna(0.0)
                cumulative_outcomes = cumulative_outcomes.drop(
                    columns=["vessels_destination_raw_evt"]
                )
        for column in (
            "vessels_evac_raw",
            "vessels_rescued_raw",
            "vessels_destination_raw",
        ):
            cumulative_outcomes[column] = pd.to_numeric(
                cumulative_outcomes[column], errors="coerce"
            ).fillna(0.0)
        cumulative_outcomes["vessels_evac_cumsum"] = cumulative_outcomes[
            "vessels_evac_raw"
        ].cumsum()
        cumulative_outcomes["vessels_rescued_cumsum"] = cumulative_outcomes[
            "vessels_rescued_raw"
        ].cumsum()
        cumulative_outcomes["vessels_destination_cumsum"] = cumulative_outcomes[
            "vessels_destination_raw"
        ].cumsum()
        outcomes_cumsum_fig = px.line(
            cumulative_outcomes.melt(
                id_vars=["tick"],
                value_vars=[
                    "vessels_evac_cumsum",
                    "vessels_rescued_cumsum",
                    "vessels_destination_cumsum",
                ],
                var_name="metric",
                value_name="count",
            ),
            x="tick",
            y="count",
            color="metric",
            title="Cumulative vessel outcomes: evac, rescued, destination reached",
        )
        apply_plotly_theme(outcomes_cumsum_fig, height=320)
        outcomes_cumsum_fig.update_layout(
            margin={"l": 24, "r": 24, "t": 110, "b": 24},
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "left",
                "x": 0.0,
                "title": {"text": ""},
            },
        )
        st.plotly_chart(outcomes_cumsum_fig, width="stretch")
