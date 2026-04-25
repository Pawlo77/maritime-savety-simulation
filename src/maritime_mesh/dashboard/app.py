"""Streamlit dashboard for maritime mesh experiment outputs."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from maritime_mesh.constants import WORLD_SIZE_NM
from maritime_mesh.enums import MethodCondition
from maritime_mesh.experiment import scenarios
from maritime_mesh.experiment.analysis import StatisticalAnalyser
from maritime_mesh.experiment.runner import ExperimentRunner

KPI_COLUMNS = [
    "fatal_per_1k_hrs",
    "collision_per_1k_hrs",
    "survival_ratio",
    "avg_tta_hours",
    "evac_activation_rate",
    "mean_p_prep",
]

SCENARIO_CHOICES = [
    "scenario_1_calm_passage",
    "scenario_2_storm_corridor",
    "scenario_3_blind_shore",
    "scenario_4_deep_water_rescue",
]

KPI_DESCRIPTIONS = {
    "fatal_per_1k_hrs": "Fatal events per 1,000 ship-hours (lower is better).",
    "collision_per_1k_hrs": "Collision events per 1,000 ship-hours (lower is better).",
    "survival_ratio": "Survivors divided by total exposed crew (higher is better).",
    "avg_tta_hours": "Average rescue time-to-arrival in hours (lower is better).",
    "evac_activation_rate": "Fraction of vessels that entered evacuation mode.",
    "mean_p_prep": "Average preparedness score across all vessels and ticks.",
}


def _load_summary(output_dir: Path) -> pd.DataFrame:
    """Load summary CSV if available."""
    summary_path = output_dir / "summary.csv"
    if summary_path.exists():
        return pd.read_csv(summary_path)
    return pd.DataFrame()


def _load_run_log(output_dir: Path, scenario: str, method: str, seed: int) -> pd.DataFrame:
    """Load one per-run parquet tick log."""
    run_path = output_dir / f"{scenario}_{method}_{seed}.parquet"
    if not run_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(run_path)


def _parse_lane_definitions(
    raw_text: str,
) -> tuple[tuple[str, tuple[tuple[float, float], ...]], ...]:
    """Parse lane definitions from multiline text."""
    lane_definitions = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lane_name, waypoints_part = line.split(":", maxsplit=1)
        waypoints = []
        for point in waypoints_part.split(";"):
            x_str, y_str = point.strip().split(",", maxsplit=1)
            waypoints.append((float(x_str), float(y_str)))
        lane_definitions.append((lane_name.strip(), tuple(waypoints)))
    return tuple(lane_definitions)


def _map_world_size(run_df: pd.DataFrame) -> float:
    """Infer world size from run logs."""
    if "world_size_nm" in run_df.columns:
        series = run_df["world_size_nm"].dropna()
        if not series.empty:
            return float(series.max())
    xy_max = max(float(run_df["x_nm"].max()), float(run_df["y_nm"].max()))
    return max(WORLD_SIZE_NM, xy_max)


def _make_timeline_map(run_df: pd.DataFrame) -> go.Figure:
    """Build animated map with weather probes, vessels, and rescue assets."""
    world_size_nm = _map_world_size(run_df=run_df)
    ticks = sorted(run_df["tick"].dropna().unique())
    weather_df = run_df[run_df["entity_type"] == "weather_probe"]
    vessel_df = run_df[run_df["entity_type"] == "vessel"]
    station_df = run_df[run_df["entity_type"] == "coastal_station"]
    rescue_df = run_df[run_df["entity_type"] == "rescue_asset"]

    def _frame_for_tick(tick: int) -> go.Frame:
        wx_tick = weather_df[weather_df["tick"] == tick]
        vessel_tick = vessel_df[vessel_df["tick"] == tick]
        station_tick = station_df[station_df["tick"] == tick]
        rescue_tick = rescue_df[rescue_df["tick"] == tick]
        station_hover = (
            "Station<br>"
            + "broadcast=%{customdata[0]:.2f}<br>"
            + "hazard=%{customdata[1]:.2f}<br>"
            + "queued_sos=%{customdata[2]}"
        )
        vessel_hover = (
            "Vessel %{customdata[0]}<br>"
            + "state=%{customdata[1]}<br>"
            + "hazard=%{customdata[2]:.2f}<br>"
            + "blend=%{customdata[3]:.2f}<br>"
            + "prep=%{customdata[4]:.2f}<br>"
            + "mesh_obs=%{customdata[5]}<br>"
            + "shore_rx=%{customdata[6]}<br>"
            + "error_prob=%{customdata[7]:.2f}"
        )
        rescue_hover = (
            "Rescue %{customdata[0]}<br>"
            + "type=%{customdata[1]}<br>"
            + "mobilisation_left=%{customdata[2]}"
        )
        return go.Frame(
            name=str(int(tick)),
            data=[
                go.Heatmap(
                    x=wx_tick["x_nm"],
                    y=wx_tick["y_nm"],
                    z=wx_tick["hazard"],
                    colorscale="Turbo",
                    zmin=0.0,
                    zmax=1.0,
                    opacity=0.42,
                    showscale=False,
                    hovertemplate=(
                        "Weather probe<br>x=%{x:.1f}, y=%{y:.1f}<br>hazard=%{z:.2f}<extra></extra>"
                    ),
                ),
                go.Scatter(
                    x=vessel_tick["x_nm"],
                    y=vessel_tick["y_nm"],
                    mode="markers",
                    marker={
                        "size": 10,
                        "color": vessel_tick["hazard"],
                        "colorscale": "RdYlBu_r",
                        "cmin": 0.0,
                        "cmax": 1.0,
                        "line": {"width": 1, "color": "#222"},
                    },
                    name="Vessels",
                    customdata=vessel_tick[
                        [
                            "vessel_id",
                            "state",
                            "hazard",
                            "w_hat_blend",
                            "p_prep",
                            "mesh_observations",
                            "shore_rx",
                            "error_probability",
                        ]
                    ].to_numpy(),
                    hovertemplate=vessel_hover + "<extra></extra>",
                ),
                go.Scatter(
                    x=station_tick["x_nm"],
                    y=station_tick["y_nm"],
                    mode="markers+text",
                    text=["Shore"],
                    textposition="top center",
                    marker={"size": 14, "symbol": "diamond", "color": "#111"},
                    name="Coastal Station",
                    customdata=station_tick[["shore_broadcast", "hazard", "queued_sos"]]
                    .fillna(0.0)
                    .to_numpy(),
                    hovertemplate=station_hover + "<extra></extra>",
                ),
                go.Scatter(
                    x=rescue_tick["x_nm"],
                    y=rescue_tick["y_nm"],
                    mode="markers",
                    marker={"size": 12, "symbol": "x", "color": "#2ca02c"},
                    name="Rescue Assets",
                    customdata=rescue_tick[
                        ["entity_id", "asset_type", "mobilisation_ticks_remaining"]
                    ]
                    .fillna("")
                    .to_numpy(),
                    hovertemplate=rescue_hover + "<extra></extra>",
                ),
            ],
        )

    frames = [_frame_for_tick(int(tick)) for tick in ticks]
    return go.Figure(
        data=frames[0].data if frames else [],
        frames=frames,
        layout=go.Layout(
            xaxis={"range": [0, world_size_nm], "title": "X (nm)"},
            yaxis={
                "range": [0, world_size_nm],
                "title": "Y (nm)",
                "scaleanchor": "x",
                "scaleratio": 1,
            },
            template="plotly_white",
            height=760,
            margin={"l": 25, "r": 25, "t": 40, "b": 30},
            updatemenus=[
                {
                    "type": "buttons",
                    "x": 0.0,
                    "y": 1.08,
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 300, "redraw": True},
                                    "transition": {"duration": 0},
                                    "fromcurrent": True,
                                    "mode": "immediate",
                                },
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"},
                            ],
                        },
                    ],
                }
            ],
            sliders=[
                {
                    "active": 0,
                    "currentvalue": {"prefix": "Tick: "},
                    "steps": [
                        {
                            "label": str(int(tick)),
                            "method": "animate",
                            "args": [
                                [str(int(tick))],
                                {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
                            ],
                        }
                        for tick in ticks
                    ],
                }
            ],
            shapes=[
                {
                    "type": "rect",
                    "x0": 0.0,
                    "y0": world_size_nm * 0.92,
                    "x1": world_size_nm * 0.18,
                    "y1": world_size_nm,
                    "fillcolor": "#6b8e23",
                    "line": {"color": "#425b15"},
                    "opacity": 0.65,
                    "layer": "below",
                }
            ],
            annotations=[
                {
                    "x": world_size_nm * 0.09,
                    "y": world_size_nm * 0.96,
                    "text": "Land",
                    "showarrow": False,
                    "font": {"size": 11},
                },
            ],
        ),
    )


def _run_from_gui(
    output_dir: Path,
    selected_scenario_names: list[str],
    selected_methods: list[MethodCondition],
    n_seeds: int,
    n_ticks: int,
    n_vessels: int,
    world_size_nm: float,
    green_crew_fraction: float,
    shore_noise_std: float,
    shore_position: tuple[float, float],
    lane_text: str,
) -> pd.DataFrame:
    """Run experiment matrix from GUI controls."""
    scenario_lookup = {
        "scenario_1_calm_passage": scenarios.scenario_1_calm_passage,
        "scenario_2_storm_corridor": scenarios.scenario_2_storm_corridor,
        "scenario_3_blind_shore": scenarios.scenario_3_blind_shore,
        "scenario_4_deep_water_rescue": scenarios.scenario_4_deep_water_rescue,
    }
    scenario_factories = [scenario_lookup[name] for name in selected_scenario_names]
    lane_definitions = _parse_lane_definitions(lane_text)
    runner = ExperimentRunner(
        n_seeds=n_seeds,
        output_dir=output_dir,
        scenario_factories=scenario_factories,
        methods=selected_methods,
        simulation_overrides={
            "n_ticks": n_ticks,
            "world_size_nm": world_size_nm,
            "shore_station_position": shore_position,
            "lane_definitions": lane_definitions,
        },
        scenario_overrides={
            "n_vessels": n_vessels,
            "green_crew_fraction": green_crew_fraction,
            "shore_noise_std": shore_noise_std,
            "min_spawn_distance_nm": 0.0,
            "max_spawn_distance_nm": world_size_nm,
        },
    )
    return runner.run_all()


def _apply_dashboard_style() -> None:
    """Apply minimal style unification for readability."""
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 2.0rem;}
        .mm-card {
            border: 1px solid rgba(120,120,120,0.25);
            border-radius: 10px;
            padding: 0.75rem 0.9rem;
            margin-bottom: 0.6rem;
            background: rgba(250,250,250,0.45);
        }
        .mm-muted {color: #5f6368; font-size: 0.92rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def main() -> None:
    """Render dashboard views and GUI experiment launcher."""
    st.set_page_config(page_title="Maritime Mesh Dashboard", layout="wide")
    _apply_dashboard_style()
    st.title("Maritime Weather Mesh Simulation")
    st.caption(
        "Configure scenarios, run experiments, inspect KPI outcomes, "
        "and replay simulation trajectories."
    )

    output_dir = Path(st.sidebar.text_input("Output directory", "outputs/maritime_mesh"))
    st.sidebar.header("Run Experiment")
    st.sidebar.caption("Define simulation setup and launch runs from the GUI.")
    selected_scenario_names = st.sidebar.multiselect(
        "Scenarios",
        SCENARIO_CHOICES,
        default=["scenario_1_calm_passage"],
        help="Pick one or multiple scenarios to execute.",
    )
    selected_method_values = st.sidebar.multiselect(
        "Methods",
        [condition.value for condition in MethodCondition],
        default=[MethodCondition.PROPOSED.value],
        help="Choose baseline/proposed methods to compare.",
    )
    n_seeds = st.sidebar.number_input(
        "Number of seeds",
        min_value=1,
        max_value=200,
        value=5,
        step=1,
        help="How many random seeds to run per scenario x method.",
    )
    n_ticks = st.sidebar.number_input(
        "Ticks per run",
        min_value=1,
        max_value=2000,
        value=120,
        step=5,
        help="Simulation horizon per run.",
    )
    n_vessels = st.sidebar.number_input(
        "Vessels",
        min_value=1,
        max_value=500,
        value=25,
        step=1,
        help="Number of vessel agents spawned in each run.",
    )
    world_size_nm = st.sidebar.number_input(
        "Map size (nm)",
        min_value=20.0,
        max_value=1000.0,
        value=float(WORLD_SIZE_NM),
        step=10.0,
        help="World side length in nautical miles.",
    )
    green_crew_fraction = st.sidebar.slider(
        "Green crew fraction",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        help="Fraction of inexperienced crews.",
    )
    shore_noise_std = st.sidebar.slider(
        "Shore noise std",
        min_value=0.0,
        max_value=1.0,
        value=0.18,
        help="Noise injected into shore weather broadcast.",
    )
    shore_x = st.sidebar.number_input(
        "Shore station X (nm)", value=0.0, step=1.0, help="X coordinate of shore station."
    )
    shore_y = st.sidebar.number_input(
        "Shore station Y (nm)",
        value=world_size_nm / 2.0,
        step=1.0,
        help="Y coordinate of shore station.",
    )
    lane_text = st.sidebar.text_area(
        "Lane waypoints (one line: name:x1,y1;x2,y2;...)",
        value=(
            f"north_south:20,0;20,{world_size_nm}\n"
            f"east_west:0,{world_size_nm * 0.6};{world_size_nm},{world_size_nm * 0.6}\n"
            f"diagonal:{world_size_nm * 0.1},{world_size_nm * 0.1};"
            f"{world_size_nm * 0.9},{world_size_nm * 0.9}"
        ),
        height=140,
        help=(
            "Each row defines one lane. Example: east_west:0,60;100,60. "
            "At least two waypoints per lane."
        ),
    )
    run_button = st.sidebar.button("Run Experiment Matrix", use_container_width=True)

    results = _load_summary(output_dir)
    if run_button:
        if not selected_scenario_names:
            st.error("Select at least one scenario before running.")
        elif not selected_method_values:
            st.error("Select at least one method before running.")
        else:
            selected_methods = [MethodCondition(value) for value in selected_method_values]
            try:
                with st.spinner("Running simulations from GUI..."):
                    results = _run_from_gui(
                        output_dir=output_dir,
                        selected_scenario_names=selected_scenario_names,
                        selected_methods=selected_methods,
                        n_seeds=int(n_seeds),
                        n_ticks=int(n_ticks),
                        n_vessels=int(n_vessels),
                        world_size_nm=float(world_size_nm),
                        green_crew_fraction=float(green_crew_fraction),
                        shore_noise_std=float(shore_noise_std),
                        shore_position=(float(shore_x), float(shore_y)),
                        lane_text=lane_text,
                    )
                st.success("Experiment run complete. Views refreshed with new results.")
            except ValueError as exc:
                st.error(f"Invalid GUI configuration: {exc}")
                return

    if results.empty:
        st.warning("No summary.csv found. Configure parameters in sidebar and run experiment.")
        return

    scenario = st.sidebar.selectbox("Scenario", sorted(results["scenario"].unique()))
    method_filter = st.sidebar.multiselect(
        "Methods to display",
        sorted(results["method"].unique()),
        default=sorted(results["method"].unique()),
    )
    filtered = results[(results["scenario"] == scenario) & (results["method"].isin(method_filter))]
    if filtered.empty:
        st.warning("No rows match current scenario/method filters.")
        return

    tab_overview, tab_map, tab_kpis, tab_hyp = st.tabs(
        ["Overview", "Simulation Map", "KPI Explorer", "Hypothesis Tests"]
    )

    with tab_overview:
        st.markdown("### Run Overview")
        st.markdown(
            "This table lists all completed runs for the selected scenario and methods. "
            "Use it to inspect seed-level outputs."
        )
        _render_kpi_cards(filtered=filtered)
        st.dataframe(filtered, use_container_width=True)

    with tab_map:
        st.markdown("### Interactive Experiment Map")
        st.markdown(
            "Replay one run tick-by-tick. Colors represent weather hazard intensity; "
            "hover points for vessel/station/rescue details."
        )
        selected_method = st.selectbox(
            "Method for map playback",
            sorted(filtered["method"].unique()),
            key="map_method",
        )
        seed_candidates = sorted(filtered[filtered["method"] == selected_method]["seed"].unique())
        selected_seed = st.selectbox("Seed for map playback", seed_candidates, key="map_seed")
        run_df = _load_run_log(
            output_dir=output_dir,
            scenario=scenario,
            method=selected_method,
            seed=int(selected_seed),
        )
        if run_df.empty:
            st.info("Per-run parquet not found for this selection.")
        else:
            st.plotly_chart(_make_timeline_map(run_df=run_df), use_container_width=True)

    with tab_kpis:
        st.markdown("### KPI Explorer")
        st.markdown(
            "Choose one or multiple KPIs for interactive comparison. "
            "Bars are grouped by method (not stacked)."
        )
        available_kpis = [column for column in KPI_COLUMNS if column in filtered.columns]
        selected_kpis = st.multiselect(
            "KPIs to visualize",
            available_kpis,
            default=(
                ["survival_ratio"] if "survival_ratio" in available_kpis else available_kpis[:1]
            ),
        )
        chart_mode = st.radio(
            "Chart mode",
            ["Single KPI detail", "Compare multiple KPIs"],
            horizontal=True,
        )

        if not selected_kpis:
            st.info("Select at least one KPI to display charts.")
        elif chart_mode == "Single KPI detail":
            selected_kpi = st.selectbox("KPI", selected_kpis, index=0)
            st.caption(KPI_DESCRIPTIONS.get(selected_kpi, ""))
            col_box, col_mean = st.columns(2)
            with col_box:
                st.markdown("**Distribution by Method (interactive boxplot)**")
                fig_box = px.box(
                    filtered,
                    x="method",
                    y=selected_kpi,
                    color="method",
                    points="all",
                    hover_data=["seed", "scenario"],
                    template="plotly_white",
                )
                fig_box.update_layout(showlegend=False, height=420)
                st.plotly_chart(fig_box, use_container_width=True)
            with col_mean:
                st.markdown("**Method Means (separate bars)**")
                means = (
                    filtered.groupby("method", as_index=False)[selected_kpi]
                    .mean()
                    .sort_values(selected_kpi, ascending=False)
                )
                fig_mean = px.bar(
                    means,
                    x="method",
                    y=selected_kpi,
                    color="method",
                    template="plotly_white",
                    text_auto=".3f",
                )
                fig_mean.update_layout(showlegend=False, height=420, barmode="group")
                st.plotly_chart(fig_mean, use_container_width=True)
        else:
            long_df = filtered.melt(
                id_vars=["scenario", "method", "seed"],
                value_vars=selected_kpis,
                var_name="kpi",
                value_name="value",
            )
            st.markdown("**Method means per KPI (faceted, not stacked)**")
            means_long = (
                long_df.groupby(["kpi", "method"], as_index=False)["value"]
                .mean()
                .sort_values(["kpi", "value"], ascending=[True, False])
            )
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

        st.markdown("**Method Means Table**")
        means_table = filtered.groupby("method", as_index=False)[available_kpis].mean()
        st.dataframe(means_table, use_container_width=True)

    with tab_hyp:
        st.markdown("### Hypothesis Tests")
        st.markdown(
            "This table shows non-parametric test outcomes (p-value, effect size, CI) "
            "for configured hypothesis comparisons."
        )
        analyser = StatisticalAnalyser(results_df=results)
        st.dataframe(analyser.full_report(), use_container_width=True)


if __name__ == "__main__":
    main()
