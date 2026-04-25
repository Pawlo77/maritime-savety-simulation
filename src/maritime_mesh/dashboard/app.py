"""Streamlit dashboard for maritime mesh experiment outputs."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from maritime_mesh.constants import WORLD_SIZE_NM
from maritime_mesh.enums import MethodCondition
from maritime_mesh.experiment import scenarios
from maritime_mesh.experiment.analysis import StatisticalAnalyser
from maritime_mesh.experiment.runner import ExperimentRunner


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


def main() -> None:
    """Render dashboard views and GUI experiment launcher."""
    st.set_page_config(page_title="Maritime Mesh Dashboard", layout="wide")
    st.title("Maritime Weather Mesh Simulation")

    output_dir = Path(st.sidebar.text_input("Output directory", "outputs/maritime_mesh"))
    st.sidebar.header("Run Experiment")
    selected_scenario_names = st.sidebar.multiselect(
        "Scenarios",
        [
            "scenario_1_calm_passage",
            "scenario_2_storm_corridor",
            "scenario_3_blind_shore",
            "scenario_4_deep_water_rescue",
        ],
        default=["scenario_1_calm_passage"],
    )
    selected_method_values = st.sidebar.multiselect(
        "Methods",
        [condition.value for condition in MethodCondition],
        default=[MethodCondition.PROPOSED.value],
    )
    n_seeds = st.sidebar.number_input(
        "Number of seeds", min_value=1, max_value=200, value=5, step=1
    )
    n_ticks = st.sidebar.number_input(
        "Ticks per run", min_value=1, max_value=2000, value=120, step=5
    )
    n_vessels = st.sidebar.number_input("Vessels", min_value=1, max_value=500, value=25, step=1)
    world_size_nm = st.sidebar.number_input(
        "Map size (nm)",
        min_value=20.0,
        max_value=1000.0,
        value=float(WORLD_SIZE_NM),
        step=10.0,
    )
    green_crew_fraction = st.sidebar.slider(
        "Green crew fraction", min_value=0.0, max_value=1.0, value=0.3
    )
    shore_noise_std = st.sidebar.slider("Shore noise std", min_value=0.0, max_value=1.0, value=0.18)
    shore_x = st.sidebar.number_input("Shore station X (nm)", value=0.0, step=1.0)
    shore_y = st.sidebar.number_input("Shore station Y (nm)", value=world_size_nm / 2.0, step=1.0)
    lane_text = st.sidebar.text_area(
        "Lane waypoints (one line: name:x1,y1;x2,y2;...)",
        value=(
            f"north_south:20,0;20,{world_size_nm}\n"
            f"east_west:0,{world_size_nm * 0.6};{world_size_nm},{world_size_nm * 0.6}\n"
            f"diagonal:{world_size_nm * 0.1},{world_size_nm * 0.1};"
            f"{world_size_nm * 0.9},{world_size_nm * 0.9}"
        ),
        height=140,
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
    st.subheader("KPI Distribution")
    st.dataframe(filtered, use_container_width=True)

    st.subheader("Interactive Experiment Map")
    selected_method = st.selectbox("Method for map playback", sorted(filtered["method"].unique()))
    seed_candidates = sorted(filtered[filtered["method"] == selected_method]["seed"].unique())
    selected_seed = st.selectbox("Seed for map playback", seed_candidates)
    run_df = _load_run_log(
        output_dir=output_dir, scenario=scenario, method=selected_method, seed=int(selected_seed)
    )
    if run_df.empty:
        st.info("Per-run parquet not found for this selection.")
    else:
        st.plotly_chart(_make_timeline_map(run_df=run_df), use_container_width=True)

    st.subheader("Method Means")
    means = filtered.groupby("method").mean(numeric_only=True).reset_index()
    st.bar_chart(
        means.set_index("method")[["fatal_per_1k_hrs", "collision_per_1k_hrs", "survival_ratio"]]
    )

    st.subheader("Hypothesis Table")
    analyser = StatisticalAnalyser(results_df=results)
    st.dataframe(analyser.full_report(), use_container_width=True)


if __name__ == "__main__":
    main()
