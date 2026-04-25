"""Streamlit dashboard for maritime mesh experiment outputs."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from maritime_mesh.constants import WORLD_SIZE_NM
from maritime_mesh.experiment.analysis import StatisticalAnalyser


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


def _make_timeline_map(run_df: pd.DataFrame) -> go.Figure:
    """Build animated map with weather probes, vessels, and rescue assets."""
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
            xaxis={"range": [0, WORLD_SIZE_NM], "title": "X (nm)"},
            yaxis={
                "range": [0, WORLD_SIZE_NM],
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
                    "y0": 92.0,
                    "x1": 18.0,
                    "y1": 100.0,
                    "fillcolor": "#6b8e23",
                    "line": {"color": "#425b15"},
                    "opacity": 0.65,
                    "layer": "below",
                }
            ],
            annotations=[
                {"x": 9.0, "y": 96.0, "text": "Land", "showarrow": False, "font": {"size": 11}},
            ],
        ),
    )


def main() -> None:
    """Render dashboard views for precomputed experiment data."""
    st.set_page_config(page_title="Maritime Mesh Dashboard", layout="wide")
    st.title("Maritime Weather Mesh Simulation")
    output_dir = Path(st.sidebar.text_input("Output directory", "outputs/maritime_mesh"))
    results = _load_summary(output_dir)
    if results.empty:
        st.warning("No summary.csv found. Run experiment first.")
        return

    scenario = st.sidebar.selectbox("Scenario", sorted(results["scenario"].unique()))
    method_filter = st.sidebar.multiselect(
        "Methods",
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
        st.info(
            "Per-run parquet not found for this selection. "
            "Run experiments after this upgrade to generate full map logs."
        )
    else:
        fig = _make_timeline_map(run_df=run_df)
        st.plotly_chart(fig, use_container_width=True)

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
