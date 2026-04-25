"""Plotly map rendering for simulation playback."""

import pandas as pd
import plotly.graph_objects as go

from maritime_mesh.dashboard.data_access import map_world_size


def make_timeline_map(run_df: pd.DataFrame) -> go.Figure:
    """Build animated map with weather probes, vessels, and rescue assets."""
    world_size_nm = map_world_size(run_df=run_df)
    ticks = sorted(run_df["tick"].dropna().unique())
    weather_df = run_df[run_df["entity_type"] == "weather_probe"]
    vessel_df = run_df[run_df["entity_type"] == "vessel"]
    station_df = run_df[run_df["entity_type"] == "coastal_station"]
    rescue_df = run_df[run_df["entity_type"] == "rescue_asset"]
    event_df = run_df[run_df["entity_type"] == "intervention_event"]
    land_df = run_df[run_df["entity_type"] == "landmass"]

    def _frame_for_tick(tick: int) -> go.Frame:
        """Build one animation frame for a single tick."""
        wx_tick = weather_df[weather_df["tick"] == tick]
        vessel_tick = vessel_df[vessel_df["tick"] == tick]
        station_tick = station_df[station_df["tick"] == tick]
        rescue_tick = rescue_df[rescue_df["tick"] == tick]
        if "event_kind" in event_df.columns:
            land_collision_tick = event_df[
                (event_df["tick"] == tick) & (event_df["event_kind"] == "land_collision")
            ]
        else:
            land_collision_tick = event_df.iloc[0:0]
        if "source_id" in land_collision_tick.columns:
            land_collision_custom = land_collision_tick[["source_id"]].fillna("").to_numpy()
        else:
            land_collision_custom = []
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
                    hovertemplate=(
                        "Vessel %{customdata[0]}<br>"
                        "state=%{customdata[1]}<br>"
                        "hazard=%{customdata[2]:.2f}<br>"
                        "blend=%{customdata[3]:.2f}<br>"
                        "prep=%{customdata[4]:.2f}<br>"
                        "mesh_obs=%{customdata[5]}<br>"
                        "shore_rx=%{customdata[6]}<br>"
                        "error_prob=%{customdata[7]:.2f}<extra></extra>"
                    ),
                ),
                go.Scatter(
                    x=station_tick["x_nm"],
                    y=station_tick["y_nm"],
                    mode="markers+text",
                    text=[f"Shore {idx + 1}" for idx in range(len(station_tick))],
                    textposition="top center",
                    marker={"size": 14, "symbol": "diamond", "color": "#111"},
                    name="Coastal Stations",
                    customdata=station_tick[["shore_broadcast", "hazard", "queued_sos"]]
                    .fillna(0.0)
                    .to_numpy(),
                    hovertemplate=(
                        "Station<br>"
                        "broadcast=%{customdata[0]:.2f}<br>"
                        "hazard=%{customdata[1]:.2f}<br>"
                        "queued_sos=%{customdata[2]}<extra></extra>"
                    ),
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
                    hovertemplate=(
                        "Rescue %{customdata[0]}<br>"
                        "type=%{customdata[1]}<br>"
                        "mobilisation_left=%{customdata[2]}<extra></extra>"
                    ),
                ),
                go.Scatter(
                    x=land_collision_tick["x_nm"],
                    y=land_collision_tick["y_nm"],
                    mode="markers",
                    marker={"size": 13, "symbol": "triangle-up", "color": "#8b0000"},
                    name="Land collisions",
                    customdata=land_collision_custom,
                    hovertemplate="Grounding vessel=%{customdata[0]}<extra></extra>",
                ),
            ],
        )

    frames = [_frame_for_tick(int(tick)) for tick in ticks]
    land_shapes = []
    if not land_df.empty:
        first_tick = int(ticks[0]) if ticks else 0
        for _, row in land_df[land_df["tick"] == first_tick].iterrows():
            land_shapes.append(
                {
                    "type": "rect",
                    "xref": "x",
                    "yref": "y",
                    "x0": float(row["x0_nm"]),
                    "y0": float(row["y0_nm"]),
                    "x1": float(row["x1_nm"]),
                    "y1": float(row["y1_nm"]),
                    "fillcolor": "#6b8e23",
                    "line": {"color": "#425b15"},
                    "opacity": 0.45,
                    "layer": "below",
                }
            )
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
            shapes=land_shapes,
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
        ),
    )
