"""Plotly map rendering for simulation playback."""

import pandas as pd
import plotly.graph_objects as go

from maritime_mesh.dashboard.data_access import map_world_size
from maritime_mesh.dashboard.ui import apply_plotly_theme


def _as_numeric_id(value) -> int | None:
    """Convert IDs to ints when possible for link matching."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def make_timeline_map(
    run_df: pd.DataFrame,
    show_communication_links: bool = False,
    show_event_markers: bool = True,
    collision_vessel_linger_ticks: int = 6,
) -> go.Figure:
    """Build animated map with weather probes, vessels, and rescue assets."""
    world_size_nm = map_world_size(run_df=run_df)
    ticks = sorted(run_df["tick"].dropna().unique())
    weather_df = run_df[run_df["entity_type"] == "weather_probe"]
    vessel_df = run_df[run_df["entity_type"] == "vessel"]
    station_df = run_df[run_df["entity_type"] == "coastal_station"]
    rescue_df = run_df[run_df["entity_type"] == "rescue_asset"]
    event_df = run_df[run_df["entity_type"] == "intervention_event"]
    land_df = run_df[run_df["entity_type"] == "landmass"]
    link_df = run_df[run_df["entity_type"] == "communication_link"]
    if "event_kind" in event_df.columns:
        collision_events = event_df[event_df["event_kind"] == "collision"].copy()
        land_collision_events = event_df[event_df["event_kind"] == "land_collision"].copy()
    else:
        collision_events = event_df.iloc[0:0].copy()
        land_collision_events = event_df.iloc[0:0].copy()

    vessel_collisions_long = pd.DataFrame(columns=["vessel_id", "collision_tick"])
    if not collision_events.empty:
        source_collisions = collision_events[["source_id", "tick"]].rename(
            columns={"source_id": "vessel_id", "tick": "collision_tick"}
        )
        target_collisions = collision_events[["target_id", "tick"]].rename(
            columns={"target_id": "vessel_id", "tick": "collision_tick"}
        )
        vessel_collisions_long = pd.concat(
            [source_collisions, target_collisions],
            ignore_index=True,
        )
        vessel_collisions_long["vessel_id"] = pd.to_numeric(
            vessel_collisions_long["vessel_id"], errors="coerce"
        )
        vessel_collisions_long = vessel_collisions_long.dropna(subset=["vessel_id"])
        vessel_collisions_long["vessel_id"] = vessel_collisions_long["vessel_id"].astype(int)
        vessel_collisions_long = vessel_collisions_long.groupby("vessel_id", as_index=False)[
            "collision_tick"
        ].min()

    if not vessel_collisions_long.empty:
        vessel_df = vessel_df.copy()
        vessel_df["vessel_id"] = pd.to_numeric(vessel_df["vessel_id"], errors="coerce")
        vessel_df = vessel_df.dropna(subset=["vessel_id"])
        vessel_df["vessel_id"] = vessel_df["vessel_id"].astype(int)
        vessel_df = vessel_df.merge(vessel_collisions_long, on="vessel_id", how="left")
        vessel_df = vessel_df[
            vessel_df["collision_tick"].isna()
            | (
                vessel_df["tick"]
                <= (vessel_df["collision_tick"] + max(0, int(collision_vessel_linger_ticks)))
            )
        ]

    collision_markers = pd.DataFrame(columns=["tick", "x_nm", "y_nm", "source_id"])
    if not collision_events.empty:
        collision_positions: list[pd.DataFrame] = []
        for side in ("source_id", "target_id"):
            side_events = collision_events[["tick", side]].rename(columns={side: "vessel_id"})
            side_events["vessel_id"] = pd.to_numeric(side_events["vessel_id"], errors="coerce")
            side_events = side_events.dropna(subset=["vessel_id"])
            side_events["vessel_id"] = side_events["vessel_id"].astype(int)
            vessel_positions = vessel_df[["tick", "vessel_id", "x_nm", "y_nm"]]
            joined = side_events.merge(vessel_positions, on=["tick", "vessel_id"], how="left")
            joined["source_id"] = joined["vessel_id"]
            collision_positions.append(joined[["tick", "x_nm", "y_nm", "source_id"]])
        if collision_positions:
            collision_markers = pd.concat(collision_positions, ignore_index=True)

    if not land_collision_events.empty:
        land_collision_markers = land_collision_events[["tick", "x_nm", "y_nm", "source_id"]].copy()
        collision_markers = pd.concat(
            [collision_markers, land_collision_markers], ignore_index=True
        )

    collision_markers = collision_markers.dropna(subset=["x_nm", "y_nm"]).copy()
    if not collision_markers.empty:
        collision_markers["source_id"] = pd.to_numeric(
            collision_markers["source_id"], errors="coerce"
        ).fillna(-1)
        collision_markers["source_id"] = collision_markers["source_id"].astype(int)

    def _frame_for_tick(tick: int) -> go.Frame:
        """Build one animation frame for a single tick."""
        wx_tick = weather_df[weather_df["tick"] == tick]
        vessel_tick = vessel_df[vessel_df["tick"] == tick]
        station_tick = station_df[station_df["tick"] == tick]
        rescue_tick = rescue_df[rescue_df["tick"] == tick]
        link_tick = link_df[link_df["tick"] == tick]
        node_lookup: dict[int, tuple[float, float]] = {}
        for _, row in pd.concat([vessel_tick, station_tick], ignore_index=True).iterrows():
            for candidate in (row.get("vessel_id"), row.get("entity_id")):
                key = _as_numeric_id(candidate)
                if key is not None:
                    node_lookup[key] = (float(row["x_nm"]), float(row["y_nm"]))
        link_x: list[float | None] = []
        link_y: list[float | None] = []
        if show_communication_links:
            for _, link in link_tick.iterrows():
                source = _as_numeric_id(link.get("source_id"))
                target = _as_numeric_id(link.get("target_id"))
                if source is None or target is None:
                    continue
                if source not in node_lookup or target not in node_lookup:
                    continue
                start = node_lookup[source]
                end = node_lookup[target]
                link_x.extend([start[0], end[0], None])
                link_y.extend([start[1], end[1], None])
        collisions_until_tick = (
            collision_markers[collision_markers["tick"] <= tick]
            if show_event_markers
            else collision_markers.iloc[0:0]
        )
        collision_custom = (
            collisions_until_tick[["source_id", "tick"]].to_numpy()
            if not collisions_until_tick.empty
            else []
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
                    hovertemplate=(
                        "Vessel %{customdata[0]}<br>"
                        "State: %{customdata[1]}<br>"
                        "Local hazard score: %{customdata[2]:.2f}<br>"
                        "Risk blend score: %{customdata[3]:.2f}<br>"
                        "Preparedness score: %{customdata[4]:.2f}<br>"
                        "Mesh observations: %{customdata[5]}<br>"
                        "Shore signal received: %{customdata[6]}<br>"
                        "Transmission error probability: %{customdata[7]:.2f}<extra></extra>"
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
                        "Broadcast strength: %{customdata[0]:.2f}<br>"
                        "Local hazard score: %{customdata[1]:.2f}<br>"
                        "Queued SOS reports: %{customdata[2]}<extra></extra>"
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
                        "Asset type: %{customdata[1]}<br>"
                        "Mobilization ticks remaining: %{customdata[2]}<extra></extra>"
                    ),
                ),
                go.Scatter(
                    x=link_x,
                    y=link_y,
                    mode="lines",
                    line={"width": 1.5, "color": "#9467bd"},
                    name="Communication links",
                    hovertemplate="Communication relay<extra></extra>",
                ),
                go.Scatter(
                    x=collisions_until_tick["x_nm"],
                    y=collisions_until_tick["y_nm"],
                    mode="markers",
                    marker={"size": 12, "symbol": "x", "color": "#8b0000"},
                    name="Collision points",
                    customdata=collision_custom,
                    hovertemplate=(
                        "Collision marker<br>"
                        "vessel=%{customdata[0]}<br>"
                        "first_seen_tick=%{customdata[1]}<extra></extra>"
                    ),
                ),
            ],
        )

    frames = [_frame_for_tick(int(tick)) for tick in ticks]
    land_shapes = []
    if not land_df.empty:
        first_tick = int(ticks[0]) if ticks else 0
        for _, row in land_df[land_df["tick"] == first_tick].iterrows():
            geometry_type = row.get("geometry_type", "rectangle")
            if geometry_type == "polygon" and isinstance(row.get("points_nm"), str):
                points = []
                for pair in row["points_nm"].split(";"):
                    x_str, y_str = pair.split(",", maxsplit=1)
                    points.append((float(x_str), float(y_str)))
                path = "M " + " L ".join(f"{x},{y}" for x, y in points) + " Z"
                land_shapes.append(
                    {
                        "type": "path",
                        "path": path,
                        "xref": "x",
                        "yref": "y",
                        "fillcolor": "#6b8e23",
                        "line": {"color": "#425b15"},
                        "opacity": 0.45,
                        "layer": "below",
                    }
                )
            else:
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
    figure = go.Figure(
        data=frames[0].data if frames else [],
        frames=frames,
        layout=go.Layout(
            xaxis={"range": [0, world_size_nm], "title": "X (nm)", "fixedrange": True},
            yaxis={
                "range": [0, world_size_nm],
                "title": "Y (nm)",
                "scaleanchor": "x",
                "scaleratio": 1,
                "fixedrange": True,
            },
            height=760,
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
    return apply_plotly_theme(figure, height=760)
