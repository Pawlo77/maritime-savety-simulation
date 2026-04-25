"""Plotly map rendering for simulation playback."""

import itertools
from math import cos, pi, sin

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
    show_rescue_assets: bool = False,
    show_sos_vessels: bool = False,
    show_configured_overlays: bool = False,
    show_land: bool = True,
    collision_vessel_linger_ticks: int = 6,
    weather_layer: str = "hazard",
    tick_stride: int = 1,
    frame_duration_ms: int = 280,
    transition_duration_ms: int = 60,
    configured_shore_positions: list[tuple[float, float]] | None = None,
    configured_lane_definitions: list[tuple[str, list[tuple[float, float]]]] | None = None,
    vessel_radio_range_nm: float = 0.0,
    shore_broadcast_radius_nm: float = 0.0,
    min_spawn_distance_nm: float = 0.0,
    max_spawn_distance_nm: float = 0.0,
) -> go.Figure:
    """Build animated map with weather probes, vessels, and rescue assets."""
    world_size_nm = map_world_size(run_df=run_df)
    ticks = sorted(run_df["tick"].dropna().unique())
    stride = max(1, int(tick_stride))
    ticks = ticks[::stride]
    weather_df = run_df[run_df["entity_type"] == "weather_probe"]
    vessel_df = run_df[run_df["entity_type"] == "vessel"]
    station_df = run_df[run_df["entity_type"] == "coastal_station"]
    rescue_df = run_df[run_df["entity_type"] == "rescue_asset"]
    event_df = run_df[run_df["entity_type"] == "intervention_event"]
    land_df = run_df[run_df["entity_type"] == "landmass"]
    link_df = run_df[run_df["entity_type"] == "communication_link"]
    weather_by_tick = {int(tick): frame for tick, frame in weather_df.groupby("tick", sort=False)}
    vessel_by_tick = {int(tick): frame for tick, frame in vessel_df.groupby("tick", sort=False)}
    station_by_tick = {int(tick): frame for tick, frame in station_df.groupby("tick", sort=False)}
    rescue_by_tick = {int(tick): frame for tick, frame in rescue_df.groupby("tick", sort=False)}
    link_by_tick = {int(tick): frame for tick, frame in link_df.groupby("tick", sort=False)}
    weather_layer_lookup = {
        "Hazard": ("hazard", "Weather hazard", "YlOrRd"),
        "Sea state": ("sea_state", "Sea state", "Blues"),
        "Wind": ("wind_norm", "Wind intensity", "PuRd"),
        "Low visibility": ("visibility", "Low visibility", "Greys"),
    }
    layer_key, layer_label, layer_scale = weather_layer_lookup.get(
        weather_layer, ("hazard", "Weather hazard", "YlOrRd")
    )
    weather_grid_by_tick: dict[int, pd.DataFrame] = {}
    last_weather_grid = pd.DataFrame(columns=["x_nm", "y_nm", "layer_value"])
    for tick in ticks:
        wx_tick = weather_by_tick.get(int(tick), weather_df.iloc[0:0])
        if not wx_tick.empty:
            if layer_key not in wx_tick.columns:
                layer_values = wx_tick["hazard"]
            elif layer_key == "visibility":
                layer_values = 1.0 - wx_tick["visibility"]
            else:
                layer_values = wx_tick[layer_key]
            wx_grid = wx_tick.assign(layer_value=layer_values).pivot_table(
                index="y_nm",
                columns="x_nm",
                values="layer_value",
                aggfunc="mean",
            )
            last_weather_grid = wx_grid.sort_index().sort_index(axis=1)
        weather_grid_by_tick[int(tick)] = last_weather_grid
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
    wreck_df = run_df[
        (run_df["entity_type"] == "vessel")
        & (run_df["state"].isin(["sunk", "rescued"]) if "state" in run_df.columns else False)
    ].copy()
    if not wreck_df.empty:
        wreck_df = (
            wreck_df.sort_values(["vessel_id", "tick"]).groupby("vessel_id", as_index=False).head(1)
        )
    wreck_until_tick: dict[int, pd.DataFrame] = {}
    if not wreck_df.empty:
        wreck_df = wreck_df.sort_values("tick")
        for tick in ticks:
            wreck_until_tick[int(tick)] = wreck_df[wreck_df["tick"] <= tick]

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
    collision_until_tick: dict[int, pd.DataFrame] = {}
    if not collision_markers.empty:
        collision_markers = collision_markers.sort_values("tick")
        for tick in ticks:
            collision_until_tick[int(tick)] = collision_markers[collision_markers["tick"] <= tick]

    def _frame_for_tick(tick: int) -> go.Frame:
        """Build one animation frame for a single tick."""

        def _col_or_empty(frame: pd.DataFrame, column: str):
            return frame[column] if column in frame.columns else []

        wx_grid = weather_grid_by_tick.get(int(tick), last_weather_grid)
        vessel_tick = vessel_by_tick.get(int(tick), vessel_df.iloc[0:0])
        station_tick = station_by_tick.get(int(tick), station_df.iloc[0:0])
        rescue_tick = rescue_by_tick.get(int(tick), rescue_df.iloc[0:0])
        if "is_idle" in rescue_tick.columns:
            idle_raw = rescue_tick["is_idle"].infer_objects(copy=False)
            if str(idle_raw.dtype) in {"bool", "boolean"}:
                idle_mask = idle_raw.astype("boolean").fillna(False)
            else:
                idle_mask = (
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
                    .fillna(False)
                )
            rescue_tick = rescue_tick[~idle_mask]
        link_tick = link_by_tick.get(int(tick), link_df.iloc[0:0])
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
        wrecks_until_tick = (
            wreck_until_tick.get(int(tick), wreck_df.iloc[0:0])
            if show_event_markers
            else wreck_df.iloc[0:0]
        )
        rescued_wrecks = (
            wrecks_until_tick[wrecks_until_tick["state"] == "rescued"]
            if "state" in wrecks_until_tick.columns
            else wrecks_until_tick.iloc[0:0]
        )
        dead_wrecks = (
            wrecks_until_tick[wrecks_until_tick["state"] == "sunk"]
            if "state" in wrecks_until_tick.columns
            else wrecks_until_tick.iloc[0:0]
        )

        def _wreck_customdata(frame: pd.DataFrame):
            if frame.empty:
                return []
            out = frame.copy()
            for column in ("vessel_id", "tick", "n_survivors", "sos_reason"):
                if column not in out.columns:
                    out[column] = ""
            return out[["vessel_id", "tick", "n_survivors", "sos_reason"]].to_numpy()

        rescue_hover_columns = ["entity_id", "asset_type", "mobilisation_ticks_remaining"]
        rescue_tick_safe = rescue_tick.copy()
        for column in rescue_hover_columns:
            if column not in rescue_tick_safe.columns:
                rescue_tick_safe[column] = ""
        sos_tick = (
            vessel_tick[vessel_tick["sos_sent"]]
            if "sos_sent" in vessel_tick.columns
            else vessel_tick.iloc[0:0]
        )
        configured_shores = configured_shore_positions or []
        configured_x = [position[0] for position in configured_shores]
        configured_y = [position[1] for position in configured_shores]
        return go.Frame(
            name=str(int(tick)),
            data=[
                go.Heatmap(
                    x=list(wx_grid.columns),
                    y=list(wx_grid.index),
                    z=wx_grid.to_numpy(),
                    colorscale=layer_scale,
                    zmin=0.0,
                    zmax=1.0,
                    opacity=0.28,
                    showscale=True,
                    colorbar={"title": layer_label},
                    hovertemplate=(
                        f"{layer_label}<br>x=%{{x:.1f}}, y=%{{y:.1f}}"
                        "<br>value=%{z:.2f}<extra></extra>"
                    ),
                ),
                go.Scatter(
                    x=_col_or_empty(vessel_tick, "x_nm"),
                    y=_col_or_empty(vessel_tick, "y_nm"),
                    mode="markers",
                    marker={
                        "size": 10,
                        "color": _col_or_empty(vessel_tick, "hazard"),
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
                    x=_col_or_empty(station_tick, "x_nm"),
                    y=_col_or_empty(station_tick, "y_nm"),
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
                    x=configured_x if show_configured_overlays else [],
                    y=configured_y if show_configured_overlays else [],
                    mode="markers",
                    marker={"size": 10, "symbol": "diamond-open", "color": "#b8ebff"},
                    name="Configured shore stations",
                    hovertemplate="Configured shore<br>x=%{x:.1f}<br>y=%{y:.1f}<extra></extra>",
                ),
                go.Scatter(
                    x=_col_or_empty(rescue_tick, "x_nm") if show_rescue_assets else [],
                    y=_col_or_empty(rescue_tick, "y_nm") if show_rescue_assets else [],
                    mode="markers",
                    marker={"size": 9, "symbol": "x", "color": "#48d06f"},
                    name="Rescue Assets",
                    customdata=(
                        rescue_tick_safe[rescue_hover_columns].fillna("").to_numpy()
                        if show_rescue_assets
                        else []
                    ),
                    hovertemplate=(
                        "Rescue %{customdata[0]}<br>"
                        "Asset type: %{customdata[1]}<br>"
                        "Mobilization ticks remaining: %{customdata[2]}<extra></extra>"
                    ),
                ),
                go.Scatter(
                    x=sos_tick["x_nm"] if show_sos_vessels else [],
                    y=sos_tick["y_nm"] if show_sos_vessels else [],
                    mode="markers",
                    marker={"size": 12, "symbol": "star", "color": "#ffd166"},
                    name="SOS vessels",
                    customdata=sos_tick[["vessel_id", "n_survivors"]].to_numpy()
                    if (show_sos_vessels and not sos_tick.empty)
                    else [],
                    hovertemplate=(
                        "SOS vessel %{customdata[0]}<br>survivors=%{customdata[1]}<extra></extra>"
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
                    x=rescued_wrecks["x_nm"] if "x_nm" in rescued_wrecks.columns else [],
                    y=rescued_wrecks["y_nm"] if "y_nm" in rescued_wrecks.columns else [],
                    mode="markers",
                    marker={"size": 11, "symbol": "x", "color": "#111111"},
                    name="Wrecks (rescued)",
                    customdata=_wreck_customdata(rescued_wrecks),
                    hovertemplate=(
                        "Rescued wreck<br>"
                        "vessel=%{customdata[0]}<br>"
                        "tick=%{customdata[1]}<br>"
                        "survivors=%{customdata[2]}<br>"
                        "SOS reason=%{customdata[3]}<extra></extra>"
                    ),
                ),
                go.Scatter(
                    x=dead_wrecks["x_nm"] if "x_nm" in dead_wrecks.columns else [],
                    y=dead_wrecks["y_nm"] if "y_nm" in dead_wrecks.columns else [],
                    mode="markers",
                    marker={"size": 11, "symbol": "x", "color": "#8b0000"},
                    name="Wrecks (dead)",
                    customdata=_wreck_customdata(dead_wrecks),
                    hovertemplate=(
                        "Dead wreck<br>"
                        "vessel=%{customdata[0]}<br>"
                        "tick=%{customdata[1]}<br>"
                        "survivors=%{customdata[2]}<br>"
                        "SOS reason=%{customdata[3]}<extra></extra>"
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
    if configured_lane_definitions:
        for _, waypoints in configured_lane_definitions:
            if len(waypoints) < 2:
                continue
            for start, end in itertools.pairwise(waypoints):
                land_shapes.append(
                    {
                        "type": "line",
                        "xref": "x",
                        "yref": "y",
                        "x0": float(start[0]),
                        "y0": float(start[1]),
                        "x1": float(end[0]),
                        "y1": float(end[1]),
                        "line": {
                            "color": (
                                "rgba(176, 229, 255, 0.45)"
                                if show_configured_overlays
                                else "rgba(176, 229, 255, 0.0)"
                            ),
                            "width": 2,
                            "dash": "dot",
                        },
                        "layer": "below",
                    }
                )
    range_overlays = go.Figure()
    if show_configured_overlays and configured_lane_definitions:
        for lane_name, waypoints in configured_lane_definitions:
            if len(waypoints) < 2:
                continue
            range_overlays.add_trace(
                go.Scatter(
                    x=[point[0] for point in waypoints],
                    y=[point[1] for point in waypoints],
                    mode="lines+markers",
                    name=f"Route: {lane_name}",
                    line={"width": 2},
                    marker={"size": 6},
                    hovertemplate="route point<br>x=%{x:.1f}<br>y=%{y:.1f}<extra></extra>",
                )
            )
            endpoints = (waypoints[0], waypoints[-1])
            for endpoint_x, endpoint_y in endpoints:
                for radius_nm, color, label, legend_group in (
                    (max_spawn_distance_nm, "#1f77b4", "Spawn max range", "spawn_max_range"),
                    (min_spawn_distance_nm, "#d62728", "Spawn min range", "spawn_min_range"),
                    (vessel_radio_range_nm, "#9467bd", "Vessel radio range", "vessel_radio_range"),
                ):
                    if radius_nm <= 0:
                        continue
                    angles = [2.0 * pi * idx / 120.0 for idx in range(121)]
                    xs = [endpoint_x + (radius_nm * cos(angle)) for angle in angles]
                    ys = [endpoint_y + (radius_nm * sin(angle)) for angle in angles]
                    range_overlays.add_trace(
                        go.Scatter(
                            x=xs,
                            y=ys,
                            mode="lines",
                            line={"color": color, "width": 1.6, "dash": "dot"},
                            opacity=0.5,
                            name=label,
                            legendgroup=legend_group,
                            showlegend=not any(
                                trace.legendgroup == legend_group for trace in range_overlays.data
                            ),
                            hoverinfo="skip",
                        )
                    )
    if show_configured_overlays and configured_shore_positions and shore_broadcast_radius_nm > 0:
        for shore_x, shore_y in configured_shore_positions:
            angles = [2.0 * pi * idx / 120.0 for idx in range(121)]
            xs = [shore_x + (shore_broadcast_radius_nm * cos(angle)) for angle in angles]
            ys = [shore_y + (shore_broadcast_radius_nm * sin(angle)) for angle in angles]
            range_overlays.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line={"color": "#17becf", "width": 1.8, "dash": "dash"},
                    opacity=0.55,
                    name="Shore broadcast range",
                    legendgroup="shore_broadcast_range",
                    showlegend=not any(
                        trace.legendgroup == "shore_broadcast_range"
                        for trace in range_overlays.data
                    ),
                    hoverinfo="skip",
                )
            )
    play_pause_menu = {
        "type": "buttons",
        "x": 0.0,
        "y": 1.16,
        "xanchor": "left",
        "yanchor": "top",
        "showactive": False,
        "direction": "down",
        "bgcolor": "rgba(18, 58, 88, 0.92)",
        "bordercolor": "rgba(149, 209, 242, 0.55)",
        "borderwidth": 1,
        "font": {"size": 13, "color": "#eaf6ff"},
        "pad": {"t": 2, "r": 8, "l": 2, "b": 2},
        "buttons": [
            {
                "label": "▶ Play",
                "method": "animate",
                "args": [
                    None,
                    {
                        "frame": {"duration": max(0, int(frame_duration_ms)), "redraw": True},
                        "transition": {"duration": max(0, int(transition_duration_ms))},
                        "fromcurrent": True,
                        "mode": "immediate",
                    },
                ],
            },
            {
                "label": "⏸ Pause",
                "method": "animate",
                "args": [
                    [None],
                    {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"},
                ],
            },
        ],
    }
    slider_config = {
        "active": 0,
        "currentvalue": {"prefix": "Tick: "},
        "pad": {"t": 48},
        "steps": [
            {
                "label": str(int(tick)),
                "method": "animate",
                "args": [
                    [str(int(tick))],
                    {
                        "frame": {"duration": 0, "redraw": True},
                        "transition": {"duration": 0},
                        "mode": "immediate",
                    },
                ],
            }
            for tick in ticks
        ],
    }
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
            shapes=land_shapes if show_land else [],
            updatemenus=[play_pause_menu],
            sliders=[slider_config],
        ),
    )
    for overlay_trace in range_overlays.data:
        figure.add_trace(overlay_trace)
    figure = apply_plotly_theme(figure, height=760)
    figure.update_layout(
        title={"text": ""},
        margin={"l": 24, "r": 24, "t": 112, "b": 24},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0.18,
            "bgcolor": "rgba(8, 35, 56, 0.72)",
        },
    )
    return figure
