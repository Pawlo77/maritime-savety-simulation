"""Run-experiment page with map-first lane setup."""

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from maritime_mesh.constants import (
    MAX_HOP_COUNT,
    RADIO_RANGE_FALLOFF,
    RADIO_WEATHER_INTERFERENCE,
    SHORE_BROADCAST_RADIUS_NM,
    VESSEL_RADIO_RANGE_NM,
    WORLD_SIZE_NM,
)
from maritime_mesh.dashboard.constants import SCENARIO_CHOICES
from maritime_mesh.dashboard.runner import run_from_gui
from maritime_mesh.enums import MethodCondition
from maritime_mesh.world.land import WorldLand


def _make_setup_preview_map(
    world_size_nm: float,
    lane_store: dict[str, list[tuple[float, float]]],
    shore_positions: tuple[tuple[float, float], ...],
    min_spawn_distance_nm: float,
    max_spawn_distance_nm: float,
) -> go.Figure:
    """Build setup preview map with land, routes, shore, and spawn bounds."""
    figure = go.Figure()
    land = WorldLand.default_for_world_size(world_size_nm)
    for rectangle in land.rectangles:
        figure.add_shape(
            type="rect",
            x0=rectangle.x0,
            y0=rectangle.y0,
            x1=rectangle.x1,
            y1=rectangle.y1,
            fillcolor="#6b8e23",
            line={"color": "#425b15"},
            opacity=0.6,
        )
    # Spawn distance zone around shore station.
    reference_shore = shore_positions[0]
    if max_spawn_distance_nm > 0:
        figure.add_shape(
            type="circle",
            x0=reference_shore[0] - max_spawn_distance_nm,
            y0=reference_shore[1] - max_spawn_distance_nm,
            x1=reference_shore[0] + max_spawn_distance_nm,
            y1=reference_shore[1] + max_spawn_distance_nm,
            line={"color": "#1f77b4", "dash": "dot"},
            opacity=0.35,
        )
    if min_spawn_distance_nm > 0:
        figure.add_shape(
            type="circle",
            x0=reference_shore[0] - min_spawn_distance_nm,
            y0=reference_shore[1] - min_spawn_distance_nm,
            x1=reference_shore[0] + min_spawn_distance_nm,
            y1=reference_shore[1] + min_spawn_distance_nm,
            line={"color": "#d62728", "dash": "dot"},
            opacity=0.45,
        )
    figure.add_trace(
        go.Scatter(
            x=[position[0] for position in shore_positions],
            y=[position[1] for position in shore_positions],
            mode="markers+text",
            marker={"size": 14, "symbol": "diamond", "color": "#111"},
            text=[f"Shore {index + 1}" for index in range(len(shore_positions))],
            textposition="top center",
            name="Shore stations",
            hovertemplate="Shore station<br>x=%{x:.1f}<br>y=%{y:.1f}<extra></extra>",
        )
    )
    for lane_name, points in lane_store.items():
        if not points:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        figure.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=f"Route: {lane_name}",
                text=[f"{lane_name}#{idx + 1}" for idx in range(len(points))],
                hovertemplate="point=%{text}<br>x=%{x:.1f}<br>y=%{y:.1f}<extra></extra>",
            )
        )
    figure.update_layout(
        template="plotly_white",
        xaxis={"range": [0, world_size_nm], "title": "X (nm)"},
        yaxis={"range": [0, world_size_nm], "title": "Y (nm)", "scaleanchor": "x", "scaleratio": 1},
        height=470,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return figure


def _lane_builder(world_size_nm: float) -> tuple[tuple[str, tuple[tuple[float, float], ...]], ...]:
    """Render map-first lane builder and return lane definitions."""
    land = WorldLand.default_for_world_size(world_size_nm)
    if "lane_store" not in st.session_state:
        st.session_state.lane_store = {
            "coastal_corridor": [
                (world_size_nm * 0.16, world_size_nm * 0.12),
                (world_size_nm * 0.22, world_size_nm * 0.32),
                (world_size_nm * 0.20, world_size_nm * 0.54),
                (world_size_nm * 0.24, world_size_nm * 0.86),
            ],
            "southern_arc": [
                (world_size_nm * 0.16, world_size_nm * 0.14),
                (world_size_nm * 0.36, world_size_nm * 0.18),
                (world_size_nm * 0.58, world_size_nm * 0.28),
                (world_size_nm * 0.86, world_size_nm * 0.36),
            ],
            "northern_bypass": [
                (world_size_nm * 0.16, world_size_nm * 0.70),
                (world_size_nm * 0.38, world_size_nm * 0.72),
                (world_size_nm * 0.58, world_size_nm * 0.70),
                (world_size_nm * 0.78, world_size_nm * 0.84),
                (world_size_nm * 0.94, world_size_nm * 0.84),
            ],
        }
    st.markdown("#### Lane Builder (Map-First)")
    st.caption("Select a lane, add points with coordinates, and preview the route on the map.")
    lane_name = st.selectbox("Lane", [*st.session_state.lane_store.keys(), "new_lane"])
    if lane_name == "new_lane":
        new_name = st.text_input("New lane name", value="diagonal")
        if st.button("Create lane") and new_name:
            st.session_state.lane_store.setdefault(new_name, [])
            lane_name = new_name
    col_x, col_y, col_add = st.columns([1, 1, 1])
    with col_x:
        x_nm = st.number_input("X (nm)", min_value=0.0, max_value=world_size_nm, value=0.0)
    with col_y:
        y_nm = st.number_input("Y (nm)", min_value=0.0, max_value=world_size_nm, value=0.0)
    with col_add:
        st.write("")
        if st.button("Add point", use_container_width=True):
            new_point = (float(x_nm), float(y_nm))
            lane_points = st.session_state.lane_store.setdefault(lane_name, [])
            if land.is_land(new_point):
                st.error("Waypoint cannot be placed on land/shore.")
            elif lane_points and land.segment_intersects_land(lane_points[-1], new_point):
                st.error("Segment intersects land/shore. Choose a water-only waypoint.")
            else:
                lane_points.append(new_point)
    if st.button("Undo last point", use_container_width=True):
        points = st.session_state.lane_store.get(lane_name, [])
        if points:
            points.pop()

    valid_lanes = []
    for name, points in st.session_state.lane_store.items():
        if len(points) < 2:
            continue
        is_valid = True
        for idx in range(len(points) - 1):
            if land.segment_intersects_land(points[idx], points[idx + 1]):
                is_valid = False
                break
        if is_valid:
            valid_lanes.append((name, tuple(points)))
        else:
            st.warning(f"Lane '{name}' is ignored because it intersects land/shore.")
    return tuple(valid_lanes)


def _shore_station_builder(world_size_nm: float) -> tuple[tuple[float, float], ...]:
    """Render controls for creating multiple shore station positions."""
    if "shore_station_store" not in st.session_state:
        st.session_state.shore_station_store = [
            (0.0, world_size_nm * 0.25),
            (0.0, world_size_nm * 0.70),
        ]
    st.markdown("#### Shore Stations")
    st.caption("Add one or more shore stations; rescue assets launch from receiving stations.")
    col_x, col_y, col_add = st.columns([1, 1, 1])
    with col_x:
        shore_x = st.number_input("Shore X (nm)", value=0.0, step=1.0)
    with col_y:
        shore_y = st.number_input("Shore Y (nm)", value=world_size_nm / 2.0, step=1.0)
    with col_add:
        st.write("")
        if st.button("Add shore station", use_container_width=True):
            st.session_state.shore_station_store.append((float(shore_x), float(shore_y)))
    if (
        st.button("Undo last shore station", use_container_width=True)
        and len(st.session_state.shore_station_store) > 1
    ):
        st.session_state.shore_station_store.pop()
    return tuple(st.session_state.shore_station_store)


def render(output_dir: Path) -> None:
    """Render run controls and execute experiment matrix."""
    st.markdown("### Run Experiment")
    st.markdown("Configure simulation settings and launch runs directly from the GUI.")
    (
        tab_basics,
        tab_population,
        tab_shore,
        tab_lanes,
        tab_comms,
        tab_preview,
    ) = st.tabs(["Basics", "Population", "Shore & Spawn", "Lanes", "Comms", "Preview"])

    with tab_basics:
        selected_scenario_names = st.multiselect(
            "Scenarios",
            SCENARIO_CHOICES,
            default=SCENARIO_CHOICES[:1],
        )
        selected_method_values = st.multiselect(
            "Methods",
            [condition.value for condition in MethodCondition],
            default=[MethodCondition.PROPOSED.value],
        )
        n_seeds = st.number_input("Number of seeds", min_value=1, max_value=200, value=5, step=1)
        n_ticks = st.number_input("Ticks per run", min_value=1, max_value=2000, value=120, step=5)
        world_size_nm = st.number_input(
            "Map size (nm)",
            min_value=20.0,
            max_value=1000.0,
            value=float(WORLD_SIZE_NM),
            step=10.0,
        )

    with tab_population:
        n_vessels = st.number_input("Vessels", min_value=1, max_value=500, value=25, step=1)
        green_crew_fraction = st.slider(
            "Green crew fraction", min_value=0.0, max_value=1.0, value=0.3
        )
        shore_noise_std = st.slider("Shore noise std", min_value=0.0, max_value=1.0, value=0.18)

    with tab_shore:
        shore_positions = _shore_station_builder(world_size_nm=world_size_nm)
        min_spawn_distance_nm = st.number_input(
            "Min spawn distance from shore (nm)",
            min_value=0.0,
            max_value=world_size_nm,
            value=0.0,
            step=1.0,
        )
        max_spawn_distance_nm = st.number_input(
            "Max spawn distance from shore (nm)",
            min_value=0.0,
            max_value=world_size_nm,
            value=world_size_nm,
            step=1.0,
        )

    with tab_lanes:
        lane_definitions = _lane_builder(world_size_nm=world_size_nm)

    with tab_comms:
        vessel_radio_range_nm = st.number_input(
            "Vessel radio range (nm)",
            min_value=0.1,
            max_value=500.0,
            value=float(VESSEL_RADIO_RANGE_NM),
            step=0.5,
        )
        max_hop_count = st.number_input(
            "Mesh max hop count",
            min_value=0,
            max_value=10,
            value=int(MAX_HOP_COUNT),
            step=1,
        )
        shore_broadcast_radius_nm = st.number_input(
            "Shore broadcast radius (nm)",
            min_value=0.1,
            max_value=1000.0,
            value=float(SHORE_BROADCAST_RADIUS_NM),
            step=1.0,
        )
        radio_range_falloff = st.number_input(
            "Shore radio range falloff",
            min_value=0.01,
            max_value=200.0,
            value=float(RADIO_RANGE_FALLOFF),
            step=0.1,
        )
        radio_weather_interference = st.number_input(
            "Weather interference factor",
            min_value=0.0,
            max_value=5.0,
            value=float(RADIO_WEATHER_INTERFERENCE),
            step=0.05,
        )
        radio_packet_loss_rate = st.slider(
            "Radio packet loss rate",
            min_value=0.0,
            max_value=0.5,
            value=0.02,
            step=0.01,
        )

    with tab_preview:
        st.markdown("#### Full Setup Preview")
        st.caption(
            "Preview includes generated land area, shore station, configured routes, "
            "and spawn-distance zones used for vessel initialization."
        )
        setup_preview = _make_setup_preview_map(
            world_size_nm=world_size_nm,
            lane_store=st.session_state.lane_store,
            shore_positions=shore_positions,
            min_spawn_distance_nm=float(min_spawn_distance_nm),
            max_spawn_distance_nm=float(max_spawn_distance_nm),
        )
        st.plotly_chart(setup_preview, use_container_width=True)

    st.divider()
    if st.button("Run Experiment Matrix", type="primary", use_container_width=True):
        if not selected_scenario_names:
            st.error("Select at least one scenario.")
            return
        if not selected_method_values:
            st.error("Select at least one method.")
            return
        if not lane_definitions:
            st.error("Add at least one lane with two points.")
            return
        if min_spawn_distance_nm > max_spawn_distance_nm:
            st.error("Min spawn distance must be <= max spawn distance.")
            return
        selected_methods = [MethodCondition(value) for value in selected_method_values]
        try:
            with st.spinner("Running simulations..."):
                run_from_gui(
                    output_dir=output_dir,
                    selected_scenario_names=selected_scenario_names,
                    selected_methods=selected_methods,
                    n_seeds=int(n_seeds),
                    n_ticks=int(n_ticks),
                    n_vessels=int(n_vessels),
                    world_size_nm=float(world_size_nm),
                    green_crew_fraction=float(green_crew_fraction),
                    shore_noise_std=float(shore_noise_std),
                    shore_position=shore_positions[0],
                    shore_positions=shore_positions,
                    lane_definitions=lane_definitions,
                    min_spawn_distance_nm=float(min_spawn_distance_nm),
                    max_spawn_distance_nm=float(max_spawn_distance_nm),
                    vessel_radio_range_nm=float(vessel_radio_range_nm),
                    max_hop_count=int(max_hop_count),
                    shore_broadcast_radius_nm=float(shore_broadcast_radius_nm),
                    radio_range_falloff=float(radio_range_falloff),
                    radio_weather_interference=float(radio_weather_interference),
                    radio_packet_loss_rate=float(radio_packet_loss_rate),
                )
        except ValueError as exc:
            st.error(str(exc))
            return
        st.success("Experiment run complete. Navigate to Results pages.")
