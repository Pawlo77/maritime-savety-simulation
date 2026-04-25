from math import dist
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from maritime_mesh.constants import (
    MAX_HOP_COUNT,
    RADIO_RANGE_FALLOFF,
    RADIO_WEATHER_INTERFERENCE,
    SHORE_BROADCAST_RADIUS_NM,
    VESSEL_RADIO_RANGE_NM,
    WEATHER_GRID_CELLS,
    WORLD_SIZE_NM,
)
from maritime_mesh.dashboard.constants import (
    SCENARIO_CHOICES,
    display_method_name,
    display_scenario_name,
)
from maritime_mesh.dashboard.runner import run_from_gui
from maritime_mesh.dashboard.ui import apply_plotly_theme, info_panel, page_intro, section_intro
from maritime_mesh.enums import MethodCondition
from maritime_mesh.weather.weather_field import WeatherField
from maritime_mesh.world.land import WorldLand

ROUTE_START_NEAR_SHORE_NM = 45.0
ROUTE_END_NEAR_SHORE_NM = 60.0
ROUTE_END_OFFMAP_MARGIN_NM = 12.0
LaneDefinitions = tuple[tuple[str, tuple[tuple[float, float], ...]], ...]
LaneEndpointWeights = tuple[tuple[str, tuple[float, float]], ...]
LAND_PROFILE_LABELS = {
    "natural_coast": "Natural Coast (recommended)",
    "legacy_rectangles": "Legacy Rectangles",
}


def _default_shore_stations(world_size_nm: float) -> list[tuple[float, float]]:
    """Return default shore stations: mainland + corner island."""
    return [
        (0.0, world_size_nm * 0.50),
        (world_size_nm * 0.78, world_size_nm * 0.96),
    ]


def _default_lane_store(world_size_nm: float) -> dict[str, list[tuple[float, float]]]:
    """Return default lane geometry presets (3 W-E + 2 N-S)."""
    return {
        "west_east_southern": [
            (world_size_nm * 0.18, world_size_nm * 0.16),
            (world_size_nm * 0.34, world_size_nm * 0.28),
            (world_size_nm * 0.52, world_size_nm * 0.34),
            (world_size_nm * 0.70, world_size_nm * 0.32),
            (world_size_nm * 0.90, world_size_nm * 0.40),
        ],
        "west_east_mid_channel": [
            (world_size_nm * 0.20, world_size_nm * 0.24),
            (world_size_nm * 0.36, world_size_nm * 0.30),
            (world_size_nm * 0.52, world_size_nm * 0.34),
            (world_size_nm * 0.66, world_size_nm * 0.36),
            (world_size_nm * 0.92, world_size_nm * 0.38),
        ],
        "west_east_northern_arc": [
            (world_size_nm * 0.18, world_size_nm * 0.70),
            (world_size_nm * 0.36, world_size_nm * 0.74),
            (world_size_nm * 0.58, world_size_nm * 0.72),
            (world_size_nm * 0.80, world_size_nm * 0.82),
            (world_size_nm * 0.94, world_size_nm * 0.86),
        ],
        "north_south_west_channel": [
            (world_size_nm * 0.20, world_size_nm * 0.18),
            (world_size_nm * 0.28, world_size_nm * 0.30),
            (world_size_nm * 0.36, world_size_nm * 0.46),
            (world_size_nm * 0.38, world_size_nm * 0.66),
            (world_size_nm * 0.42, world_size_nm * 0.90),
        ],
        "north_south_central_channel": [
            (world_size_nm * 0.22, world_size_nm * 0.28),
            (world_size_nm * 0.30, world_size_nm * 0.40),
            (world_size_nm * 0.40, world_size_nm * 0.56),
            (world_size_nm * 0.50, world_size_nm * 0.72),
            (world_size_nm * 0.62, world_size_nm * 0.90),
        ],
    }


def _add_bounded_circle(
    figure: go.Figure,
    center_x: float,
    center_y: float,
    radius_nm: float,
    line: dict[str, str | int | float],
    opacity: float,
    name: str,
    legendgroup: str,
    showlegend: bool,
) -> None:
    """Draw a circle as a toggleable trace."""
    if radius_nm <= 0.0:
        return
    angles = np.linspace(0.0, 2.0 * np.pi, num=121)
    xs = center_x + (radius_nm * np.cos(angles))
    ys = center_y + (radius_nm * np.sin(angles))
    figure.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=line,
            opacity=opacity,
            name=name,
            legendgroup=legendgroup,
            showlegend=showlegend,
            hoverinfo="skip",
        )
    )


def _is_near_boundary(point: tuple[float, float], world_size_nm: float, margin_nm: float) -> bool:
    """Return whether point is near map boundary."""
    return (
        point[0] <= margin_nm
        or point[1] <= margin_nm
        or point[0] >= world_size_nm - margin_nm
        or point[1] >= world_size_nm - margin_nm
    )


def _make_setup_preview_map(
    world_size_nm: float,
    lane_store: dict[str, list[tuple[float, float]]],
    shore_positions: tuple[tuple[float, float], ...],
    min_spawn_distance_nm: float,
    max_spawn_distance_nm: float,
    vessel_radio_range_nm: float,
    shore_broadcast_radius_nm: float,
    land_profile: str,
    land_clearance_nm: float,
) -> go.Figure:
    """Build setup preview map with land, routes, shore, and endpoint spawn bounds."""
    figure = go.Figure()
    weather_field = WeatherField(rng=np.random.default_rng(0), world_size_nm=world_size_nm)
    weather_grid = np.array(
        [
            [
                weather_field.hazard_at(
                    ((x_idx + 0.5) * world_size_nm) / WEATHER_GRID_CELLS,
                    ((y_idx + 0.5) * world_size_nm) / WEATHER_GRID_CELLS,
                )
                for x_idx in range(WEATHER_GRID_CELLS)
            ]
            for y_idx in range(WEATHER_GRID_CELLS)
        ]
    )
    axis_points = [
        ((idx + 0.5) * world_size_nm) / WEATHER_GRID_CELLS for idx in range(WEATHER_GRID_CELLS)
    ]
    figure.add_trace(
        go.Heatmap(
            x=axis_points,
            y=axis_points,
            z=weather_grid,
            colorscale=[(0.0, "#ffffff"), (1.0, "#1f5fbf")],
            zmin=0.0,
            zmax=1.0,
            opacity=0.32,
            name="Weather hazard",
            colorbar={"title": "Hazard"},
            hovertemplate=("Weather<br>x=%{x:.1f}, y=%{y:.1f}<br>hazard=%{z:.2f}<extra></extra>"),
        )
    )
    land = WorldLand.default_for_world_size(world_size_nm, profile=land_profile)
    land_patches: list[list[tuple[float, float]]] = []
    for rectangle in land.rectangles:
        land_patches.append(
            [
                (rectangle.x0, rectangle.y0),
                (rectangle.x1, rectangle.y0),
                (rectangle.x1, rectangle.y1),
                (rectangle.x0, rectangle.y1),
            ]
        )
    for polygon in land.polygons:
        land_patches.append(list(polygon.points))
    for idx, patch in enumerate(land_patches):
        figure.add_trace(
            go.Scatter(
                x=[point[0] for point in patch] + [patch[0][0]],
                y=[point[1] for point in patch] + [patch[0][1]],
                mode="lines",
                fill="toself",
                fillcolor="rgba(107,142,35,0.6)",
                line={"color": "#425b15"},
                name="Land",
                legendgroup="land",
                showlegend=idx == 0,
                hoverinfo="skip",
            )
        )
    # Draw spawn range circles around route endpoints.
    spawn_max_legend_shown = False
    spawn_min_legend_shown = False
    for points in lane_store.values():
        if len(points) < 2:
            continue
        endpoints = (points[0], points[-1])
        for endpoint_x, endpoint_y in endpoints:
            if max_spawn_distance_nm > 0:
                _add_bounded_circle(
                    figure=figure,
                    center_x=endpoint_x,
                    center_y=endpoint_y,
                    radius_nm=max_spawn_distance_nm,
                    line={"color": "#1f77b4", "dash": "dot", "width": 2},
                    opacity=0.55,
                    name="Spawn max range",
                    legendgroup="spawn_max_range",
                    showlegend=not spawn_max_legend_shown,
                )
                spawn_max_legend_shown = True
            if min_spawn_distance_nm > 0:
                _add_bounded_circle(
                    figure=figure,
                    center_x=endpoint_x,
                    center_y=endpoint_y,
                    radius_nm=min_spawn_distance_nm,
                    line={"color": "#d62728", "dash": "dot", "width": 2},
                    opacity=0.60,
                    name="Spawn min range",
                    legendgroup="spawn_min_range",
                    showlegend=not spawn_min_legend_shown,
                )
                spawn_min_legend_shown = True

    # Draw shore broadcast circles for every shore station.
    shore_broadcast_legend_shown = False
    for shore_x, shore_y in shore_positions:
        if shore_broadcast_radius_nm > 0:
            _add_bounded_circle(
                figure=figure,
                center_x=shore_x,
                center_y=shore_y,
                radius_nm=shore_broadcast_radius_nm,
                line={"color": "#17becf", "dash": "dash", "width": 2},
                opacity=0.55,
                name="Shore broadcast range",
                legendgroup="shore_broadcast_range",
                showlegend=not shore_broadcast_legend_shown,
            )
            shore_broadcast_legend_shown = True
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
    vessel_radio_legend_shown = False
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
        if vessel_radio_range_nm > 0:
            endpoints = (points[0], points[-1])
            for endpoint_x, endpoint_y in endpoints:
                _add_bounded_circle(
                    figure=figure,
                    center_x=endpoint_x,
                    center_y=endpoint_y,
                    radius_nm=vessel_radio_range_nm,
                    line={"color": "#9467bd", "dash": "dot", "width": 2},
                    opacity=0.45,
                    name="Vessel radio range",
                    legendgroup="vessel_radio_range",
                    showlegend=not vessel_radio_legend_shown,
                )
                vessel_radio_legend_shown = True
    figure.update_layout(
        title={
            "text": f"Setup preview (land clearance {land_clearance_nm:.1f} nm)",
            "x": 0.0,
            "xanchor": "left",
            "y": 0.99,
        },
        xaxis={
            "range": [0, world_size_nm],
            "autorange": False,
            "constrain": "domain",
            "title": "X (nm)",
            "fixedrange": True,
        },
        yaxis={
            "range": [0, world_size_nm],
            "autorange": False,
            "constrain": "domain",
            "title": "Y (nm)",
            "scaleanchor": "x",
            "scaleratio": 1,
            "fixedrange": True,
        },
        height=820,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.06,
            "xanchor": "left",
            "x": 0.0,
            "bgcolor": "rgba(255,255,255,0.8)",
            "groupclick": "togglegroup",
        },
    )
    return apply_plotly_theme(figure, height=820)


def _lane_builder(
    world_size_nm: float,
    shore_positions: tuple[tuple[float, float], ...],
) -> tuple[LaneDefinitions, LaneEndpointWeights]:
    """Render map-first lane builder and return lane definitions."""
    land_profile = st.session_state.get("land_profile", "natural_coast")
    land_clearance_nm = float(st.session_state.get("land_clearance_nm", 0.0))
    land = WorldLand.default_for_world_size(world_size_nm, profile=land_profile)
    if "lane_store" not in st.session_state or st.session_state.get(
        "lane_store_world_size_nm"
    ) != float(world_size_nm):
        st.session_state.lane_store = _default_lane_store(world_size_nm)
        st.session_state.lane_store_world_size_nm = float(world_size_nm)
    section_intro(
        "Lane Builder (Map-first)",
        (
            "Step 1: choose lane, Step 2: add waypoints, Step 3: ensure at "
            "least 2 points per lane and valid start/end placement."
        ),
    )
    lane_name = st.selectbox("Lane", [*st.session_state.lane_store.keys(), "new_lane"])
    if lane_name == "new_lane":
        new_name = st.text_input("New lane name", value="diagonal")
        if st.button("Create lane") and new_name:
            st.session_state.lane_store.setdefault(new_name, [])
            lane_name = new_name
    col_x, col_y, col_add = st.columns([1, 1, 1])
    with col_x:
        x_nm = st.number_input(
            "X (nm)",
            min_value=0.0,
            max_value=world_size_nm,
            value=0.0,
            key="lane_builder_x_nm",
        )
    with col_y:
        y_nm = st.number_input(
            "Y (nm)",
            min_value=0.0,
            max_value=world_size_nm,
            value=0.0,
            key="lane_builder_y_nm",
        )
    with col_add:
        st.write("")
        if st.button("Add point", use_container_width=True):
            new_point = (float(x_nm), float(y_nm))
            lane_points = st.session_state.lane_store.setdefault(lane_name, [])
            if land.distance_to_land(new_point) <= land_clearance_nm:
                st.error(
                    "Waypoint is inside the shoreline clearance zone. "
                    "Move it farther into open water."
                )
            elif lane_points and land.segment_intersects_land(
                lane_points[-1],
                new_point,
                clearance_nm=land_clearance_nm,
            ):
                st.error(
                    "Segment crosses shoreline clearance. "
                    "Add an intermediate offshore waypoint and retry."
                )
            else:
                lane_points.append(new_point)

    col_undo, col_remove_lane, col_clear_lanes, col_reset_defaults = st.columns(4)
    with col_undo:
        if st.button("Undo last point", use_container_width=True):
            points = st.session_state.lane_store.get(lane_name, [])
            if points:
                points.pop()
                st.rerun()
    with col_remove_lane:
        if st.button("Remove selected lane", use_container_width=True) and lane_name != "new_lane":
            st.session_state.lane_store.pop(lane_name, None)
            st.rerun()
    with col_clear_lanes:
        if st.button("Clear all lanes", use_container_width=True):
            st.session_state.lane_store = {}
            st.rerun()
    with col_reset_defaults:
        if st.button("Reset default lanes", use_container_width=True):
            st.session_state.lane_store = _default_lane_store(world_size_nm)
            st.rerun()

    valid_lanes = []
    lane_status_lines: list[str] = []
    lane_issues: list[str] = []
    for name, points in st.session_state.lane_store.items():
        lane_status = "valid"
        if len(points) < 2:
            lane_status_lines.append(
                "<span class='mm-badge mm-badge-warning'>"
                f"{name}: needs 2+ points (currently {len(points)})"
                "</span>"
            )
            continue
        is_valid = True
        for idx in range(len(points) - 1):
            if land.segment_intersects_land(
                points[idx],
                points[idx + 1],
                clearance_nm=land_clearance_nm,
            ):
                is_valid = False
                lane_status = "intersects_land"
                break
        if is_valid:
            start_point = points[0]
            end_point = points[-1]
            if not any(
                dist(start_point, shore) <= ROUTE_START_NEAR_SHORE_NM for shore in shore_positions
            ):
                is_valid = False
                lane_status = "start_far_from_shore"
                lane_issues.append(
                    f"`{name}`: start point is not near shore "
                    f"(<= {ROUTE_START_NEAR_SHORE_NM:.1f} nm)."
                )
            elif not (
                any(dist(end_point, shore) <= ROUTE_END_NEAR_SHORE_NM for shore in shore_positions)
                or _is_near_boundary(end_point, world_size_nm, ROUTE_END_OFFMAP_MARGIN_NM)
            ):
                is_valid = False
                lane_status = "end_invalid"
                lane_issues.append(
                    f"`{name}`: end point must be near shore or "
                    f"<= {ROUTE_END_OFFMAP_MARGIN_NM:.1f} nm from map edge."
                )
        if is_valid:
            valid_lanes.append((name, tuple(points)))
            lane_status_lines.append(
                "<span class='mm-badge mm-badge-success'>"
                f"{name}: valid ({len(points)} points)"
                "</span>"
            )
        else:
            if len(points) >= 2:
                if lane_status == "intersects_land":
                    lane_issues.append(f"`{name}`: intersects land/shore clearance.")
                if lane_status == "intersects_land":
                    lane_status_lines.append(
                        "<span class='mm-badge mm-badge-error'>"
                        f"{name}: intersects shoreline clearance"
                        "</span>"
                    )
                elif lane_status == "start_far_from_shore":
                    lane_status_lines.append(
                        "<span class='mm-badge mm-badge-warning'>"
                        f"{name}: move start closer to shore station"
                        "</span>"
                    )
                elif lane_status == "end_invalid":
                    lane_status_lines.append(
                        "<span class='mm-badge mm-badge-warning'>"
                        f"{name}: move end near shore or map boundary"
                        "</span>"
                    )
    if lane_issues:
        st.info("Invalid lanes are excluded from simulation:\n\n- " + "\n- ".join(lane_issues))
    if lane_status_lines:
        st.markdown("".join(lane_status_lines), unsafe_allow_html=True)

    section_intro(
        "Spawn Share Per Route Endpoint",
        (
            "Set relative spawn weights for each route endpoint "
            "(start/end). Values are normalized automatically."
        ),
    )
    endpoint_weights: list[tuple[str, tuple[float, float]]] = []
    for lane_name_valid, _ in valid_lanes:
        key_start = f"spawn_weight::{lane_name_valid}::start"
        key_end = f"spawn_weight::{lane_name_valid}::end"
        if key_start not in st.session_state:
            st.session_state[key_start] = 1.0
        if key_end not in st.session_state:
            st.session_state[key_end] = 1.0
        col_lane, col_start, col_end = st.columns([2, 1, 1])
        with col_lane:
            st.markdown(f"`{lane_name_valid}`")
        with col_start:
            start_weight = st.number_input(
                "start %",
                min_value=0.0,
                value=float(st.session_state[key_start]),
                key=f"input_{key_start}",
            )
        with col_end:
            end_weight = st.number_input(
                "end %",
                min_value=0.0,
                value=float(st.session_state[key_end]),
                key=f"input_{key_end}",
            )
        st.session_state[key_start] = float(start_weight)
        st.session_state[key_end] = float(end_weight)
        endpoint_weights.append((lane_name_valid, (float(start_weight), float(end_weight))))
    return tuple(valid_lanes), tuple(endpoint_weights)


def _shore_station_builder(world_size_nm: float) -> tuple[tuple[float, float], ...]:
    """Render controls for creating multiple shore station positions."""
    if "shore_station_store" not in st.session_state:
        st.session_state.shore_station_store = _default_shore_stations(world_size_nm)
    section_intro(
        "Shore Stations",
        "Add one or more shore stations; rescue assets launch from receiving stations.",
    )
    col_x, col_y, col_add = st.columns([1, 1, 1])
    with col_x:
        shore_x = st.number_input(
            "Shore X (nm)",
            value=0.0,
            step=1.0,
            key="shore_builder_x_nm",
        )
    with col_y:
        shore_y = st.number_input(
            "Shore Y (nm)",
            value=world_size_nm / 2.0,
            step=1.0,
            key="shore_builder_y_nm",
        )
    with col_add:
        st.write("")
        if st.button("Add shore station", use_container_width=True):
            st.session_state.shore_station_store.append((float(shore_x), float(shore_y)))

    if st.session_state.shore_station_store:
        st.caption("Configured shore stations")
        for idx, position in enumerate(st.session_state.shore_station_store):
            st.markdown(
                "<span class='mm-badge mm-badge-success'>"
                f"Shore {idx + 1}: ({position[0]:.1f}, {position[1]:.1f})"
                "</span>",
                unsafe_allow_html=True,
            )

    col_remove_pick, col_remove_action, col_clear = st.columns([2, 1, 1])
    with col_remove_pick:
        selected = None
        if st.session_state.shore_station_store:
            shore_labels = [
                f"Shore {idx + 1}: ({position[0]:.1f}, {position[1]:.1f})"
                for idx, position in enumerate(st.session_state.shore_station_store)
            ]
            selected = st.selectbox(
                "Select shore station to remove",
                shore_labels,
                key="shore_remove_select",
            )
    with col_remove_action:
        st.write("")
        if (
            selected is not None
            and st.button("Remove selected", use_container_width=True)
            and st.session_state.shore_station_store
        ):
            selected_idx = shore_labels.index(selected)
            st.session_state.shore_station_store.pop(selected_idx)
            st.rerun()
    with col_clear:
        st.write("")
        if st.button("Reset default shores", use_container_width=True):
            st.session_state.shore_station_store = _default_shore_stations(world_size_nm)
            st.rerun()

    if not st.session_state.shore_station_store:
        st.warning("No shore stations configured. Add at least one station before running.")
    return tuple(st.session_state.shore_station_store)


def render(output_dir: Path) -> None:
    """Render run controls and execute experiment matrix."""
    page_intro(
        "Run Experiment",
        (
            "Configure scenarios, communication settings, and route geometry, "
            "then launch a full experiment matrix."
        ),
    )
    info_panel(
        "How To Use",
        (
            "Start from Basics, then Population and Shore settings. "
            "Build routes in Lanes, check Preview, and run the matrix."
        ),
    )
    (
        tab_basics,
        tab_population,
        tab_shore,
        tab_lanes,
        tab_comms,
        tab_preview,
    ) = st.tabs(["Basics", "Population", "Shore & Spawn", "Lanes", "Comms", "Preview"])

    with tab_basics:
        section_intro(
            "Core Simulation Setup",
            "Choose scenarios and methods, then tune run depth and map geometry.",
        )
        selected_scenario_names = st.multiselect(
            "Scenarios",
            SCENARIO_CHOICES,
            default=SCENARIO_CHOICES[:1],
            format_func=display_scenario_name,
            help=("Select one or more weather/navigation contexts to include in this run."),
        )
        selected_method_values = st.multiselect(
            "Methods",
            [condition.value for condition in MethodCondition],
            default=[MethodCondition.PROPOSED.value],
            format_func=display_method_name,
            help="Choose decision strategies to compare under the same conditions.",
        )
        n_seeds = st.number_input(
            "Number of seeds",
            min_value=1,
            max_value=200,
            value=5,
            step=1,
            help=(
                "Independent random trials per scenario-method pair. More seeds improve stability."
            ),
        )
        n_ticks = st.number_input(
            "Ticks per run",
            min_value=1,
            max_value=2000,
            value=120,
            step=5,
            help=(
                "Simulation horizon length. Larger values model longer voyages "
                "but increase runtime."
            ),
        )
        world_size_nm = st.number_input(
            "Map size (nm)",
            min_value=20.0,
            max_value=1000.0,
            value=float(WORLD_SIZE_NM),
            step=10.0,
            help="Square world side length in nautical miles.",
        )
        land_profile = st.selectbox(
            "Land profile",
            options=["natural_coast", "legacy_rectangles"],
            index=0,
            format_func=lambda option: LAND_PROFILE_LABELS.get(option, option),
            help=(
                "Natural coast adds realistic shoreline geometry; legacy is a "
                "simpler rectangular approximation."
            ),
        )
        land_clearance_nm = st.number_input(
            "Land clearance (nm)",
            min_value=0.0,
            max_value=20.0,
            value=1.0,
            step=0.1,
            help="Safety buffer around land. Routes and spawn points must stay outside this band.",
        )
        st.session_state.land_profile = land_profile
        st.session_state.land_clearance_nm = float(land_clearance_nm)

    with tab_population:
        section_intro(
            "Population Risk Profile",
            "Define fleet size and crew risk characteristics.",
        )
        n_vessels = st.number_input(
            "Vessels",
            min_value=1,
            max_value=500,
            value=25,
            step=1,
            help="Number of vessels simulated in each run.",
        )
        green_crew_fraction = st.slider(
            "Green crew fraction",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            help=(
                "Fraction of less experienced crew. Higher values typically "
                "increase operational risk."
            ),
        )
        shore_noise_std = st.slider(
            "Shore signal noise (std)",
            min_value=0.0,
            max_value=1.0,
            value=0.18,
            help=(
                "Uncertainty in shore-side hazard observations. "
                "Higher values mean noisier shore estimates."
            ),
        )

    with tab_shore:
        section_intro(
            "Shore Stations And Spawn Radius",
            (
                "Place stations on land, then set how far vessel spawn points "
                "can appear from each route endpoint."
            ),
        )
        shore_positions = _shore_station_builder(world_size_nm=world_size_nm)
        st.caption(
            "Spawn annulus is centered on the selected route endpoint; "
            "sampling is biased toward the outer ring (near max radius)."
        )
        min_spawn_distance_nm, max_spawn_distance_nm = st.slider(
            "Spawn radius range from selected route endpoint (nm)",
            min_value=0.0,
            max_value=float(world_size_nm),
            value=(0.0, min(5.0, float(world_size_nm))),
            step=1.0,
            help=(
                "Choose minimum and maximum spawn radius in one control. "
                "Values are applied as an annulus around each route endpoint."
            ),
        )

    with tab_lanes:
        lane_definitions, lane_endpoint_spawn_weights = _lane_builder(
            world_size_nm=world_size_nm,
            shore_positions=shore_positions,
        )

    with tab_comms:
        section_intro(
            "Communication Model",
            "Start with core radio settings; expand advanced tuning only when needed.",
        )
        col_radio_a, col_radio_b = st.columns(2)
        with col_radio_a:
            vessel_radio_range_nm = st.number_input(
                "Vessel radio range (nm)",
                min_value=0.1,
                max_value=500.0,
                value=float(VESSEL_RADIO_RANGE_NM),
                step=0.5,
                help=("Maximum direct vessel-to-vessel communication distance in calm conditions."),
            )
            max_hop_count = st.number_input(
                "Mesh max hop count",
                min_value=0,
                max_value=10,
                value=int(MAX_HOP_COUNT),
                step=1,
                help=(
                    "Maximum relay hops allowed per message. Higher values "
                    "increase reach but may raise latency."
                ),
            )
        with col_radio_b:
            shore_broadcast_radius_nm = st.number_input(
                "Shore broadcast radius (nm)",
                min_value=0.1,
                max_value=1000.0,
                value=float(SHORE_BROADCAST_RADIUS_NM),
                step=1.0,
                help="Maximum shore-to-vessel broadcast coverage.",
            )
            radio_packet_loss_rate = st.slider(
                "Radio packet loss rate",
                min_value=0.0,
                max_value=0.5,
                value=0.02,
                step=0.01,
                help=(
                    "Fraction of packets dropped before delivery. "
                    "Higher values degrade communications."
                ),
            )
        with st.expander("Advanced propagation tuning", expanded=False):
            radio_range_falloff = st.number_input(
                "Shore radio range falloff",
                min_value=0.01,
                max_value=200.0,
                value=float(RADIO_RANGE_FALLOFF),
                step=0.1,
                help=(
                    "How quickly shore signal quality decays with distance. "
                    "Higher values mean faster degradation."
                ),
            )
            radio_weather_interference = st.number_input(
                "Weather interference factor",
                min_value=0.0,
                max_value=5.0,
                value=float(RADIO_WEATHER_INTERFERENCE),
                step=0.05,
                help=(
                    "How strongly adverse weather increases transmission failures. "
                    "0 = no weather impact, larger values = stronger disruption."
                ),
            )

    with tab_preview:
        section_intro(
            "Full Setup Preview",
            (
                "Validate shoreline, station placement, route geometry, and "
                "spawn rings before launching runs."
            ),
        )
        preview_lane_store = {name: list(points) for name, points in lane_definitions}
        setup_preview = _make_setup_preview_map(
            world_size_nm=world_size_nm,
            lane_store=preview_lane_store,
            shore_positions=shore_positions,
            min_spawn_distance_nm=float(min_spawn_distance_nm),
            max_spawn_distance_nm=float(max_spawn_distance_nm),
            vessel_radio_range_nm=float(vessel_radio_range_nm),
            shore_broadcast_radius_nm=float(shore_broadcast_radius_nm),
            land_profile=land_profile,
            land_clearance_nm=float(land_clearance_nm),
        )
        if land_clearance_nm > 0.0:
            st.caption(
                f"Active shoreline safety clearance: {float(land_clearance_nm):.1f} nm "
                "(routes and spawn must remain outside this band)."
            )
        st.plotly_chart(setup_preview, use_container_width=True)

    st.divider()
    if st.button("Run Experiment Matrix", type="primary", use_container_width=True):
        if not selected_scenario_names:
            st.error("Select at least one scenario in Basics to define the experiment context.")
            return
        if not selected_method_values:
            st.error("Select at least one method in Basics so runs can be compared.")
            return
        if not shore_positions:
            st.error("Add at least one shore station in Shore & Spawn and place it on land.")
            return
        if not lane_definitions:
            st.error(
                "Add at least one valid lane with 2+ points in Lanes; "
                "ensure it does not cross shoreline clearance."
            )
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
                    land_profile=land_profile,
                    land_clearance_nm=float(land_clearance_nm),
                    route_start_near_shore_nm=ROUTE_START_NEAR_SHORE_NM,
                    route_end_near_shore_nm=ROUTE_END_NEAR_SHORE_NM,
                    route_end_offmap_margin_nm=ROUTE_END_OFFMAP_MARGIN_NM,
                    lane_endpoint_spawn_weights=lane_endpoint_spawn_weights,
                )
        except ValueError as exc:
            st.error(f"{exc} Please adjust lane geometry or shore placement and run again.")
            return
        st.success("Experiment run complete. Navigate to Results pages.")
