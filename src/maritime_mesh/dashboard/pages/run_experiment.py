"""Run-experiment page with map-first lane setup."""

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from maritime_mesh.constants import WORLD_SIZE_NM
from maritime_mesh.dashboard.constants import SCENARIO_CHOICES
from maritime_mesh.dashboard.runner import run_from_gui
from maritime_mesh.enums import MethodCondition


def _make_setup_preview_map(
    world_size_nm: float,
    lane_store: dict[str, list[tuple[float, float]]],
    shore_position: tuple[float, float],
    min_spawn_distance_nm: float,
    max_spawn_distance_nm: float,
) -> go.Figure:
    """Build setup preview map with land, routes, shore, and spawn bounds."""
    figure = go.Figure()
    # Stylized landmass in top-left corner.
    figure.add_shape(
        type="rect",
        x0=0.0,
        y0=world_size_nm * 0.88,
        x1=world_size_nm * 0.22,
        y1=world_size_nm,
        fillcolor="#6b8e23",
        line={"color": "#425b15"},
        opacity=0.65,
    )
    # Spawn distance zone around shore station.
    if max_spawn_distance_nm > 0:
        figure.add_shape(
            type="circle",
            x0=shore_position[0] - max_spawn_distance_nm,
            y0=shore_position[1] - max_spawn_distance_nm,
            x1=shore_position[0] + max_spawn_distance_nm,
            y1=shore_position[1] + max_spawn_distance_nm,
            line={"color": "#1f77b4", "dash": "dot"},
            opacity=0.35,
        )
    if min_spawn_distance_nm > 0:
        figure.add_shape(
            type="circle",
            x0=shore_position[0] - min_spawn_distance_nm,
            y0=shore_position[1] - min_spawn_distance_nm,
            x1=shore_position[0] + min_spawn_distance_nm,
            y1=shore_position[1] + min_spawn_distance_nm,
            line={"color": "#d62728", "dash": "dot"},
            opacity=0.45,
        )
    figure.add_trace(
        go.Scatter(
            x=[shore_position[0]],
            y=[shore_position[1]],
            mode="markers+text",
            marker={"size": 14, "symbol": "diamond", "color": "#111"},
            text=["Shore"],
            textposition="top center",
            name="Shore station",
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
    if "lane_store" not in st.session_state:
        st.session_state.lane_store = {
            "north_south": [(20.0, 0.0), (20.0, world_size_nm)],
            "east_west": [(0.0, world_size_nm * 0.6), (world_size_nm, world_size_nm * 0.6)],
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
            st.session_state.lane_store.setdefault(lane_name, []).append((float(x_nm), float(y_nm)))
    if st.button("Undo last point", use_container_width=True):
        points = st.session_state.lane_store.get(lane_name, [])
        if points:
            points.pop()

    return tuple(
        (name, tuple(points))
        for name, points in st.session_state.lane_store.items()
        if len(points) >= 2
    )


def render(output_dir: Path) -> None:
    """Render run controls and execute experiment matrix."""
    st.markdown("### Run Experiment")
    st.markdown("Configure simulation settings and launch runs directly from the GUI.")
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
    n_vessels = st.number_input("Vessels", min_value=1, max_value=500, value=25, step=1)
    world_size_nm = st.number_input(
        "Map size (nm)",
        min_value=20.0,
        max_value=1000.0,
        value=float(WORLD_SIZE_NM),
        step=10.0,
    )
    green_crew_fraction = st.slider("Green crew fraction", min_value=0.0, max_value=1.0, value=0.3)
    shore_noise_std = st.slider("Shore noise std", min_value=0.0, max_value=1.0, value=0.18)
    shore_x = st.number_input("Shore station X (nm)", value=0.0, step=1.0)
    shore_y = st.number_input("Shore station Y (nm)", value=world_size_nm / 2.0, step=1.0)
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
    lane_definitions = _lane_builder(world_size_nm=world_size_nm)
    st.markdown("#### Full Setup Preview")
    st.caption(
        "Preview includes generated land area, shore station, configured routes, "
        "and spawn-distance zones used for vessel initialization."
    )
    setup_preview = _make_setup_preview_map(
        world_size_nm=world_size_nm,
        lane_store=st.session_state.lane_store,
        shore_position=(float(shore_x), float(shore_y)),
        min_spawn_distance_nm=float(min_spawn_distance_nm),
        max_spawn_distance_nm=float(max_spawn_distance_nm),
    )
    st.plotly_chart(setup_preview, use_container_width=True)

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
                shore_position=(float(shore_x), float(shore_y)),
                lane_definitions=lane_definitions,
                min_spawn_distance_nm=float(min_spawn_distance_nm),
                max_spawn_distance_nm=float(max_spawn_distance_nm),
            )
        st.success("Experiment run complete. Navigate to Results pages.")
