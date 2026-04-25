"""Streamlit dashboard app entrypoint with multi-page navigation."""

import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from maritime_mesh.dashboard.data_access import can_render_map, load_summary, map_world_size
from maritime_mesh.dashboard.map_view import make_timeline_map
from maritime_mesh.dashboard.pages import home, hypothesis, map_playback, results, run_experiment
from maritime_mesh.dashboard.runner import parse_lane_definitions
from maritime_mesh.dashboard.styles import apply_dashboard_style
from maritime_mesh.experiment import scenarios
from maritime_mesh.experiment.runner import ExperimentRunner
from maritime_mesh.logging_config import configure_logging

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Render dashboard pages with Streamlit native navigation."""
    configure_logging()
    LOGGER.info("Starting dashboard app")
    st.set_page_config(
        page_title="Maritime Mesh Command",
        page_icon="⚓",
        layout="wide",
    )
    apply_dashboard_style()
    with st.sidebar:
        st.markdown("### Workspace")
        output_dir = Path(
            st.text_input(
                "Output directory",
                "outputs/maritime_mesh",
                help=(
                    "Folder containing run artifacts such as summary.csv and per-run parquet logs. "
                    "Relative paths are resolved from the project root. "
                    "Use one experiment family per directory to avoid mixing stale outputs."
                ),
            )
        )
        summary_file = output_dir / "summary.csv"
        if summary_file.exists():
            st.success(f"Output ready: found `{summary_file}`.")
            st.caption(
                "Tip: keep this directory dedicated to one run configuration family "
                "for cleaner comparisons."
            )
        else:
            st.info(
                "No summary file found yet in selected directory. "
                "Run experiments to generate outputs."
            )
    pages = [
        st.Page(
            lambda: home.render(output_dir=output_dir),
            title="Home",
            icon=":material/home:",
            url_path="home",
            default=True,
        ),
        st.Page(
            lambda: run_experiment.render(output_dir=output_dir),
            title="Run",
            icon=":material/play_arrow:",
            url_path="run",
        ),
        st.Page(
            lambda: results.render(output_dir=output_dir),
            title="Results",
            icon=":material/insights:",
            url_path="results",
        ),
        st.Page(
            lambda: map_playback.render(output_dir=output_dir),
            title="Map Playback",
            icon=":material/map:",
            url_path="map-playback",
        ),
        st.Page(
            lambda: hypothesis.render(output_dir=output_dir),
            title="Hypothesis",
            icon=":material/analytics:",
            url_path="hypothesis",
        ),
    ]
    nav = st.navigation(pages, position="top")
    nav.run()


# Backward-compatible aliases used in tests and other imports.
_load_summary = load_summary
_parse_lane_definitions = parse_lane_definitions
_map_world_size = map_world_size
_can_render_map = can_render_map
_make_timeline_map = make_timeline_map


def _run_from_gui(
    output_dir: Path,
    selected_scenario_names: list[str],
    selected_methods: list,
    n_seeds: int,
    n_ticks: int,
    n_vessels: int,
    world_size_nm: float,
    green_crew_fraction: float,
    shore_noise_std: float,
    shore_position: tuple[float, float],
    lane_text: str,
    shore_positions: tuple[tuple[float, float], ...] | None = None,
    max_workers: int = 1,
) -> pd.DataFrame:
    """Backward-compatible run helper accepting raw lane text."""
    scenario_lookup = {
        "scenario_1_calm_passage": scenarios.scenario_1_calm_passage,
        "scenario_2_storm_corridor": scenarios.scenario_2_storm_corridor,
        "scenario_3_blind_shore": scenarios.scenario_3_blind_shore,
        "scenario_4_deep_water_rescue": scenarios.scenario_4_deep_water_rescue,
    }
    scenario_factories = [scenario_lookup[name] for name in selected_scenario_names]
    runner = ExperimentRunner(
        n_seeds=n_seeds,
        max_workers=max_workers,
        output_dir=output_dir,
        scenario_factories=scenario_factories,
        methods=selected_methods,
        simulation_overrides={
            "n_ticks": n_ticks,
            "world_size_nm": world_size_nm,
            "shore_station_position": shore_position,
            "shore_station_positions": shore_positions or (shore_position,),
            "lane_definitions": parse_lane_definitions(lane_text),
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


def _load_run_log(output_dir: Path, scenario: str, method: str, seed: int) -> pd.DataFrame:
    """Backward-compatible run-log loader alias."""
    from maritime_mesh.dashboard.data_access import load_run_log

    return load_run_log(output_dir=output_dir, scenario=scenario, method=method, seed=seed)


def _apply_dashboard_style() -> None:
    """Backward-compatible style helper alias."""
    apply_dashboard_style()


def _render_kpi_cards(filtered: pd.DataFrame) -> None:
    """Backward-compatible KPI cards alias."""
    from maritime_mesh.dashboard.pages.results import _render_kpi_cards as _render_cards

    _render_cards(filtered=filtered)


if __name__ == "__main__":
    main()
