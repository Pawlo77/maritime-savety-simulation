"""Unit tests for Streamlit dashboard helpers."""

from pathlib import Path

import pandas as pd

from maritime_mesh.constants import WORLD_SIZE_NM
from maritime_mesh.dashboard import app as dashboard_app
from maritime_mesh.enums import MethodCondition


def _sample_run_df() -> pd.DataFrame:
    """Create a minimal run log dataframe accepted by timeline map renderer."""
    rows: list[dict] = []
    for tick in (0, 1):
        rows.append(
            {
                "tick": tick,
                "entity_type": "weather_probe",
                "x_nm": 0.0,
                "y_nm": 0.0,
                "hazard": 0.2 + (0.1 * tick),
                "world_size_nm": 140.0,
            }
        )
        rows.append(
            {
                "tick": tick,
                "entity_type": "vessel",
                "vessel_id": 1,
                "state": "active",
                "x_nm": 10.0 + tick,
                "y_nm": 20.0 + tick,
                "hazard": 0.3,
                "w_hat_blend": 0.35,
                "p_prep": 0.8,
                "mesh_observations": 2,
                "shore_rx": True,
                "error_probability": 0.1,
                "world_size_nm": 140.0,
            }
        )
        rows.append(
            {
                "tick": tick,
                "entity_type": "coastal_station",
                "x_nm": 0.0,
                "y_nm": 70.0,
                "shore_broadcast": 0.4,
                "hazard": 0.3,
                "queued_sos": 0,
                "world_size_nm": 140.0,
            }
        )
        rows.append(
            {
                "tick": tick,
                "entity_type": "rescue_asset",
                "entity_id": "r-1",
                "x_nm": 5.0 + tick,
                "y_nm": 6.0 + tick,
                "asset_type": "helicopter",
                "mobilisation_ticks_remaining": 0,
                "world_size_nm": 140.0,
            }
        )
    return pd.DataFrame(rows)


def test_parse_lane_definitions_parses_multiple_lanes() -> None:
    """Lane parser should convert text into immutable lane tuples."""
    lane_text = "north:0,0;0,100\nsouth:10,10;20,20;30,30\n"
    parsed = dashboard_app._parse_lane_definitions(lane_text)
    assert parsed[0][0] == "north"
    assert parsed[0][1][1] == (0.0, 100.0)
    assert parsed[1][0] == "south"
    assert len(parsed[1][1]) == 3


def test_map_world_size_prefers_logged_world_size_column() -> None:
    """World size helper should use logged world_size_nm when available."""
    frame = pd.DataFrame({"world_size_nm": [120.0, 155.0], "x_nm": [5.0, 8.0], "y_nm": [4.0, 6.0]})
    assert dashboard_app._map_world_size(frame) == 155.0


def test_map_world_size_fallback_uses_xy_extent() -> None:
    """World size helper should fallback to x/y extents and default floor."""
    frame = pd.DataFrame({"x_nm": [10.0, 45.0], "y_nm": [20.0, 50.0]})
    assert dashboard_app._map_world_size(frame) == WORLD_SIZE_NM


def test_map_world_size_without_position_columns_returns_default() -> None:
    """World size helper should not crash when x/y columns are absent."""
    frame = pd.DataFrame({"tick": [0, 1], "entity_type": ["vessel", "vessel"]})
    assert dashboard_app._map_world_size(frame) == WORLD_SIZE_NM


def test_can_render_map_requires_minimum_columns() -> None:
    """Map rendering guard should detect missing geometry columns."""
    complete = pd.DataFrame({"tick": [0], "entity_type": ["vessel"], "x_nm": [1.0], "y_nm": [2.0]})
    incomplete = pd.DataFrame({"tick": [0], "entity_type": ["vessel"]})
    assert dashboard_app._can_render_map(complete) is True
    assert dashboard_app._can_render_map(incomplete) is False


def test_make_timeline_map_has_frames_and_dynamic_range() -> None:
    """Timeline map should include one frame per tick and dynamic axis range."""
    figure = dashboard_app._make_timeline_map(_sample_run_df())
    assert len(figure.frames) == 2
    assert figure.layout.xaxis.range[1] == 140.0
    assert figure.layout.yaxis.range[1] == 140.0


def test_run_from_gui_builds_runner_with_overrides(monkeypatch) -> None:
    """GUI run function should pass parsed overrides into ExperimentRunner."""
    captured: dict = {}

    class DummyRunner:
        """Dummy runner used to capture constructor arguments."""

        def __init__(self, **kwargs) -> None:
            """Capture constructor keyword arguments for assertions."""
            captured.update(kwargs)

        def run_all(self) -> pd.DataFrame:
            """Return minimal dataframe expected by calling test."""
            return pd.DataFrame({"scenario": ["scenario_1_calm_passage"], "method": ["proposed"]})

    monkeypatch.setattr(dashboard_app, "ExperimentRunner", DummyRunner)
    result = dashboard_app._run_from_gui(
        output_dir=Path("outputs/maritime_mesh"),
        selected_scenario_names=["scenario_1_calm_passage"],
        selected_methods=[MethodCondition.PROPOSED],
        n_seeds=3,
        n_ticks=200,
        n_vessels=40,
        world_size_nm=180.0,
        green_crew_fraction=0.4,
        shore_noise_std=0.2,
        shore_position=(5.0, 90.0),
        lane_text="lane_a:0,0;100,100",
    )

    assert not result.empty
    assert captured["n_seeds"] == 3
    assert captured["simulation_overrides"]["world_size_nm"] == 180.0
    assert captured["scenario_overrides"]["n_vessels"] == 40
    assert captured["simulation_overrides"]["lane_definitions"][0][0] == "lane_a"
