"""Unit tests for land constraints and grounding behaviour."""

from dataclasses import replace

import pytest

from maritime_mesh.enums import VesselState
from maritime_mesh.model.maritime_model import MaritimeModel
from maritime_mesh.world.land import WorldLand


def test_shore_station_must_be_on_land(default_simulation_config) -> None:
    """Model construction should reject offshore shore stations."""
    config = replace(
        default_simulation_config,
        shore_station_position=(99.0, 1.0),
        shore_station_positions=(),
    )
    with pytest.raises(ValueError, match="must be placed on land"):
        MaritimeModel(config)


def test_lane_intersecting_land_is_rejected(default_simulation_config) -> None:
    """Model construction should reject lane segments crossing land."""
    config = replace(
        default_simulation_config,
        lane_definitions=(("invalid_lane", ((5.0, 10.0), (40.0, 10.0))),),
    )
    with pytest.raises(ValueError, match="intersects land"):
        MaritimeModel(config)


def test_grounding_triggers_evac_and_event(default_simulation_config) -> None:
    """Grounding should trigger evacuation-like state transition and event log."""
    model = MaritimeModel(default_simulation_config)
    vessel = model.vessels[0]
    vessel.state = VesselState.ACTIVE
    vessel.position = (5.0, 50.0)

    events = model._land_collisions(previous_positions={vessel.unique_id: (20.0, 50.0)})

    assert events
    assert vessel.state == VesselState.EVAC
    assert vessel.has_evacuated is True


def test_natural_land_profile_exposes_polygons() -> None:
    """Natural profile should provide non-rectangular land geometry."""
    land = WorldLand.default_for_world_size(100.0, profile="natural_coast")
    assert land.polygons
    assert land.rectangles


def test_lane_spawn_produces_safe_initial_leg(default_simulation_config) -> None:
    """Spawned vessels should start on safe water and keep safe first leg."""
    model = MaritimeModel(default_simulation_config)
    for vessel in model.vessels:
        assert model.land.distance_to_land(vessel.position) > model.config.land_clearance_nm
        waypoint = vessel.lane.waypoint_at(vessel._target_waypoint_index)
        assert not model.land.segment_intersects_land(
            vessel.position,
            (waypoint.x_nm, waypoint.y_nm),
            clearance_nm=model.config.land_clearance_nm,
        )


def test_navigation_guardrail_detects_unsafe_segment(default_simulation_config) -> None:
    """Navigation guard should reject a segment crossing buffered shoreline."""
    model = MaritimeModel(default_simulation_config)
    vessel = model.vessels[0]
    assert vessel._move_is_safe((20.0, 50.0), (22.0, 50.0))
    assert not vessel._move_is_safe((20.0, 50.0), (6.0, 50.0))
