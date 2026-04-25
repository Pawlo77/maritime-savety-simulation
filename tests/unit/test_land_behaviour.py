"""Unit tests for land constraints and grounding behaviour."""

from dataclasses import replace

import pytest

from maritime_mesh.enums import VesselState
from maritime_mesh.model.maritime_model import MaritimeModel


def test_shore_station_must_be_on_land(default_simulation_config) -> None:
    """Model construction should reject offshore shore stations."""
    config = replace(
        default_simulation_config,
        shore_station_position=(90.0, 50.0),
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
