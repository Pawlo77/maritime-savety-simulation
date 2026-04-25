"""Unit tests for lane-following navigation."""

import pytest

from maritime_mesh.agents.vessel_agent import VesselAgent
from maritime_mesh.constants import MACRO_TICK_HOURS
from maritime_mesh.world.lane import ShippingLane, Waypoint


def test_shipping_lane_requires_at_least_two_waypoints() -> None:
    """Lane construction should reject single-point definitions."""
    with pytest.raises(ValueError):
        ShippingLane("invalid", [Waypoint(0.0, 0.0)])


def test_navigate_lane_follows_multi_waypoint_path() -> None:
    """Vessel should continue to subsequent waypoints within one tick."""
    lane = ShippingLane(
        "turning_route",
        [Waypoint(0.0, 0.0), Waypoint(10.0, 0.0), Waypoint(10.0, 10.0)],
    )
    vessel = VesselAgent.__new__(VesselAgent)
    vessel.lane = lane
    vessel.position = (0.0, 0.0)
    vessel.speed_kn = 50.0
    vessel.heading_deg = 0.0
    vessel._target_waypoint_index = 1

    vessel._navigate_lane()

    expected_step = vessel.speed_kn * MACRO_TICK_HOURS
    assert expected_step > 10.0
    assert vessel.position == pytest.approx((10.0, expected_step - 10.0))
    assert vessel._target_waypoint_index == 2


def test_navigate_lane_loops_back_to_first_waypoint() -> None:
    """Vessel should wrap to lane start after reaching last waypoint."""
    lane = ShippingLane(
        "looping_route",
        [Waypoint(0.0, 0.0), Waypoint(10.0, 0.0), Waypoint(10.0, 10.0)],
    )
    vessel = VesselAgent.__new__(VesselAgent)
    vessel.lane = lane
    vessel.position = (10.0, 10.0)
    vessel.speed_kn = 10.0
    vessel.heading_deg = 0.0
    vessel._target_waypoint_index = 0

    vessel._navigate_lane()

    moved_nm = vessel.speed_kn * MACRO_TICK_HOURS
    assert vessel.position == pytest.approx(
        (10.0 - moved_nm / (2**0.5), 10.0 - moved_nm / (2**0.5))
    )
    assert vessel._target_waypoint_index == 0
