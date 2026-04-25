"""Shipping lanes and waypoint navigation helpers."""

from dataclasses import dataclass
from math import dist


@dataclass(frozen=True)
class Waypoint:
    """Single lane waypoint in world coordinates."""

    x_nm: float
    y_nm: float


class ShippingLane:
    """Ordered waypoint sequence representing a shipping route."""

    def __init__(self, name: str, waypoints: list[Waypoint]) -> None:
        """Initialize lane metadata and waypoints."""
        self.name = name
        self.waypoints = waypoints

    def next_waypoint(self, current_position: tuple[float, float]) -> Waypoint:
        """Return nearest waypoint to current vessel position."""
        return min(
            self.waypoints,
            key=lambda waypoint: dist(current_position, (waypoint.x_nm, waypoint.y_nm)),
        )
