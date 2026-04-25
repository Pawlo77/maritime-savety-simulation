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
        if len(waypoints) < 2:
            raise ValueError("Shipping lanes require at least two waypoints.")
        self.name = name
        self.waypoints = waypoints

    def closest_waypoint_index(self, current_position: tuple[float, float]) -> int:
        """Return index of the nearest waypoint to current position."""
        return min(
            range(len(self.waypoints)),
            key=lambda index: dist(
                current_position,
                (self.waypoints[index].x_nm, self.waypoints[index].y_nm),
            ),
        )

    def waypoint_at(self, index: int) -> Waypoint:
        """Return lane waypoint at index with looped wrapping."""
        return self.waypoints[index % len(self.waypoints)]

    def next_index(self, index: int) -> int:
        """Return next waypoint index with looped wrapping."""
        return (index + 1) % len(self.waypoints)

    def next_waypoint(self, current_position: tuple[float, float]) -> Waypoint:
        """Return nearest waypoint to current vessel position."""
        return self.waypoint_at(self.closest_waypoint_index(current_position))
