"""Vessel collision detection utilities."""

from math import dist

from maritime_mesh.constants import COLLISION_RADIUS_NM
from maritime_mesh.enums import VesselState


class CollisionDetector:
    """Detect pairwise collisions and apply vessel damage."""

    def __init__(self, collision_radius_nm: float = COLLISION_RADIUS_NM) -> None:
        """Initialize collision radius."""
        self.collision_radius_nm = collision_radius_nm

    def check_and_apply(self, vessels: list) -> list[tuple[int, int]]:
        """Detect collisions and update vessel damage counters."""
        collisions: list[tuple[int, int]] = []
        for idx, vessel_a in enumerate(vessels):
            for vessel_b in vessels[idx + 1 :]:
                if (
                    dist(vessel_a.position, vessel_b.position)
                    <= self.collision_radius_nm * 2.0
                    and vessel_a.state == VesselState.ACTIVE
                    and vessel_b.state == VesselState.ACTIVE
                ):
                    vessel_a.damage += 1
                    vessel_b.damage += 1
                    if vessel_a.damage >= vessel_a.stability_threshold:
                        vessel_a.state = VesselState.SUNK
                    if vessel_b.damage >= vessel_b.stability_threshold:
                        vessel_b.state = VesselState.SUNK
                    collisions.append((vessel_a.unique_id, vessel_b.unique_id))
        return collisions
