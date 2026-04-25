from maritime_mesh.constants import COLLISION_RADIUS_NM
from maritime_mesh.enums import VesselState
from maritime_mesh.world.collision_detector import CollisionDetector


class DummyVessel:
    def __init__(self, unique_id: int, position: tuple[float, float]) -> None:
        self.unique_id = unique_id
        self.position = position
        self.state = VesselState.ACTIVE
        self.damage = 0
        self.stability_threshold = 1


def test_overlapping_vessels_collide() -> None:
    detector = CollisionDetector()
    v1 = DummyVessel(1, (0.0, 0.0))
    v2 = DummyVessel(2, (COLLISION_RADIUS_NM, 0.0))
    collisions = detector.check_and_apply([v1, v2])
    assert collisions == [(1, 2)]


def test_sunk_transition_on_damage_threshold() -> None:
    detector = CollisionDetector()
    v1 = DummyVessel(1, (0.0, 0.0))
    v2 = DummyVessel(2, (COLLISION_RADIUS_NM, 0.0))
    detector.check_and_apply([v1, v2])
    assert v1.state == VesselState.SUNK
