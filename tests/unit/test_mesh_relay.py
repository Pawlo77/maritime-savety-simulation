import numpy as np

from maritime_mesh.communication.mesh_relay import MeshRelayProtocol
from maritime_mesh.communication.packet import MeshPacket


def test_hop_limit_rejects_packet(seeded_rng: np.random.Generator) -> None:
    relay = MeshRelayProtocol(seeded_rng)
    packet = MeshPacket(
        sender_id=1, position=(0.0, 0.0), observed_hazard=0.4, tick_sent=1, hop_count=2
    )
    assert relay.should_relay(packet) is False


def test_duplicate_suppressed_same_tick(seeded_rng: np.random.Generator) -> None:
    relay = MeshRelayProtocol(seeded_rng)
    packet = MeshPacket(sender_id=1, position=(0.0, 0.0), observed_hazard=0.4, tick_sent=1)
    assert relay.should_relay(packet) is True
    assert relay.should_relay(packet) is False
