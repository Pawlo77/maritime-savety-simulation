"""Mesh relay protocol for weather and SOS packets."""

from dataclasses import replace

import numpy as np

from maritime_mesh.communication.packet import MeshPacket, SosPacket
from maritime_mesh.constants import MAX_HOP_COUNT


class MeshRelayProtocol:
    """Hop-limited duplicate-suppressing relay of mesh packets."""

    def __init__(self, rng: np.random.Generator) -> None:
        """Initialize protocol with RNG and per-tick seen set."""
        self.rng = rng
        self._seen: set[tuple[int, int, str]] = set()

    def reset_tick(self) -> None:
        """Clear duplicate-tracking state for current tick."""
        self._seen.clear()

    def should_relay(self, packet: MeshPacket | SosPacket) -> bool:
        """Return whether packet may be relayed in current tick."""
        signature = (packet.sender_id, packet.tick_sent, packet.__class__.__name__)
        if packet.hop_count >= MAX_HOP_COUNT or signature in self._seen:
            return False
        self._seen.add(signature)
        return True

    def relay(self, packet: MeshPacket | SosPacket) -> MeshPacket | SosPacket:
        """Return packet copy with incremented hop count."""
        return replace(packet, hop_count=packet.hop_count + 1)
