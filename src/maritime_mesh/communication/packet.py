"""Packet dataclasses relayed through shore and mesh channels."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MeshPacket:
    """Immutable mesh weather-observation packet."""

    sender_id: int
    position: tuple[float, float]
    observed_hazard: float
    tick_sent: int
    hop_count: int = 0


@dataclass(frozen=True)
class SosPacket:
    """Immutable distress signal packet."""

    sender_id: int
    position: tuple[float, float]
    tick_sent: int
    hop_count: int = 0
