"""Shore broadcast and reception model."""

import math

import numpy as np

from maritime_mesh.constants import (
    RADIO_RANGE_FALLOFF,
    RADIO_WEATHER_INTERFERENCE,
    SHORE_BROADCAST_RADIUS_NM,
    SHORE_NOISE_MEAN,
    SHORE_NOISE_STD_NORMAL,
)


class ShoreRadioModel:
    """Noisy shore forecast broadcast with probabilistic reception."""

    def __init__(
        self,
        rng: np.random.Generator,
        noise_mean: float = SHORE_NOISE_MEAN,
        noise_std: float = SHORE_NOISE_STD_NORMAL,
        broadcast_radius_nm: float = SHORE_BROADCAST_RADIUS_NM,
        range_falloff: float = RADIO_RANGE_FALLOFF,
        weather_interference: float = RADIO_WEATHER_INTERFERENCE,
        packet_loss_rate: float = 0.0,
    ) -> None:
        """Initialize shore radio parameters."""
        self.rng = rng
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.broadcast_radius_nm = broadcast_radius_nm
        self.range_falloff = range_falloff
        self.weather_interference = weather_interference
        self.packet_loss_rate = packet_loss_rate

    def broadcast(self, true_hazard: float) -> float:
        """Generate clipped noisy shore hazard estimate."""
        noisy = true_hazard + float(self.rng.normal(self.noise_mean, self.noise_std))
        return float(min(1.0, max(0.0, noisy)))

    def receive_probability(self, distance_nm: float, local_hazard: float) -> float:
        """Compute reception probability given distance and interference."""
        safe_falloff = max(1e-6, self.range_falloff)
        x_value = ((self.broadcast_radius_nm - distance_nm) / safe_falloff) - (
            self.weather_interference * local_hazard
        )
        return 1.0 / (1.0 + math.exp(-x_value))

    def attempt_receive(self, distance_nm: float, local_hazard: float) -> bool:
        """Draw Bernoulli trial for packet reception."""
        probability = self.receive_probability(distance_nm=distance_nm, local_hazard=local_hazard)
        effective_probability = probability * (1.0 - self.packet_loss_rate)
        return bool(self.rng.random() < effective_probability)
