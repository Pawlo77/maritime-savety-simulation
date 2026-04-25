"""Shore-trust decay and forecast blending models."""

import math

from maritime_mesh.constants import SHORE_TRUST_DECAY_K


class ShoreTrustDecay:
    """Adaptive trust weighting for stale shore information."""

    def __init__(self, decay_k: float = SHORE_TRUST_DECAY_K) -> None:
        """Initialize decay constant."""
        self.decay_k = decay_k

    def compute(self, shore_age_ticks: int) -> float:
        """Compute trust weight as staleness-dependent lambda."""
        age = max(1, shore_age_ticks)
        return 1.0 - math.exp(-float(age - 1) / self.decay_k)


class ForecastFuser:
    """Blend ship and shore hazard estimates with adaptive lambda."""

    def __init__(self, trust_decay: ShoreTrustDecay | None = None) -> None:
        """Initialize with trust-decay model."""
        self.trust_decay = trust_decay or ShoreTrustDecay()

    def fuse(
        self,
        w_hat_ship: float | None,
        w_hat_shore: float | None,
        shore_age_ticks: int,
    ) -> float:
        """Blend forecasts with source-aware fallback behaviour."""
        if w_hat_ship is None and w_hat_shore is None:
            return 0.0
        if w_hat_ship is None:
            return float(w_hat_shore)
        if w_hat_shore is None:
            return float(w_hat_ship)
        lam = self.trust_decay.compute(shore_age_ticks=shore_age_ticks)
        return float((lam * w_hat_ship) + ((1.0 - lam) * w_hat_shore))
