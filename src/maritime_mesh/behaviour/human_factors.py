"""Human-factor models for vigilance and decision errors."""

import math

from maritime_mesh.constants import CIRCADIAN_NADIR_HOUR_UTC, CIRCADIAN_WEIGHT, ERROR_BIAS, FATIGUE_WEIGHT


def _sigmoid(value: float) -> float:
    """Compute logistic function."""
    return 1.0 / (1.0 + math.exp(-value))


class CircadianModel:
    """Cosine circadian-vigilance model."""

    def vigilance(self, t_utc_hours: float) -> float:
        """Return vigilance factor for current UTC hour."""
        return math.cos((2.0 * math.pi / 24.0) * (t_utc_hours - CIRCADIAN_NADIR_HOUR_UTC))


class ErrorProbabilityModel:
    """Logit model for crew decision-error probability."""

    def __init__(self, circadian_model: CircadianModel | None = None, enabled: bool = True) -> None:
        """Initialize error model with optional ablation switch."""
        self.circadian_model = circadian_model or CircadianModel()
        self.enabled = enabled

    def compute(self, hours_awake: float, t_utc_hours: float, archetype_modifier: float) -> float:
        """Compute crew error probability for a vessel."""
        if not self.enabled:
            return 0.0
        f_circ = self.circadian_model.vigilance(t_utc_hours=t_utc_hours)
        logit = (FATIGUE_WEIGHT * hours_awake) + (CIRCADIAN_WEIGHT * f_circ) + archetype_modifier - ERROR_BIAS
        return _sigmoid(logit)
