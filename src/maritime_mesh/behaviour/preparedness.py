"""Preparedness scoring model."""

from maritime_mesh.constants import PREP_ARCHETYPE_PENALTY, PREP_FORECAST_ERROR_PENALTY


class PreparednessScorer:
    """Forecast-and-archetype based preparedness score model."""

    def __init__(
        self,
        forecast_error_penalty: float = PREP_FORECAST_ERROR_PENALTY,
        archetype_penalty: float = PREP_ARCHETYPE_PENALTY,
        fixed_value: float | None = None,
    ) -> None:
        """Initialize scoring parameters."""
        self.forecast_error_penalty = forecast_error_penalty
        self.archetype_penalty = archetype_penalty
        self.fixed_value = fixed_value

    def score(self, forecast_error: float, archetype_modifier: float) -> float:
        """Compute preparedness score in [0, 1]."""
        if self.fixed_value is not None:
            return float(min(1.0, max(0.0, self.fixed_value)))
        value = 1.0 - (self.forecast_error_penalty * forecast_error) - (
            self.archetype_penalty * archetype_modifier
        )
        return float(min(1.0, max(0.0, value)))
