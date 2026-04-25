"""Weather grid cell model."""

from dataclasses import dataclass

from maritime_mesh.constants import (
    WEATHER_WEIGHT_SEA_STATE,
    WEATHER_WEIGHT_VISIBILITY,
    WEATHER_WEIGHT_WIND,
)


@dataclass
class WeatherCell:
    """Single weather-grid cell over three channels."""

    sea_state: float
    visibility: float
    wind_norm: float

    def hazard(self) -> float:
        """Compute weighted composite hazard for this cell."""
        value = (
            WEATHER_WEIGHT_SEA_STATE * self.sea_state
            + WEATHER_WEIGHT_VISIBILITY * (1.0 - self.visibility)
            + WEATHER_WEIGHT_WIND * self.wind_norm
        )
        return float(min(1.0, max(0.0, value)))
