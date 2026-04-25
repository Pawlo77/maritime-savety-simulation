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
    weight_sea_state: float = WEATHER_WEIGHT_SEA_STATE
    weight_visibility: float = WEATHER_WEIGHT_VISIBILITY
    weight_wind: float = WEATHER_WEIGHT_WIND

    def hazard(self) -> float:
        """Compute weighted composite hazard for this cell."""
        weight_sum = self.weight_sea_state + self.weight_visibility + self.weight_wind
        if weight_sum <= 0.0:
            return 0.0
        value = (
            self.weight_sea_state * self.sea_state
            + self.weight_visibility * (1.0 - self.visibility)
            + self.weight_wind * self.wind_norm
        )
        return float(min(1.0, max(0.0, value / weight_sum)))
