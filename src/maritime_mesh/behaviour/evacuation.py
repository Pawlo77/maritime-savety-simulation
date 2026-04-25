"""Evacuation decision policy."""

import math

import numpy as np

from maritime_mesh.constants import (
    EVAC_BASELINE_BIAS,
    EVAC_ERROR_PENALTY,
    EVAC_HAZARD_WEIGHT,
    EVAC_PREP_WEIGHT,
)


class EvacuationPolicy:
    """Bernoulli evacuation policy with logistic score."""

    def __init__(
        self,
        rng: np.random.Generator,
        hazard_weight: float = EVAC_HAZARD_WEIGHT,
        error_penalty: float = EVAC_ERROR_PENALTY,
        prep_weight: float = EVAC_PREP_WEIGHT,
        baseline_bias: float = EVAC_BASELINE_BIAS,
        enabled: bool = True,
    ) -> None:
        """Initialize evacuation model parameters."""
        self.rng = rng
        self.hazard_weight = hazard_weight
        self.error_penalty = error_penalty
        self.prep_weight = prep_weight
        self.baseline_bias = baseline_bias
        self.enabled = enabled

    def evacuation_probability(
        self, w_hat_blend: float, forecast_error: float, p_prep: float
    ) -> float:
        """Return evacuation probability without drawing a random sample."""
        if not self.enabled:
            return 0.0
        logit = (
            self.baseline_bias
            + (self.hazard_weight * w_hat_blend)
            - self.error_penalty * forecast_error
            + self.prep_weight * p_prep
        )
        return 1.0 / (1.0 + math.exp(-logit))

    def should_evacuate(self, w_hat_blend: float, forecast_error: float, p_prep: float) -> bool:
        """Draw evacuation decision based on modelled probability."""
        probability = self.evacuation_probability(
            w_hat_blend=w_hat_blend, forecast_error=forecast_error, p_prep=p_prep
        )
        return bool(self.rng.random() < probability)
