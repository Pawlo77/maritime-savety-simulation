"""Evacuation decision policy."""

import math

import numpy as np

from maritime_mesh.constants import EVAC_ERROR_PENALTY, EVAC_HAZARD_WEIGHT, EVAC_PREP_WEIGHT


class EvacuationPolicy:
    """Bernoulli evacuation policy with logistic score."""

    def __init__(
        self,
        rng: np.random.Generator,
        hazard_weight: float = EVAC_HAZARD_WEIGHT,
        error_penalty: float = EVAC_ERROR_PENALTY,
        prep_weight: float = EVAC_PREP_WEIGHT,
        enabled: bool = True,
    ) -> None:
        """Initialize evacuation model parameters."""
        self.rng = rng
        self.hazard_weight = hazard_weight
        self.error_penalty = error_penalty
        self.prep_weight = prep_weight
        self.enabled = enabled

    def should_evacuate(self, w_hat_blend: float, forecast_error: float, p_prep: float) -> bool:
        """Draw evacuation decision based on modelled probability."""
        if not self.enabled:
            return False
        logit = (
            self.hazard_weight * w_hat_blend
            - self.error_penalty * forecast_error
            + self.prep_weight * p_prep
        )
        probability = 1.0 / (1.0 + math.exp(-logit))
        return bool(self.rng.random() < probability)
