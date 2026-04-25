"""Life-raft deployment and survival models."""

import numpy as np

from maritime_mesh.constants import RAFT_BASE_SUCCESS, RAFT_PREP_BONUS, RAFT_WEATHER_PENALTY


class RaftDeploymentModel:
    """Probabilistic raft deployment success model."""

    def __init__(
        self,
        rng: np.random.Generator,
        base_success: float = RAFT_BASE_SUCCESS,
        weather_penalty: float = RAFT_WEATHER_PENALTY,
        prep_bonus: float = RAFT_PREP_BONUS,
    ) -> None:
        """Initialize deployment probability parameters."""
        self.rng = rng
        self.base_success = base_success
        self.weather_penalty = weather_penalty
        self.prep_bonus = prep_bonus

    def deploy(self, current_hazard: float, p_prep: float) -> bool:
        """Attempt deployment with hazard and preparedness effects."""
        probability = self.base_success - (self.weather_penalty * current_hazard) + (
            self.prep_bonus * p_prep
        )
        probability = float(min(1.0, max(0.0, probability)))
        return bool(self.rng.random() < probability)


class SurvivalModel:
    """Per-tick raft survival model influenced by local hazard."""

    def __init__(self, rng: np.random.Generator) -> None:
        """Initialize with seeded RNG."""
        self.rng = rng

    def survives_tick(self, current_hazard: float) -> bool:
        """Draw per-tick survival Bernoulli."""
        probability = float(min(1.0, max(0.0, 1.0 - current_hazard)))
        return bool(self.rng.random() < probability)
