"""Base agent abstraction."""

from abc import ABC, abstractmethod

import numpy as np

from maritime_mesh.mesa_compat import Agent, Model


class AbstractMesaAgent(Agent, ABC):
    """Thin abstract base over mesa.Agent."""

    def __init__(self, model: Model, rng: np.random.Generator) -> None:
        """Initialize agent with shared model and RNG."""
        super().__init__(model)
        self.rng = rng

    @abstractmethod
    def step(self) -> None:
        """Advance one scheduler tick."""
