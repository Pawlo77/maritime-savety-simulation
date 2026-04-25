"""Small Mesa-compatible abstractions used by this project."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class Stepable(Protocol):
    """Protocol for scheduler-managed agents."""

    def step(self) -> None:
        """Advance the agent by one tick."""


@dataclass
class Model:
    """Minimal model base with scheduler-owned lifecycle."""

    running: bool = True


@dataclass
class Agent:
    """Minimal agent base mirroring Mesa's model linkage."""

    model: Model
    unique_id: int | None = None


@dataclass
class RandomActivation:
    """Simple scheduler that activates agents in insertion order."""

    model: Model
    agents: list[Stepable] = field(default_factory=list)

    def add(self, agent: Stepable) -> None:
        """Register an agent in the scheduler."""
        self.agents.append(agent)

    def step(self) -> None:
        """Step all registered agents once."""
        for agent in list(self.agents):
            agent.step()
