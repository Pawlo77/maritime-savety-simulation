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

    def remove(self, agent: Stepable) -> None:
        """Unregister an agent if present."""
        if agent in self.agents:
            self.agents.remove(agent)

    def step(self) -> None:
        """Step all registered agents once."""
        for agent in list(self.agents):
            agent.step()


@dataclass
class PhaseScheduler:
    """Phase-based scheduler with explicit activation order."""

    model: Model
    phases: tuple[str, ...]
    _phase_agents: dict[str, list[Stepable]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize phase buckets in configured order."""
        self._phase_agents = {phase: [] for phase in self.phases}

    def add(self, phase: str, agent: Stepable) -> None:
        """Register an agent to a named phase."""
        if phase not in self._phase_agents:
            raise ValueError(f"Unknown scheduler phase '{phase}'.")
        self._phase_agents[phase].append(agent)

    def remove(self, phase: str, agent: Stepable) -> None:
        """Unregister an agent from a named phase if present."""
        if phase not in self._phase_agents:
            raise ValueError(f"Unknown scheduler phase '{phase}'.")
        if agent in self._phase_agents[phase]:
            self._phase_agents[phase].remove(agent)

    def step(self) -> None:
        """Step all agents according to phase order."""
        for phase in self.phases:
            for agent in list(self._phase_agents[phase]):
                agent.step()
