"""Player action protocol, instance factories and per-player context."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .types import Action, Observation


@dataclass(frozen=True, slots=True)
class AgentContext:
    player_id: int
    hand_number: int
    seed: int  # The agent owns a local RNG initialized from this seed.


class Agent(Protocol):
    def act(self, observation: Observation) -> Action:
        """Choose an action using this player's observation."""
        ...


@dataclass(frozen=True, slots=True)
class AgentSpec:
    name: str
    create: Callable[[AgentContext], Agent]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip() or not callable(self.create):
            raise ValueError("an agent specification needs a name and a factory")
