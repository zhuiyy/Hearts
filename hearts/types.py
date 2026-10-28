"""Immutable values crossing the engine / player boundary."""

from dataclasses import dataclass
from enum import Enum, IntEnum

from .cards import Card

Scores = tuple[int, int, int, int]
Hands = tuple[tuple[Card, ...], tuple[Card, ...], tuple[Card, ...], tuple[Card, ...]]


class PassDirection(IntEnum):
    LEFT = 0
    RIGHT = 1
    ACROSS = 2
    KEEP = 3

    @property
    def offset(self) -> int:
        """Seats increase clockwise; left is the next clockwise seat."""
        return (1, 3, 2, 0)[self]


class Phase(Enum):
    PASSING = "passing"
    PLAYING = "playing"
    FINISHED = "finished"


@dataclass(frozen=True, slots=True)
class PassAction:
    cards: tuple[Card, Card, Card]


@dataclass(frozen=True, slots=True)
class PlayAction:
    card: Card


Action = PassAction | PlayAction


@dataclass(frozen=True, slots=True)
class Play:
    player_id: int
    card: Card


@dataclass(frozen=True, slots=True)
class Trick:
    plays: tuple[Play, Play, Play, Play]
    winner: int
    points: int


@dataclass(frozen=True, slots=True)
class HandResult:
    raw_scores: Scores
    scores: Scores  # After the shoot-the-moon adjustment.
    moon_shooter: int | None


@dataclass(frozen=True, slots=True)
class PublicState:
    hand_number: int  # Zero-based: determines the four-hand passing cycle.
    phase: Phase
    current_player: int | None
    pass_direction: PassDirection
    hearts_broken: bool
    hand_sizes: tuple[int, int, int, int]
    raw_scores: Scores
    match_scores: Scores  # Cumulative scores BEFORE this hand.
    current_trick: tuple[Play, ...]
    completed_tricks: tuple[Trick, ...]
    result: HandResult | None


@dataclass(frozen=True, slots=True)
class Observation:
    player_id: int
    public: PublicState
    hand: tuple[Card, ...]
    legal_cards: tuple[Card, ...]
    passed_cards: tuple[Card, ...]
    received_cards: tuple[Card, ...]


@dataclass(frozen=True, slots=True)
class ActionRecord:
    player_id: int
    action: Action
