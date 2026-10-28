"""Explicit deals make duplicate games and cross-version replays possible."""

import random
from dataclasses import dataclass

from .cards import DECK, Card
from .types import Hands


@dataclass(frozen=True, slots=True)
class Deal:
    hands: Hands

    def __post_init__(self) -> None:
        if type(self.hands) is not tuple or len(self.hands) != 4:
            raise ValueError("a deal must contain four immutable hands")
        if any(type(hand) is not tuple or len(hand) != 13 for hand in self.hands):
            raise ValueError("each immutable hand must contain 13 cards")
        cards = tuple(card for hand in self.hands for card in hand)
        if any(not isinstance(card, Card) for card in cards) or set(cards) != set(DECK):
            raise ValueError("a deal must contain each of the 52 cards exactly once")

    @classmethod
    def shuffled(cls, seed: int) -> "Deal":
        if type(seed) is not int or seed < 0:
            raise ValueError("deal seed must be a non-negative integer")
        cards = list(DECK)
        random.Random(seed).shuffle(cards)
        return cls(tuple(tuple(sorted(cards[seat::4])) for seat in range(4)))
