"""Card identity is stable: clubs, diamonds, spades, hearts; 2 through ace."""

from dataclasses import dataclass
from enum import IntEnum


class Suit(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    SPADES = 2
    HEARTS = 3


@dataclass(frozen=True, slots=True, order=True)
class Card:
    suit: Suit
    rank: int  # 2..14; ace is 14 everywhere, including comparisons.

    def __post_init__(self) -> None:
        if not isinstance(self.suit, Suit):
            raise ValueError("suit must be a Suit")
        if type(self.rank) is not int or not 2 <= self.rank <= 14:
            raise ValueError("rank must be an integer from 2 to 14")

    @property
    def id(self) -> int:
        return int(self.suit) * 13 + self.rank - 2

    @property
    def points(self) -> int:
        if self.suit == Suit.HEARTS:
            return 1
        return 13 if self.suit == Suit.SPADES and self.rank == 12 else 0

    @classmethod
    def from_id(cls, card_id: int) -> "Card":
        if type(card_id) is not int or not 0 <= card_id < 52:
            raise ValueError("card id must be an integer from 0 to 51")
        suit, rank_index = divmod(card_id, 13)
        return cls(Suit(suit), rank_index + 2)

    def __str__(self) -> str:
        rank = {11: "J", 12: "Q", 13: "K", 14: "A"}.get(self.rank, str(self.rank))
        return "CDSH"[self.suit] + rank


DECK = tuple(Card.from_id(card_id) for card_id in range(52))
TWO_OF_CLUBS = Card(Suit.CLUBS, 2)
