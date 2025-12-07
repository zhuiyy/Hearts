from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple
import random


class Suit(IntEnum):
    HEARTS = 0
    DIAMONDS = 1
    CLUBS = 2
    SPADES = 3


class PassDirection(IntEnum):
    LEFT = 0      # Clockwise
    RIGHT = 1     # Counter-Clockwise
    ACROSS = 2    # Across
    KEEP = 3      # No Pass


@dataclass(frozen=True)
class Card:
    suit: Suit
    rank: int  # 1-13, Ace = 1

    def __repr__(self) -> str:
        suit_symbols = {
            Suit.HEARTS: "H",
            Suit.DIAMONDS: "D",
            Suit.CLUBS: "C",
            Suit.SPADES: "S",
        }
        return f"{suit_symbols[self.suit]}{self.rank}"

    def __lt__(self, other: "Card") -> bool:
        if self.suit == other.suit:
            return self.rank < other.rank
        return self.suit < other.suit

    def value(self) -> int:
        if self.suit == Suit.HEARTS:
            return 1
        if self.suit == Suit.SPADES and self.rank == 12:
            return 13
        return 0

    def to_id(self) -> int:
        return self.suit.value * 13 + (self.rank - 1)


@dataclass
class Player:
    player_id: int
    hand: List[Card] = field(default_factory=list)
    points: int = 0
    table: List[Card] = field(default_factory=list)


@dataclass
class TrickRecord:
    winner: int
    score: int
    lead_suit: Optional[Suit]
    cards: List[Tuple[Card, int]] = field(default_factory=list)


@dataclass
class PassEvent:
    """
    Represents the event of passing cards at the start of the game.
    """
    player_id: int
    direction: PassDirection
    passed_cards: List[Card]
    received_cards: List[Card]


@dataclass
class PlayEvent:
    """
    Represents a single card play event with full context for Transformer updates.
    """
    player_id: int
    card: Card
    round_number: int  # 1-13
    trick_number: int  # 1-13
    is_lead: bool
    current_table: List[Tuple[Card, int]]  # Cards played so far in this trick
    heart_broken: bool
    piggy_pulled: bool
    legal_actions: List[Card]  # Actions that were available to the player


@dataclass
class GameState:
    players: List[Player] = field(default_factory=lambda: [Player(player_id=i) for i in range(4)])
    deck: List[Card] = field(default_factory=list)
    table: List[Tuple[Card, int]] = field(default_factory=list)
    current_table: List[Tuple[Card, int]] = field(default_factory=list)
    current_suit: Optional[Suit] = None
    heart_broken: bool = False
    piggy_pulled: bool = False

    def reset(self) -> None:
        self.current_suit = None
        self.heart_broken = False
        self.piggy_pulled = False
        self.table.clear()
        self.current_table.clear()
        self.deck = [Card(suit, rank) for suit in Suit for rank in range(1, 14)]
        random.shuffle(self.deck)
        for player in self.players:
            player.hand.clear()
            player.points = 0
            player.table.clear()
        for index, card in enumerate(self.deck):
            self.players[index % 4].hand.append(card)
        for player in self.players:
            player.hand.sort()

    def get_first_player(self) -> int:
        for i, player in enumerate(self.players):
            for card in player.hand:
                if card.suit == Suit.CLUBS and card.rank == 2:
                    return i
        return 0

    def player_info(self, player_id: int) -> dict:
        player_state = self.players[player_id]
        return {
            'player_id': player_state.player_id,
            'hand': list(player_state.hand),
            'points': player_state.points,
            'player_table': list(player_state.table),
            'scoreboard': [p.points for p in self.players],
            'table': list(self.table),
            'current_table': list(self.current_table),
            'current_suit': self.current_suit,
            'heart_broken': self.heart_broken,
            'piggy_pulled': self.piggy_pulled,
            'current_order': len(self.current_table)
        }
   
    def game_info(self) -> dict:
        return {
            'table': list(self.table),
            'current_table': list(self.current_table),
            'current_suit': self.current_suit,
            'heart_broken': self.heart_broken,
            'piggy_pulled': self.piggy_pulled,
            'current_order': len(self.current_table)
            
        }
