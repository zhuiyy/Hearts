"""The only game state machine: validate first, then apply exactly one action."""

from .cards import TWO_OF_CLUBS, Card, Suit
from .deal import Deal
from .rules import finish_trick, legal_plays, score_hand
from .types import (
    Action,
    ActionRecord,
    HandResult,
    Observation,
    PassAction,
    PassDirection,
    Phase,
    Play,
    PlayAction,
    PublicState,
    Scores,
    Trick,
)


class IllegalAction(ValueError):
    """An invalid submission leaves the engine unchanged."""


class HeartsHand:
    def __init__(
        self,
        deal: Deal,
        hand_number: int = 0,
        match_scores: Scores = (0, 0, 0, 0),
        pass_direction: PassDirection | None = None,
    ) -> None:
        if type(hand_number) is not int or hand_number < 0:
            raise ValueError("hand_number must be a non-negative integer")
        if (
            type(match_scores) is not tuple
            or len(match_scores) != 4
            or any(type(score) is not int or score < 0 for score in match_scores)
        ):
            raise ValueError("match_scores must be four non-negative integers")
        if pass_direction is not None and not isinstance(pass_direction, PassDirection):
            raise ValueError("pass_direction must be a PassDirection")

        self._deal = deal
        self._hand_number = hand_number
        self._match_scores = match_scores
        self._direction = (
            PassDirection(hand_number % 4) if pass_direction is None else pass_direction
        )
        self._hands = [set(hand) for hand in deal.hands]
        self._passed: list[tuple[Card, ...]] = [(), (), (), ()]
        self._received: list[tuple[Card, ...]] = [(), (), (), ()]
        self._pending_passes: dict[int, tuple[Card, ...]] = {}
        self._table: list[Play] = []
        self._tricks: list[Trick] = []
        self._points = [0, 0, 0, 0]
        self._hearts_broken = False
        self._result: HandResult | None = None
        self._records: list[ActionRecord] = []
        self._phase = Phase.PASSING
        self._current_player: int | None = 0
        if self._direction == PassDirection.KEEP:
            self._start_playing()

    @property
    def initial_deal(self) -> Deal:
        """Privileged replay data. Never pass the engine itself to an agent."""
        return self._deal

    @property
    def action_records(self) -> tuple[ActionRecord, ...]:
        """Privileged: includes all players' private passing submissions."""
        return tuple(self._records)

    @property
    def result(self) -> HandResult | None:
        return self._result

    @property
    def public(self) -> PublicState:
        return PublicState(
            hand_number=self._hand_number,
            phase=self._phase,
            current_player=self._current_player,
            pass_direction=self._direction,
            hearts_broken=self._hearts_broken,
            hand_sizes=tuple(len(hand) for hand in self._hands),
            raw_scores=tuple(self._points),
            match_scores=self._match_scores,
            current_trick=tuple(self._table),
            completed_tricks=tuple(self._tricks),
            result=self._result,
        )

    def observe(self, player_id: int) -> Observation:
        self._validate_player(player_id)
        legal = ()
        if player_id == self._current_player:
            if self._phase == Phase.PASSING:
                legal = tuple(sorted(self._hands[player_id]))
            elif self._phase == Phase.PLAYING:
                legal = legal_plays(
                    self._hands[player_id], self._table, len(self._tricks), self._hearts_broken
                )
        return Observation(
            player_id=player_id,
            public=self.public,
            hand=tuple(sorted(self._hands[player_id])),
            legal_cards=legal,
            passed_cards=self._passed[player_id],
            received_cards=self._received[player_id],
        )

    def submit(self, player_id: int, action: Action) -> PublicState:
        self._validate_player(player_id)
        if self._phase == Phase.FINISHED:
            raise IllegalAction("the hand is already finished")
        if player_id != self._current_player:
            raise IllegalAction("it is not this player's turn")
        if self._phase == Phase.PASSING:
            self._submit_pass(player_id, action)
        else:
            self._submit_play(player_id, action)
        self._records.append(ActionRecord(player_id, action))
        return self.public

    @staticmethod
    def _validate_player(player_id: int) -> None:
        if type(player_id) is not int or not 0 <= player_id < 4:
            raise ValueError("player_id must be an integer from 0 to 3")

    def _submit_pass(self, player_id: int, action: Action) -> None:
        if not isinstance(action, PassAction) or type(action.cards) is not tuple:
            raise IllegalAction("passing requires PassAction with an immutable card tuple")
        cards = action.cards
        if len(cards) != 3 or any(not isinstance(card, Card) for card in cards):
            raise IllegalAction("pass exactly three cards")
        if len(set(cards)) != 3 or not set(cards) <= self._hands[player_id]:
            raise IllegalAction("pass three distinct cards from your own hand")

        self._pending_passes[player_id] = cards
        if len(self._pending_passes) < 4:
            self._current_player = player_id + 1
            return

        # Nobody receives (or loses) cards until all four submissions are sealed.
        for seat, selected in self._pending_passes.items():
            self._hands[seat].difference_update(selected)
        for seat in range(4):
            source = (seat - self._direction.offset) % 4
            self._passed[seat] = self._pending_passes[seat]
            self._received[seat] = self._pending_passes[source]
            self._hands[seat].update(self._received[seat])
        self._pending_passes.clear()
        self._start_playing()

    def _start_playing(self) -> None:
        self._phase = Phase.PLAYING
        self._current_player = next(
            seat for seat, hand in enumerate(self._hands) if TWO_OF_CLUBS in hand
        )

    def _submit_play(self, player_id: int, action: Action) -> None:
        if not isinstance(action, PlayAction) or not isinstance(action.card, Card):
            raise IllegalAction("playing requires PlayAction with one card")
        if action.card not in self.observe(player_id).legal_cards:
            raise IllegalAction("card is not legal in this position")

        self._hands[player_id].remove(action.card)
        self._table.append(Play(player_id, action.card))
        if action.card.suit == Suit.HEARTS:
            self._hearts_broken = True
        if len(self._table) < 4:
            self._current_player = (player_id + 1) % 4
            return

        trick = finish_trick(tuple(self._table))
        self._tricks.append(trick)
        self._points[trick.winner] += trick.points
        self._table.clear()
        self._current_player = trick.winner
        if len(self._tricks) == 13:
            self._result = score_hand(tuple(self._points))
            self._phase = Phase.FINISHED
            self._current_player = None
