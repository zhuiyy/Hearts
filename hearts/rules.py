"""Pure rules. All callers, including training and UI, use these same rules."""

from collections.abc import Collection, Sequence

from .cards import TWO_OF_CLUBS, Card, Suit
from .types import HandResult, Play, Scores, Trick

RULESET_VERSION = "standard-hearts-v1"


def legal_plays(
    hand: Collection[Card],
    current_trick: Sequence[Play],
    completed_tricks: int,
    hearts_broken: bool,
) -> tuple[Card, ...]:
    if not hand:
        return ()

    if current_trick:
        lead_suit = current_trick[0].card.suit
        following = [card for card in hand if card.suit == lead_suit]
        if following:
            return tuple(sorted(following))
        if completed_tricks == 0:
            # First-trick points are forbidden unless every remaining card scores.
            safe = [card for card in hand if card.points == 0]
            return tuple(sorted(safe or hand))
        return tuple(sorted(hand))

    if completed_tricks == 0:
        return (TWO_OF_CLUBS,) if TWO_OF_CLUBS in hand else ()
    if not hearts_broken:
        non_hearts = [card for card in hand if card.suit != Suit.HEARTS]
        return tuple(sorted(non_hearts or hand))
    return tuple(sorted(hand))


def finish_trick(plays: tuple[Play, Play, Play, Play]) -> Trick:
    lead_suit = plays[0].card.suit
    winner = max(
        (play for play in plays if play.card.suit == lead_suit),
        key=lambda play: play.card.rank,
    ).player_id
    return Trick(plays, winner, sum(play.card.points for play in plays))


def score_hand(raw_scores: Scores) -> HandResult:
    if (
        len(raw_scores) != 4
        or any(type(score) is not int or score < 0 for score in raw_scores)
        or sum(raw_scores) != 26
    ):
        raise ValueError("a completed hand must distribute exactly 26 raw points")
    shooter = next((seat for seat, score in enumerate(raw_scores) if score == 26), None)
    if shooter is None:
        return HandResult(raw_scores, raw_scores, None)
    scores = tuple(0 if seat == shooter else 26 for seat in range(4))
    return HandResult(raw_scores, scores, shooter)
