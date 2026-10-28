"""Orchestration only. Every action is executed by HeartsHand.submit()."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .agents import AgentContext, AgentSpec
from .deal import Deal
from .engine import HeartsHand
from .seeds import Seeds
from .types import HandResult, Scores


def run_hand(
    hand: HeartsHand,
    agents: Sequence[AgentSpec],
    seeds: Seeds,
    stream: tuple[str | int, ...] = (),
) -> HandResult:
    if len(agents) != 4:
        raise ValueError("exactly four agent specifications are required")
    if hand.action_records or hand.result is not None:
        raise ValueError("run_hand requires a fresh hand")
    players = [
        spec.create(
            AgentContext(seat, hand.public.hand_number, seeds.derive(*stream, "agent", seat))
        )
        for seat, spec in enumerate(agents)
    ]
    if len({id(player) for player in players}) != 4:
        raise ValueError("factories must return separate agent instances for each seat")

    while hand.result is None:
        seat = hand.public.current_player
        action = players[seat].act(hand.observe(seat))
        # Illegal actions stop the run. Never silently substitute another action.
        hand.submit(seat, action)
    return hand.result


@dataclass(frozen=True, slots=True)
class MatchResult:
    scores: Scores
    winners: tuple[int, ...]  # Ties share the win; no arbitrary tie-breaking.
    hands: tuple[HandResult, ...]


def run_match(
    agents: Sequence[AgentSpec],
    seeds: Seeds,
    target: int = 100,
    max_hands: int = 1000,
    on_hand_end: Callable[[HeartsHand], None] | None = None,
) -> MatchResult:
    if type(target) is not int or target <= 0 or type(max_hands) is not int or max_hands <= 0:
        raise ValueError("target and max_hands must be positive integers")
    scores = (0, 0, 0, 0)
    results = []
    for number in range(max_hands):
        deal = Deal.shuffled(seeds.derive("match", "deal", number))
        hand = HeartsHand(deal, hand_number=number, match_scores=scores)
        result = run_hand(hand, agents, seeds, stream=("match", number))
        results.append(result)
        scores = tuple(total + points for total, points in zip(scores, result.scores))
        if on_hand_end is not None:
            on_hand_end(hand)
        if max(scores) >= target:
            winners = tuple(seat for seat, score in enumerate(scores) if score == min(scores))
            return MatchResult(scores, winners, tuple(results))
    raise RuntimeError("match exceeded max_hands; no completed match result was produced")
