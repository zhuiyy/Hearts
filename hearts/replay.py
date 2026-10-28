"""Privileged full-hand recordings. Playback uses the same engine as live play."""

import json
from pathlib import Path

from .cards import Card
from .deal import Deal
from .engine import HeartsHand
from .rules import RULESET_VERSION
from .seeds import SEED_VERSION
from .types import PassAction, PassDirection, PlayAction

REPLAY_VERSION = 1


def record_hand(hand: HeartsHand, metadata: dict | None = None) -> dict:
    if hand.result is None:
        raise ValueError("full-information recordings can only be exported after a hand ends")
    actions = []
    for record in hand.action_records:
        if isinstance(record.action, PassAction):
            cards = [card.id for card in record.action.cards]
            kind = "pass"
        else:
            cards = [record.action.card.id]
            kind = "play"
        actions.append({"player_id": record.player_id, "kind": kind, "cards": cards})
    return {
        "replay_version": REPLAY_VERSION,
        "ruleset": RULESET_VERSION,
        "seed_version": SEED_VERSION,
        "hand_number": hand.public.hand_number,
        "pass_direction": hand.public.pass_direction.name,
        "match_scores": list(hand.public.match_scores),
        "initial_hands": [[card.id for card in cards] for cards in hand.initial_deal.hands],
        "actions": actions,
        "raw_scores": list(hand.result.raw_scores),
        "scores": list(hand.result.scores),
        "metadata": metadata or {},
    }


def replay_hand(record: dict) -> HeartsHand:
    if type(record.get("replay_version")) is not int or record["replay_version"] != REPLAY_VERSION:
        raise ValueError("unsupported replay version")
    if record.get("ruleset") != RULESET_VERSION:
        raise ValueError("unsupported ruleset")
    deal = Deal(
        tuple(tuple(Card.from_id(cid) for cid in cards) for cards in record["initial_hands"])
    )
    hand = HeartsHand(
        deal,
        hand_number=record["hand_number"],
        match_scores=tuple(record["match_scores"]),
        pass_direction=PassDirection[record["pass_direction"]],
    )
    for entry in record["actions"]:
        cards = tuple(Card.from_id(cid) for cid in entry["cards"])
        if entry["kind"] == "pass":
            action = PassAction(cards)
        elif entry["kind"] == "play" and len(cards) == 1:
            action = PlayAction(cards[0])
        else:
            raise ValueError("invalid replay action")
        hand.submit(entry["player_id"], action)
    if hand.result is None:
        raise ValueError("replay ended before the hand finished")
    if (
        list(hand.result.raw_scores) != record["raw_scores"]
        or list(hand.result.scores) != record["scores"]
    ):
        raise ValueError("replayed scores do not match the recording")
    return hand


def save_record(path: str | Path, record: dict) -> None:
    # Validate serialization before touching the filesystem.
    content = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects earlier experiments from accidental overwrites.
    with path.open("x", encoding="utf-8") as file:
        file.write(content)


def load_record(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        record = json.load(file)
    if not isinstance(record, dict):
        raise ValueError("a recording must be a JSON object")
    return record
