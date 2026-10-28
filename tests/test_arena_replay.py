import copy
import json
import tempfile
import unittest
from pathlib import Path

from hearts.agents import AgentSpec
from hearts.arena import run_hand, run_match
from hearts.deal import Deal
from hearts.engine import HeartsHand, IllegalAction
from hearts.replay import load_record, record_hand, replay_hand, save_record
from hearts.seeds import Seeds
from hearts.types import PassDirection, PlayAction
from tests.helpers import RandomInputs, random_inputs


class ArenaReplayTests(unittest.TestCase):
    def completed_hand(self, direction=PassDirection.LEFT):
        hand = HeartsHand(Deal.shuffled(88), pass_direction=direction)
        run_hand(hand, [random_inputs()] * 4, Seeds(123))
        return hand

    def test_replay_round_trip_all_directions(self):
        for direction in PassDirection:
            hand = self.completed_hand(direction)
            record = json.loads(json.dumps(record_hand(hand, {"root_seed": 123})))
            replay = replay_hand(record)
            self.assertEqual(replay.public, hand.public)
            self.assertEqual(replay.action_records, hand.action_records)

    def test_replay_rejects_tampering_and_incomplete_recordings(self):
        hand = HeartsHand(Deal.shuffled(4))
        with self.assertRaises(ValueError):
            record_hand(hand)
        record = record_hand(self.completed_hand())
        mutations = [
            ("ruleset", "unknown"),
            ("replay_version", 2),
            ("scores", [0, 0, 0, 0]),
            ("raw_scores", [0, 0, 0, 0]),
            ("actions", record["actions"][:-1]),
        ]
        for key, value in mutations:
            bad = copy.deepcopy(record)
            bad[key] = value
            with self.assertRaises(ValueError):
                replay_hand(bad)
        bad = copy.deepcopy(record)
        bad["actions"][0]["player_id"] = 1
        with self.assertRaises(IllegalAction):
            replay_hand(bad)

    def test_record_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested/hand.json"
            record = record_hand(self.completed_hand())
            save_record(path, record)
            self.assertEqual(load_record(path), record)
            with self.assertRaises(FileExistsError):
                save_record(path, record)

    def test_match_accumulates_adjusted_scores_and_stops_after_hand(self):
        completed = []
        result = run_match(
            [random_inputs()] * 4, Seeds(75), target=30, on_hand_end=completed.append
        )
        self.assertGreaterEqual(max(result.scores), 30)
        self.assertLess(max(completed[-1].public.match_scores), 30)
        self.assertEqual(
            result.scores,
            tuple(sum(hand.scores[seat] for hand in result.hands) for seat in range(4)),
        )
        self.assertEqual(
            result.winners,
            tuple(seat for seat, score in enumerate(result.scores) if score == min(result.scores)),
        )
        for number, hand in enumerate(completed):
            self.assertEqual(hand.public.pass_direction, PassDirection(number % 4))

    def test_invalid_json_record_does_not_leave_a_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hand.json"
            with self.assertRaises(ValueError):
                save_record(path, {"invalid": float("nan")})
            self.assertFalse(path.exists())
            path.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_record(path)

    def test_agents_are_separate_and_recreated_each_hand(self):
        contexts, instances = [], []

        def factory(context):
            contexts.append(context)
            instance = RandomInputs(context)
            instances.append(instance)
            return instance

        run_match([AgentSpec("fixture", factory)] * 4, Seeds(15), target=30)
        self.assertGreater(len(contexts), 4)
        self.assertEqual(len({id(instance) for instance in instances}), len(instances))
        self.assertEqual(len({context.seed for context in contexts}), len(contexts))
        shared = instances[0]
        with self.assertRaises(ValueError):
            run_hand(
                HeartsHand(Deal.shuffled(1)),
                [AgentSpec("bad", lambda context: shared)] * 4,
                Seeds(0),
            )

    def test_invalid_strategy_action_is_not_replaced(self):
        class BadInput:
            def act(self, observation):
                return PlayAction(observation.hand[0])

        hand = HeartsHand(Deal.shuffled(1))
        with self.assertRaises(IllegalAction):
            run_hand(hand, [AgentSpec("bad", lambda context: BadInput())] * 4, Seeds(0))
        self.assertEqual(hand.action_records, ())
