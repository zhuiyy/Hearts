import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from hearts.agents import AgentContext
from hearts.cli import main
from hearts.deal import Deal
from hearts.engine import HeartsHand
from hearts.evaluation import Comparison, TrialResult, make_schedule
from hearts.replay import load_record, replay_hand
from hearts.seeds import Seeds
from hearts.terminal import TerminalPlayer, UserQuit, parse_selection
from hearts.types import PassAction, Phase, PlayAction


class TerminalTests(unittest.TestCase):
    def test_input_validation(self):
        self.assertEqual(parse_selection("1, 3 5", 3, 13), (0, 2, 4))
        for text in ("1 1 2", "0 2 3", "1 2 14", "A 2 3", "1 2", ""):
            with self.assertRaises(ValueError):
                parse_selection(text, 3, 13)
        with self.assertRaises(UserQuit):
            parse_selection("q", 1, 13)

    def test_terminal_adapter_retries_invalid_input(self):
        responses = iter(["", "1 1 2", "1 2 3"])
        output = []
        terminal = TerminalPlayer(
            AgentContext(0, 0, 1), read=lambda prompt: next(responses), write=output.append
        )
        obs = HeartsHand(Deal.shuffled(42)).observe(0)
        action = terminal.act(obs)
        self.assertEqual(action, PassAction(obs.legal_cards[:3]))
        self.assertTrue(any("不同的编号" in line for line in output))

    def test_schedule_command_saves_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schedule.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(["schedule", "--seed", "42", "--blocks", "2", "--output", str(path)]), 0
                )
            record = json.loads(path.read_text())
            self.assertEqual(record["root_seed"], 42)
            self.assertEqual(len(record["trials"]), 32)

    def test_quit_is_not_reported_as_success(self):
        with patch("hearts.cli.run_hand", side_effect=UserQuit), redirect_stdout(io.StringIO()):
            self.assertEqual(main(["play"]), 130)

    def test_analyze_rejects_incomplete_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.json"
            path.write_text(
                json.dumps({"evaluation_version": 1, "ruleset": "standard-hearts-v1", "trials": []})
            )
            with self.assertRaises(SystemExit) as error, redirect_stderr(io.StringIO()):
                main(["analyze", str(path)])
            self.assertEqual(error.exception.code, 2)

    def test_manual_command_completes_and_replay_command_validates_it(self):
        # Precompute manual menu selections through the engine, then exercise
        # the real subprocess entry point and terminal parser from end to end.
        hand = HeartsHand(Deal.shuffled(Seeds(42).derive("manual", "deal", 0)))
        responses = []
        while hand.result is None:
            seat = hand.public.current_player
            obs = hand.observe(seat)
            responses.append("")  # Seat handover prompt.
            if obs.public.phase == Phase.PASSING:
                responses.append("1 2 3")
                action = PassAction(obs.legal_cards[:3])
            else:
                responses.append("1")
                action = PlayAction(obs.legal_cards[0])
            hand.submit(seat, action)
        with tempfile.TemporaryDirectory() as directory:
            process = subprocess.run(
                [sys.executable, "-m", "hearts", "play", "--seed", "42", "--record-dir", directory],
                input="\n".join(responses) + "\n",
                text=True,
                capture_output=True,
                timeout=10,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            path = Path(directory) / "hand_0000.json"
            self.assertEqual(replay_hand(load_record(path)).result, hand.result)
            replay = subprocess.run(
                [sys.executable, "-m", "hearts", "replay", str(path)],
                text=True,
                capture_output=True,
                timeout=10,
            )
            self.assertEqual(replay.returncode, 0, replay.stderr)
            self.assertIn("13/13", replay.stdout)

    def test_analyze_uses_independent_block_count(self):
        try:
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest("install .[stats] to exercise the analysis command")
        rows = []
        for trial in make_schedule(Seeds(9), 5):
            baseline = [5, 4, 4]
            baseline.insert(trial.candidate_seat, 13)
            candidate = baseline.copy()
            improvement = trial.block - 2
            candidate[trial.candidate_seat] -= improvement
            candidate[(trial.candidate_seat + 1) % 4] += improvement
            rows.append(TrialResult(trial, tuple(baseline), tuple(candidate)))
        comparison = Comparison(9, "test", "base", "candidate", ("a", "b", "c"), tuple(rows))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.json"
            path.write_text(json.dumps(comparison.to_dict()), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["analyze", str(path)]), 0)
            self.assertIn("独立牌局块: 5", output.getvalue())
            self.assertIn("双侧 p=1", output.getvalue())
