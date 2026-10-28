import copy
import json
import math
import random
import subprocess
import sys
import unittest

from hearts.agents import AgentContext, AgentSpec
from hearts.deal import Deal
from hearts.evaluation import Comparison, compare, make_schedule
from hearts.seeds import Seeds
from hearts.statistics import holm_adjust, paired_t_test
from tests.helpers import RandomInputs, random_inputs


class SeedTests(unittest.TestCase):
    def test_stable_named_streams_and_typed_paths(self):
        seeds = Seeds(42)
        expected = seeds.derive("evaluation", "test", 1)
        self.assertEqual(expected, Seeds(42).derive("evaluation", "test", 1))
        self.assertNotEqual(seeds.derive(1), seeds.derive("1"))
        self.assertNotEqual(seeds.derive("train"), seeds.derive("test"))
        self.assertNotEqual(seeds.derive("deal", 0), seeds.derive("agent", 0))
        code = "from hearts.seeds import Seeds; print(Seeds(42).derive('evaluation', 'test', 1))"
        actual = subprocess.check_output([sys.executable, "-c", code], text=True)
        self.assertEqual(int(actual), expected)

    def test_no_global_rng_mutation_or_action_to_deal_coupling(self):
        state = random.getstate()
        seeds = Seeds(0)
        expected = Deal.shuffled(seeds.derive("deal", 1))
        actions = seeds.rng("agent", 0)
        for _ in range(10000):
            actions.random()
        self.assertEqual(Deal.shuffled(seeds.derive("deal", 1)), expected)
        self.assertEqual(random.getstate(), state)

    def test_invalid_seed_inputs(self):
        for seed in (-1, True, 2.0):
            with self.assertRaises(ValueError):
                Seeds(seed)
            with self.assertRaises(ValueError):
                Deal.shuffled(seed)
        with self.assertRaises(TypeError):
            Seeds(0).derive(True)

    def test_invalid_deal_rejected(self):
        hands = Deal.shuffled(1).hands
        with self.assertRaises(ValueError):
            Deal((hands[0], hands[0], hands[2], hands[3]))
        with self.assertRaises(ValueError):
            Deal(list(hands))


class EvaluationTests(unittest.TestCase):
    def test_schedule_balances_seats_directions_and_disjoint_splits(self):
        seeds = Seeds(17)
        schedule = make_schedule(seeds, 3)
        self.assertEqual(len(schedule), 48)
        for block in range(3):
            trials = [trial for trial in schedule if trial.block == block]
            self.assertEqual(len({trial.deal_seed for trial in trials}), 1)
            self.assertEqual(
                {(int(trial.direction), trial.candidate_seat) for trial in trials},
                {(direction, seat) for direction in range(4) for seat in range(4)},
            )
        split_seeds = [
            {trial.deal_seed for trial in make_schedule(seeds, 3, split)}
            for split in ("train", "validation", "test")
        ]
        self.assertEqual(len(set.union(*split_seeds)), 9)
        self.assertEqual(make_schedule(seeds, 3), schedule)

    def test_identical_agents_are_exactly_paired(self):
        recordings = []
        fixture = random_inputs()
        result = compare(
            fixture, fixture, [fixture] * 3, Seeds(55), blocks=2, recordings=recordings
        )
        self.assertEqual(len(result.trials), 32)
        self.assertEqual(len(recordings), 64)
        self.assertEqual(result.block_differences, (0, 0))
        for trial in result.trials:
            self.assertEqual(trial.baseline_scores, trial.candidate_scores)
        for baseline, candidate in zip(recordings[::2], recordings[1::2]):
            self.assertEqual(baseline["initial_hands"], candidate["initial_hands"])
            self.assertEqual(baseline["actions"], candidate["actions"])
            self.assertEqual(
                baseline["metadata"]["agent_seeds"], candidate["metadata"]["agent_seeds"]
            )
        serialized = json.loads(json.dumps(result.to_dict(), allow_nan=False))
        self.assertEqual(serialized["block_differences"], [0, 0])
        self.assertEqual(Comparison.from_dict(serialized), result)

    def test_saved_comparisons_validate_pairing_and_recompute_summaries(self):
        fixture = random_inputs()
        result = compare(fixture, fixture, [fixture] * 3, Seeds(4), blocks=1)
        original = json.loads(json.dumps(result.to_dict()))
        # A stored summary is convenience only; inference recomputes from raw rows.
        original["block_differences"] = [999]
        self.assertEqual(Comparison.from_dict(original).block_differences, (0,))
        for mutation in ("seed", "duplicate", "scores", "missing", "split", "names"):
            bad = copy.deepcopy(original)
            if mutation == "seed":
                bad["trials"][0]["trial"]["deal_seed"] += 1
            elif mutation == "duplicate":
                bad["trials"][0] = bad["trials"][1]
            elif mutation == "scores":
                bad["trials"][0]["candidate_scores"] = [26, 0, 0, 0]
            elif mutation == "missing":
                bad["trials"].pop()
            elif mutation == "split":
                bad["split"] = "validation"
            else:
                bad["opponents"] = ["only-one"]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                Comparison.from_dict(bad)

    def test_rosters_rotate_and_opponent_seeds_are_shared_between_arms(self):
        created = []

        def specification(name):
            def factory(context: AgentContext):
                created.append((name, context))
                return RandomInputs(context)

            return AgentSpec(name, factory)

        compare(
            specification("base"),
            specification("candidate"),
            [specification(name) for name in ("opp1", "opp2", "opp3")],
            Seeds(12),
            blocks=1,
        )
        for index in range(0, len(created), 8):
            baseline, candidate = created[index : index + 4], created[index + 4 : index + 8]
            focal_seat = (index // 8) % 4
            self.assertEqual(baseline[focal_seat][0], "base")
            self.assertEqual(candidate[focal_seat][0], "candidate")
            for seat in range(4):
                self.assertEqual(baseline[seat][1].seed, candidate[seat][1].seed)
                if seat != focal_seat:
                    self.assertEqual(baseline[seat][0], candidate[seat][0])

    def test_schedule_rejects_invalid_inputs(self):
        for blocks in (0, -1, True):
            with self.assertRaises(ValueError):
                make_schedule(Seeds(0), blocks)
        with self.assertRaises(ValueError):
            make_schedule(Seeds(0), 1, "typo")


class StatisticsTests(unittest.TestCase):
    def test_holm_known_example_and_order(self):
        values = holm_adjust([0.04, 0.01, 0.03])
        for actual, expected in zip(values, [0.06, 0.03, 0.06]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(holm_adjust([1, 0, 0.8]), (1, 0, 1))
        self.assertEqual(holm_adjust([]), ())

    def test_holm_rejects_invalid_pvalues(self):
        for values in ([math.nan], [math.inf], [-0.1], [1.1]):
            with self.assertRaises(ValueError):
                holm_adjust(values)

    def test_ttest_rejects_degenerate_or_invalid_samples(self):
        for values in ([], [1], [0, 0], [1, 1], [0, math.nan], [0, math.inf]):
            with self.assertRaises(ValueError):
                paired_t_test(values)

    def test_ttest_known_symmetric_sample(self):
        try:
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest("install .[stats] to test numerical t inference")
        result = paired_t_test([-2, -1, 0, 1, 2])
        self.assertEqual(result.blocks, 5)
        self.assertEqual(result.mean_improvement, 0)
        self.assertAlmostEqual(result.standard_error, math.sqrt(0.5))
        self.assertEqual(result.statistic, 0)
        self.assertEqual(result.pvalue, 1)
        # t_(4, .975) = 2.776445105..., with standard error sqrt(.5).
        self.assertAlmostEqual(result.confidence_interval[1], 1.963243161, places=7)
        self.assertAlmostEqual(result.confidence_interval[0], -1.963243161, places=7)
