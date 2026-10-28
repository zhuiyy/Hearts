"""Paired Monte Carlo schedules. A deal, not a seat rotation, is a sample."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from statistics import mean

from .agents import AgentSpec
from .arena import run_hand
from .deal import Deal
from .engine import HeartsHand
from .replay import record_hand
from .rules import RULESET_VERSION
from .seeds import SEED_VERSION, Seeds
from .types import PassDirection, Scores


@dataclass(frozen=True, slots=True)
class Trial:
    block: int
    direction: PassDirection
    candidate_seat: int
    deal_seed: int


@dataclass(frozen=True, slots=True)
class TrialResult:
    trial: Trial
    baseline_scores: Scores
    candidate_scores: Scores

    @property
    def improvement(self) -> int:
        seat = self.trial.candidate_seat
        return self.baseline_scores[seat] - self.candidate_scores[seat]


@dataclass(frozen=True, slots=True)
class Comparison:
    root_seed: int
    split: str
    baseline: str
    candidate: str
    opponents: tuple[str, str, str]
    trials: tuple[TrialResult, ...]

    @property
    def block_differences(self) -> tuple[float, ...]:
        blocks: dict[int, list[int]] = {}
        for result in self.trials:
            blocks.setdefault(result.trial.block, []).append(result.improvement)
        return tuple(mean(values) for values in blocks.values())

    def to_dict(self) -> dict:
        return {
            "evaluation_version": 1,
            "ruleset": RULESET_VERSION,
            "seed_version": SEED_VERSION,
            **asdict(self),
            "block_differences": list(self.block_differences),
        }

    @classmethod
    def from_dict(cls, record: dict) -> "Comparison":
        """Validate raw paired rows before computing any statistical summary."""
        if type(record.get("evaluation_version")) is not int or record["evaluation_version"] != 1:
            raise ValueError("unsupported evaluation version")
        if record.get("ruleset") != RULESET_VERSION or record.get("seed_version") != SEED_VERSION:
            raise ValueError("comparison ruleset or seed version does not match")
        seeds = Seeds(record["root_seed"])
        rows = record["trials"]
        if not rows or len(rows) % 16:
            raise ValueError("each deal block requires all 16 seat/direction trials")
        expected = make_schedule(seeds, len(rows) // 16, record["split"])
        expected_trials = {
            (trial.block, int(trial.direction), trial.candidate_seat): trial for trial in expected
        }
        results = []
        seen = set()
        for row in rows:
            trial_data = row["trial"]
            coordinate = tuple(trial_data[key] for key in ("block", "direction", "candidate_seat"))
            if any(type(value) is not int for value in coordinate):
                raise ValueError("trial block, direction and seat must be integers")
            if coordinate not in expected_trials or coordinate in seen:
                raise ValueError("unknown or duplicate comparison trial")
            trial = expected_trials[coordinate]
            if (
                type(trial_data["deal_seed"]) is not int
                or trial_data["deal_seed"] != trial.deal_seed
            ):
                raise ValueError("trial deal seed does not match its scheduled block")
            seen.add(coordinate)
            baseline_scores = _read_scores(row["baseline_scores"])
            candidate_scores = _read_scores(row["candidate_scores"])
            results.append(TrialResult(trial, baseline_scores, candidate_scores))
        names = (record["baseline"], record["candidate"], *record["opponents"])
        if len(names) != 5 or any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("expected two named focal strategies and three named opponents")
        results.sort(
            key=lambda row: (row.trial.block, row.trial.direction, row.trial.candidate_seat)
        )
        return cls(
            seeds.root, record["split"], names[0], names[1], tuple(names[2:]), tuple(results)
        )


def _read_scores(values: Sequence[int]) -> Scores:
    if len(values) != 4 or any(type(score) is not int or not 0 <= score <= 26 for score in values):
        raise ValueError("expected four integer hand scores")
    normal = sum(values) == 26 and max(values) < 26
    moon = sorted(values) == [0, 26, 26, 26]
    if not (normal or moon):
        raise ValueError("scores do not represent a completed standard Hearts hand")
    return tuple(values)


def make_schedule(seeds: Seeds, blocks: int, split: str = "test") -> tuple[Trial, ...]:
    if type(blocks) is not int or blocks <= 0:
        raise ValueError("blocks must be a positive integer")
    if split not in ("train", "validation", "test"):
        raise ValueError("split must be train, validation or test")
    trials = []
    for block in range(blocks):
        deal_seed = seeds.derive("evaluation", split, "deal", block)
        # A block reuses one deal for every seat and all four passing directions.
        for direction in PassDirection:
            for seat in range(4):
                trials.append(Trial(block, direction, seat, deal_seed))
    return tuple(trials)


def compare(
    baseline: AgentSpec,
    candidate: AgentSpec,
    opponents: Sequence[AgentSpec],
    seeds: Seeds,
    blocks: int,
    split: str = "test",
    recordings: list[dict] | None = None,
) -> Comparison:
    """Replace one focal agent; keep three opponents and paired RNG streams fixed.

    Each block costs 16 trials x 2 arms = 32 full hands. Optional recordings are
    privileged audit artifacts, not observations supplied to agents.
    """
    if len(opponents) != 3:
        raise ValueError("exactly three opponent specifications are required")
    results = []
    for trial in make_schedule(seeds, blocks, split):
        deal = Deal.shuffled(trial.deal_seed)
        scores = []
        for arm, focal in (("baseline", baseline), ("candidate", candidate)):
            roster = [focal, *opponents]
            # Rotate the entire roster: opponent relative positions stay fixed.
            cut = 4 - trial.candidate_seat
            roster = roster[cut:] + roster[:cut]
            hand = HeartsHand(deal, pass_direction=trial.direction)
            stream = ("evaluation", split, trial.block, trial.direction.name, trial.candidate_seat)
            result = run_hand(hand, roster, seeds, stream=stream)
            scores.append(result.scores)
            if recordings is not None:
                recordings.append(
                    record_hand(
                        hand,
                        {
                            "root_seed": seeds.root,
                            "split": split,
                            "block": trial.block,
                            "arm": arm,
                            "agents": [spec.name for spec in roster],
                            "deal_seed": trial.deal_seed,
                            "agent_seeds": [
                                seeds.derive(*stream, "agent", seat) for seat in range(4)
                            ],
                        },
                    )
                )
        results.append(TrialResult(trial, scores[0], scores[1]))
    return Comparison(
        seeds.root,
        split,
        baseline.name,
        candidate.name,
        tuple(spec.name for spec in opponents),
        tuple(results),
    )
