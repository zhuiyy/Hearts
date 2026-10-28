"""Inference over independent deal blocks, with explicit degenerate cases."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean, stdev


@dataclass(frozen=True, slots=True)
class TTestResult:
    blocks: int
    mean_improvement: float
    standard_error: float
    statistic: float
    pvalue: float
    confidence_interval: tuple[float, float]


def paired_t_test(differences: Sequence[float], confidence: float = 0.95) -> TTestResult:
    """Two-sided test of E[baseline score - candidate score] = 0.

    Supply ONE average difference per independent deal block. Rotations and
    passing directions within that block must not be counted as extra samples.
    """
    values = tuple(float(value) for value in differences)
    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        raise ValueError("at least two finite independent block differences are required")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    deviation = stdev(values)
    if deviation == 0:
        raise ValueError("zero sample variance: t-test is undefined; report the raw differences")
    try:
        from scipy.stats import ttest_1samp
    except ImportError as error:
        raise RuntimeError(
            'install the optional statistics dependency: pip install -e ".[stats]"'
        ) from error
    result = ttest_1samp(values, popmean=0, alternative="two-sided")
    interval = result.confidence_interval(confidence_level=confidence)
    return TTestResult(
        blocks=len(values),
        mean_improvement=mean(values),
        standard_error=deviation / math.sqrt(len(values)),
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        confidence_interval=(float(interval.low), float(interval.high)),
    )


def holm_adjust(pvalues: Sequence[float]) -> tuple[float, ...]:
    """Holm family-wise correction, returned in the original hypothesis order."""
    values = tuple(float(value) for value in pvalues)
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("p-values must be finite and between zero and one")
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [0.0] * len(values)
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running_max)
    return tuple(adjusted)
