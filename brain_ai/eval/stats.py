"""
Small-N statistics for paired benchmark comparisons.

- mcnemar_exact: exact two-sided McNemar test on paired binary outcomes (Dietterich, 1998).
- paired_bootstrap: bootstrap CI for the mean difference of paired per-task scores (handles fractional
  ARC task scores).
- wilson_interval: binomial confidence interval for a single accuracy.
- holm: Holm-Bonferroni step-down adjustment for a family of p-values.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np


def _binom_two_sided_p(k: int, n: int) -> float:
    """Exact two-sided binomial test p-value for k successes in n trials with p = 0.5."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def mcnemar_exact(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> Dict[str, float]:
    """Exact McNemar test. Returns discordant counts and the two-sided p-value."""
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("Paired outcome vectors must have equal length")
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    return {"only_a": only_a, "only_b": only_b, "p_value": _binom_two_sided_p(only_a, only_a + only_b)}


def paired_bootstrap(scores_a: Sequence[float], scores_b: Sequence[float], n_boot: int = 10_000,
                     seed: int = 0, alpha: float = 0.05) -> Dict[str, float]:
    """Percentile bootstrap CI over tasks for mean(scores_a - scores_b)."""
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("Paired score vectors must be 1-D and of equal length")
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    # Two-sided bootstrap p-value for H0: mean difference = 0
    p = 2.0 * min(np.mean(boots <= 0.0), np.mean(boots >= 0.0))
    return {"mean_diff": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi), "p_value": float(min(1.0, p))}


def wilson_interval(k: int, n: int, z: float = 1.959964) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def holm(p_values: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: Dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running
    return adjusted
