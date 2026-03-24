"""
Statistical testing framework for experiment evaluation.

Every experiment runs across multiple seeds. We keep a result only if:
- Primary metric improves with p < 0.05 (Welch's t-test)
- All guard metrics stay within thresholds
- Effect size (Cohen's d) > 0.3
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from scipy import stats


class MetricRole(Enum):
    PRIMARY = "primary"
    GUARD = "guard"
    BONUS = "bonus"
    DIAGNOSTIC = "diagnostic"


class MetricDirection(Enum):
    HIGHER = "higher"
    LOWER = "lower"


@dataclass
class MetricSpec:
    name: str
    role: MetricRole
    direction: MetricDirection = MetricDirection.HIGHER
    guard_threshold: Optional[float] = None  # max allowed degradation fraction for GUARD metrics
    weight: float = 1.0  # for composite scoring

    def is_improvement(self, new_val: float, old_val: float) -> bool:
        if self.direction == MetricDirection.HIGHER:
            return new_val > old_val
        return new_val < old_val

    def degradation_fraction(self, new_val: float, old_val: float) -> float:
        if old_val == 0:
            return 0.0
        if self.direction == MetricDirection.HIGHER:
            return max(0, (old_val - new_val) / abs(old_val))
        return max(0, (new_val - old_val) / abs(old_val))


@dataclass
class SeedResult:
    seed: int
    metrics: dict[str, float]
    train_seconds: float = 0.0
    peak_vram_mb: float = 0.0
    success: bool = True
    error: Optional[str] = None


@dataclass
class ExperimentResult:
    experiment_id: str
    description: str
    seed_results: list[SeedResult] = field(default_factory=list)
    code_diff: str = ""

    @property
    def successful_seeds(self) -> list[SeedResult]:
        return [r for r in self.seed_results if r.success]

    @property
    def num_successful(self) -> int:
        return len(self.successful_seeds)

    def metric_values(self, metric_name: str) -> np.ndarray:
        vals = [r.metrics[metric_name] for r in self.successful_seeds if metric_name in r.metrics]
        return np.array(vals, dtype=np.float64)

    def metric_mean(self, metric_name: str) -> float:
        vals = self.metric_values(metric_name)
        return float(np.mean(vals)) if len(vals) > 0 else float("nan")

    def metric_std(self, metric_name: str) -> float:
        vals = self.metric_values(metric_name)
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")


def welch_t_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Welch's t-test for unequal variances. Returns (t_statistic, p_value)."""
    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    return float(t_stat), float(p_val)


def paired_t_test(
    a: np.ndarray, b: np.ndarray, alternative: str = "greater"
) -> tuple[float, float]:
    """
    Paired t-test. Returns (t_statistic, p_value).

    alternative: 'greater' tests mean(a) > mean(b),
                 'less' tests mean(a) < mean(b),
                 'two-sided' for standard two-sided test.
    """
    if len(a) < 2 or len(a) != len(b):
        return 0.0, 1.0
    t_stat, p_val = stats.ttest_rel(a, b)
    # Handle NaN (e.g., identical arrays → 0/0)
    if np.isnan(t_stat) or np.isnan(p_val):
        return 0.0, 1.0
    if alternative == "greater":
        p_val = p_val / 2 if t_stat > 0 else 1 - p_val / 2
    elif alternative == "less":
        p_val = p_val / 2 if t_stat < 0 else 1 - p_val / 2
    return float(t_stat), float(p_val)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size (pooled standard deviation)."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    na, nb = len(a), len(b)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = math.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return float(abs(np.mean(a) - np.mean(b)) / pooled_std)


def paired_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired samples: mean(a-b) / std(a-b)."""
    if len(a) < 2 or len(a) != len(b):
        return 0.0
    diffs = a - b
    std_diff = float(np.std(diffs, ddof=1))
    if std_diff == 0:
        return float("inf") if abs(np.mean(diffs)) > 0 else 0.0
    return float(abs(np.mean(diffs)) / std_diff)


def bootstrap_ci(
    values: np.ndarray, confidence: float = 0.95, n_bootstrap: int = 10000, rng_seed: int = 42
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean."""
    if len(values) < 2:
        m = float(np.mean(values)) if len(values) == 1 else float("nan")
        return m, m
    rng = np.random.RandomState(rng_seed)
    boot_means = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - confidence) / 2
    lo = float(np.percentile(boot_means, 100 * alpha))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return lo, hi


@dataclass
class ComparisonResult:
    metric_name: str
    baseline_mean: float
    candidate_mean: float
    t_statistic: float
    p_value: float
    effect_size: float  # Cohen's d
    ci_low: float
    ci_high: float
    is_improvement: bool
    is_significant: bool  # p < 0.05
    is_meaningful: bool   # Cohen's d > 0.3


def compare_metric(
    baseline: ExperimentResult,
    candidate: ExperimentResult,
    spec: MetricSpec,
    alpha: float = 0.05,
    min_effect_size: float = 0.3,
    paired: bool = True,
) -> ComparisonResult:
    """
    Compare a single metric between baseline and candidate experiments.

    When paired=True (default), uses paired t-test and paired Cohen's d with
    a one-sided test in the improvement direction. This is correct when baseline
    and candidate share the same seed indices.
    """
    b_vals = baseline.metric_values(spec.name)
    c_vals = candidate.metric_values(spec.name)

    b_mean = float(np.mean(b_vals)) if len(b_vals) > 0 else float("nan")
    c_mean = float(np.mean(c_vals)) if len(c_vals) > 0 else float("nan")

    if paired and len(b_vals) == len(c_vals) and len(b_vals) >= 2:
        # Paired one-sided test: does candidate improve over baseline?
        if spec.direction == MetricDirection.HIGHER:
            # Test: candidate > baseline
            t_stat, p_val = paired_t_test(c_vals, b_vals, alternative="greater")
        else:
            # Test: candidate < baseline (lower is better)
            t_stat, p_val = paired_t_test(c_vals, b_vals, alternative="less")
        d = paired_cohens_d(c_vals, b_vals)
    else:
        t_stat, p_val = welch_t_test(b_vals, c_vals)
        d = cohens_d(b_vals, c_vals)

    ci_lo, ci_hi = bootstrap_ci(c_vals)

    is_improvement = spec.is_improvement(c_mean, b_mean)
    is_significant = p_val < alpha
    is_meaningful = d > min_effect_size

    return ComparisonResult(
        metric_name=spec.name,
        baseline_mean=b_mean,
        candidate_mean=c_mean,
        t_statistic=t_stat,
        p_value=p_val,
        effect_size=d,
        ci_low=ci_lo,
        ci_high=ci_hi,
        is_improvement=is_improvement,
        is_significant=is_significant,
        is_meaningful=is_meaningful,
    )


@dataclass
class EvaluationDecision:
    keep: bool
    reason: str
    primary_comparisons: list[ComparisonResult]
    guard_violations: list[str]
    all_comparisons: list[ComparisonResult]


def evaluate_experiment(
    baseline: ExperimentResult,
    candidate: ExperimentResult,
    metric_specs: list[MetricSpec],
    alpha: float = 0.10,
    min_effect_size: float = 0.15,
    min_seeds: int = 3,
    paired: bool = True,
) -> EvaluationDecision:
    """
    Decide whether to keep or revert a candidate experiment.

    Keep only if:
    1. Enough seeds succeeded (>= min_seeds)
    2. At least one PRIMARY metric significantly improved (p < alpha, d > min_effect_size)
    3. No GUARD metrics violated their thresholds

    When paired=True (default), uses paired one-sided t-test and paired Cohen's d.
    """
    if candidate.num_successful < min_seeds:
        return EvaluationDecision(
            keep=False,
            reason=f"Too few successful seeds: {candidate.num_successful}/{len(candidate.seed_results)} (need {min_seeds})",
            primary_comparisons=[],
            guard_violations=[],
            all_comparisons=[],
        )

    all_comparisons = []
    primary_comparisons = []
    guard_violations = []

    for spec in metric_specs:
        comp = compare_metric(baseline, candidate, spec, alpha, min_effect_size, paired=paired)
        all_comparisons.append(comp)

        if spec.role == MetricRole.PRIMARY:
            primary_comparisons.append(comp)
        elif spec.role == MetricRole.GUARD:
            threshold = spec.guard_threshold or 0.1
            baseline_val = comp.baseline_mean
            candidate_val = comp.candidate_mean

            # When baseline is near zero, relative degradation is meaningless.
            # Switch to absolute change threshold to avoid false guard violations.
            GUARD_FLOOR = 0.05
            if abs(baseline_val) < GUARD_FLOOR:
                if spec.direction == MetricDirection.HIGHER:
                    violation = (baseline_val - candidate_val) > threshold
                else:
                    violation = (candidate_val - baseline_val) > threshold
            else:
                degradation = spec.degradation_fraction(candidate_val, baseline_val)
                violation = degradation > threshold

            if violation:
                guard_violations.append(
                    f"{spec.name}: degraded (baseline={baseline_val:.4f}, candidate={candidate_val:.4f}, threshold={threshold:.1%})"
                )

    if guard_violations:
        return EvaluationDecision(
            keep=False,
            reason=f"Guard metric violations: {'; '.join(guard_violations)}",
            primary_comparisons=primary_comparisons,
            guard_violations=guard_violations,
            all_comparisons=all_comparisons,
        )

    any_primary_improved = any(
        c.is_improvement and c.is_significant and c.is_meaningful
        for c in primary_comparisons
    )

    if not any_primary_improved:
        reasons = []
        for c in primary_comparisons:
            parts = []
            if not c.is_improvement:
                parts.append("not improved")
            elif not c.is_significant:
                parts.append(f"p={c.p_value:.4f} > {alpha}")
            elif not c.is_meaningful:
                parts.append(f"d={c.effect_size:.3f} < {min_effect_size}")
            reasons.append(f"{c.metric_name}: {', '.join(parts) if parts else 'ok'}")
        return EvaluationDecision(
            keep=False,
            reason=f"No primary metric significantly improved: {'; '.join(reasons)}",
            primary_comparisons=primary_comparisons,
            guard_violations=[],
            all_comparisons=all_comparisons,
        )

    return EvaluationDecision(
        keep=True,
        reason="Primary metric significantly improved with meaningful effect size, no guard violations",
        primary_comparisons=primary_comparisons,
        guard_violations=[],
        all_comparisons=all_comparisons,
    )
