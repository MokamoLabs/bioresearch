"""Tests for the statistical testing framework."""

import numpy as np
import pytest

from engine.metrics import (
    MetricSpec,
    MetricRole,
    MetricDirection,
    ExperimentResult,
    SeedResult,
    welch_t_test,
    paired_t_test,
    cohens_d,
    paired_cohens_d,
    bootstrap_ci,
    compare_metric,
    evaluate_experiment,
)


def _make_experiment(metric_name: str, values: list[float], experiment_id: str = "test") -> ExperimentResult:
    """Helper to create an ExperimentResult from metric values."""
    seeds = []
    for i, v in enumerate(values):
        seeds.append(SeedResult(seed=i, metrics={metric_name: v}))
    return ExperimentResult(experiment_id=experiment_id, description="test", seed_results=seeds)


class TestWelchTTest:
    def test_identical_distributions(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        _, p = welch_t_test(a, a.copy())
        assert p > 0.9  # Should not be significant

    def test_clearly_different(self):
        a = np.array([1.0, 1.1, 0.9, 1.0, 1.05])
        b = np.array([5.0, 5.1, 4.9, 5.0, 5.05])
        _, p = welch_t_test(a, b)
        assert p < 0.001  # Should be highly significant

    def test_too_few_samples(self):
        a = np.array([1.0])
        b = np.array([2.0])
        _, p = welch_t_test(a, b)
        assert p == 1.0


class TestCohensD:
    def test_no_difference(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = cohens_d(a, a.copy())
        assert d == 0.0

    def test_large_effect(self):
        a = np.array([1.0, 1.1, 0.9, 1.0, 1.05])
        b = np.array([3.0, 3.1, 2.9, 3.0, 3.05])
        d = cohens_d(a, b)
        assert d > 2.0  # Very large effect

    def test_small_effect(self):
        rng = np.random.RandomState(42)
        a = rng.randn(100)
        b = rng.randn(100) + 0.1  # Small shift
        d = cohens_d(a, b)
        assert 0 < d < 0.5


class TestPairedTTest:
    def test_identical_samples(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        _, p = paired_t_test(a, a.copy(), alternative="greater")
        assert p > 0.4  # Not significant

    def test_clearly_better(self):
        a = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        _, p = paired_t_test(a, b, alternative="greater")
        assert p < 0.01  # Highly significant (consistent improvement)

    def test_one_sided_direction(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        # a < b, so "greater" test for a should NOT be significant
        _, p = paired_t_test(a, b, alternative="greater")
        assert p > 0.9

    def test_mismatched_lengths(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0, 3.0])
        _, p = paired_t_test(a, b)
        assert p == 1.0

    def test_too_few_samples(self):
        a = np.array([1.0])
        b = np.array([2.0])
        _, p = paired_t_test(a, b)
        assert p == 1.0


class TestPairedCohensD:
    def test_no_difference(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = paired_cohens_d(a, a.copy())
        assert d == 0.0

    def test_consistent_shift(self):
        a = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = paired_cohens_d(a, b)
        # All diffs are ~0.1, std(diffs) near 0 → d is very large
        assert d > 100

    def test_noisy_improvement(self):
        rng = np.random.RandomState(42)
        a = rng.randn(10) + 0.5  # shifted up
        b = rng.randn(10)
        d = paired_cohens_d(a, b)
        assert d > 0


class TestBootstrapCI:
    def test_narrow_ci_for_constant(self):
        values = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        lo, hi = bootstrap_ci(values)
        assert lo == 1.0
        assert hi == 1.0

    def test_ci_contains_mean(self):
        rng = np.random.RandomState(42)
        values = rng.randn(50)
        lo, hi = bootstrap_ci(values)
        mean = np.mean(values)
        assert lo <= mean <= hi

    def test_single_value(self):
        lo, hi = bootstrap_ci(np.array([5.0]))
        assert lo == 5.0
        assert hi == 5.0


class TestCompareMetric:
    def test_significant_improvement(self):
        spec = MetricSpec("acc", MetricRole.PRIMARY, MetricDirection.HIGHER)
        baseline = _make_experiment("acc", [0.70, 0.71, 0.69, 0.70, 0.72])
        candidate = _make_experiment("acc", [0.85, 0.86, 0.84, 0.85, 0.87])

        result = compare_metric(baseline, candidate, spec)
        assert result.is_improvement
        assert result.is_significant
        assert result.is_meaningful

    def test_no_improvement(self):
        spec = MetricSpec("acc", MetricRole.PRIMARY, MetricDirection.HIGHER)
        baseline = _make_experiment("acc", [0.80, 0.81, 0.79, 0.80, 0.82])
        candidate = _make_experiment("acc", [0.75, 0.76, 0.74, 0.75, 0.77])

        result = compare_metric(baseline, candidate, spec)
        assert not result.is_improvement

    def test_lower_is_better(self):
        spec = MetricSpec("loss", MetricRole.PRIMARY, MetricDirection.LOWER)
        baseline = _make_experiment("loss", [1.0, 1.1, 0.9, 1.0, 1.05])
        candidate = _make_experiment("loss", [0.5, 0.51, 0.49, 0.5, 0.52])

        result = compare_metric(baseline, candidate, spec)
        assert result.is_improvement


class TestEvaluateExperiment:
    def test_keep_on_clear_improvement(self):
        specs = [
            MetricSpec("primary", MetricRole.PRIMARY, MetricDirection.HIGHER),
            MetricSpec("guard", MetricRole.GUARD, MetricDirection.HIGHER, guard_threshold=0.1),
        ]
        # Use slight variance so Cohen's d is computable
        base_vals = [0.70, 0.71, 0.69, 0.70, 0.72]
        cand_vals = [0.90, 0.91, 0.89, 0.90, 0.92]
        baseline = ExperimentResult(
            experiment_id="base", description="base",
            seed_results=[
                SeedResult(seed=i, metrics={"primary": v, "guard": 0.80 + i * 0.001})
                for i, v in enumerate(base_vals)
            ],
        )
        candidate = ExperimentResult(
            experiment_id="cand", description="cand",
            seed_results=[
                SeedResult(seed=i, metrics={"primary": v, "guard": 0.80 + i * 0.001})
                for i, v in enumerate(cand_vals)
            ],
        )

        decision = evaluate_experiment(baseline, candidate, specs)
        assert decision.keep

    def test_revert_on_guard_violation(self):
        specs = [
            MetricSpec("primary", MetricRole.PRIMARY, MetricDirection.HIGHER),
            MetricSpec("guard", MetricRole.GUARD, MetricDirection.HIGHER, guard_threshold=0.1),
        ]
        baseline = ExperimentResult(
            experiment_id="base", description="base",
            seed_results=[
                SeedResult(seed=i, metrics={"primary": 0.7, "guard": 0.8})
                for i in range(5)
            ],
        )
        candidate = ExperimentResult(
            experiment_id="cand", description="cand",
            seed_results=[
                SeedResult(seed=i, metrics={"primary": 0.9, "guard": 0.5})  # guard violated
                for i in range(5)
            ],
        )

        decision = evaluate_experiment(baseline, candidate, specs)
        assert not decision.keep
        assert len(decision.guard_violations) > 0

    def test_revert_on_too_few_seeds(self):
        specs = [MetricSpec("primary", MetricRole.PRIMARY, MetricDirection.HIGHER)]
        baseline = ExperimentResult(
            experiment_id="base", description="base",
            seed_results=[SeedResult(seed=i, metrics={"primary": 0.7}) for i in range(5)],
        )
        candidate = ExperimentResult(
            experiment_id="cand", description="cand",
            seed_results=[
                SeedResult(seed=0, metrics={"primary": 0.9}),
                SeedResult(seed=1, metrics={}, success=False),
                SeedResult(seed=2, metrics={}, success=False),
                SeedResult(seed=3, metrics={}, success=False),
                SeedResult(seed=4, metrics={}, success=False),
            ],
        )

        decision = evaluate_experiment(baseline, candidate, specs, min_seeds=3)
        assert not decision.keep

    def test_revert_on_insignificant_improvement(self):
        specs = [MetricSpec("primary", MetricRole.PRIMARY, MetricDirection.HIGHER)]
        rng = np.random.RandomState(42)
        baseline_vals = 0.7 + rng.randn(5) * 0.1
        candidate_vals = 0.71 + rng.randn(5) * 0.1  # Tiny improvement, high variance

        baseline = ExperimentResult(
            experiment_id="base", description="base",
            seed_results=[SeedResult(seed=i, metrics={"primary": v}) for i, v in enumerate(baseline_vals)],
        )
        candidate = ExperimentResult(
            experiment_id="cand", description="cand",
            seed_results=[SeedResult(seed=i, metrics={"primary": v}) for i, v in enumerate(candidate_vals)],
        )

        # Use strict thresholds to ensure revert on noisy tiny improvement
        decision = evaluate_experiment(baseline, candidate, specs, alpha=0.05, min_effect_size=0.3, paired=False)
        assert not decision.keep  # Not significant enough

    def test_keep_on_paired_consistent_improvement(self):
        """With paired testing, consistent small improvements across seeds are detectable."""
        specs = [MetricSpec("primary", MetricRole.PRIMARY, MetricDirection.HIGHER)]
        # High cross-seed variance, but each candidate is consistently 0.05 better than its paired baseline
        baseline_vals = [0.70, 0.80, 0.60, 0.75, 0.65]
        candidate_vals = [0.75, 0.85, 0.65, 0.80, 0.70]

        baseline = ExperimentResult(
            experiment_id="base", description="base",
            seed_results=[SeedResult(seed=i, metrics={"primary": v}) for i, v in enumerate(baseline_vals)],
        )
        candidate = ExperimentResult(
            experiment_id="cand", description="cand",
            seed_results=[SeedResult(seed=i, metrics={"primary": v}) for i, v in enumerate(candidate_vals)],
        )

        decision = evaluate_experiment(baseline, candidate, specs, paired=True)
        assert decision.keep  # Consistent paired improvement should be kept

    def test_revert_on_unpaired_same_data(self):
        """Same data as above but unpaired testing with strict thresholds can't detect the improvement."""
        specs = [MetricSpec("primary", MetricRole.PRIMARY, MetricDirection.HIGHER)]
        # Same high-variance data — unpaired test sees overlapping distributions
        baseline_vals = [0.70, 0.80, 0.60, 0.75, 0.65]
        candidate_vals = [0.75, 0.85, 0.65, 0.80, 0.70]

        baseline = ExperimentResult(
            experiment_id="base", description="base",
            seed_results=[SeedResult(seed=i, metrics={"primary": v}) for i, v in enumerate(baseline_vals)],
        )
        candidate = ExperimentResult(
            experiment_id="cand", description="cand",
            seed_results=[SeedResult(seed=i, metrics={"primary": v}) for i, v in enumerate(candidate_vals)],
        )

        # Unpaired with strict alpha — can't detect the small improvement buried in noise
        decision = evaluate_experiment(baseline, candidate, specs, alpha=0.05, min_effect_size=0.3, paired=False)
        assert not decision.keep  # Unpaired can't detect the improvement through the variance

    def test_guard_floor_near_zero_baseline(self):
        """When guard baseline is near zero, use absolute threshold instead of relative."""
        specs = [
            MetricSpec("primary", MetricRole.PRIMARY, MetricDirection.HIGHER),
            MetricSpec("guard", MetricRole.GUARD, MetricDirection.HIGHER, guard_threshold=0.3),
        ]
        # Guard baseline is near zero (0.017) — going to 0.0 is 100% relative degradation
        # but only 0.017 absolute change, which is below the 0.3 threshold
        baseline = ExperimentResult(
            experiment_id="base", description="base",
            seed_results=[
                SeedResult(seed=i, metrics={"primary": 0.7, "guard": 0.017})
                for i in range(5)
            ],
        )
        candidate = ExperimentResult(
            experiment_id="cand", description="cand",
            seed_results=[
                SeedResult(seed=i, metrics={"primary": 0.9, "guard": 0.0})
                for i in range(5)
            ],
        )

        decision = evaluate_experiment(baseline, candidate, specs)
        # Should NOT trigger guard violation — absolute change (0.017) < threshold (0.3)
        assert len(decision.guard_violations) == 0
        assert decision.keep  # Primary improved, guard is fine

    def test_guard_floor_real_violation(self):
        """Guard floor still catches genuinely large absolute drops."""
        specs = [
            MetricSpec("primary", MetricRole.PRIMARY, MetricDirection.HIGHER),
            MetricSpec("guard", MetricRole.GUARD, MetricDirection.HIGHER, guard_threshold=0.1),
        ]
        # Guard baseline is near zero, but candidate is much WORSE (large absolute drop)
        baseline = ExperimentResult(
            experiment_id="base", description="base",
            seed_results=[
                SeedResult(seed=i, metrics={"primary": 0.7, "guard": 0.03})
                for i in range(5)
            ],
        )
        candidate = ExperimentResult(
            experiment_id="cand", description="cand",
            seed_results=[
                SeedResult(seed=i, metrics={"primary": 0.9, "guard": -0.2})
                for i in range(5)
            ],
        )

        decision = evaluate_experiment(baseline, candidate, specs)
        # absolute change = 0.03 - (-0.2) = 0.23 > threshold 0.1 → SHOULD trigger
        assert len(decision.guard_violations) > 0
        assert not decision.keep
