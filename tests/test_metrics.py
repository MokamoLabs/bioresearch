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
    cohens_d,
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

        decision = evaluate_experiment(baseline, candidate, specs)
        assert not decision.keep  # Not significant enough
