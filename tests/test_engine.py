"""Tests for the core autoresearch engine."""

import os
import tempfile
import time

import numpy as np
import pytest

from engine.metrics import (
    MetricSpec,
    MetricRole,
    MetricDirection,
    ExperimentResult,
    SeedResult,
    evaluate_experiment,
)
from engine.tracker import ExperimentTracker, ExperimentRecord


class TestExperimentTracker:
    def test_log_and_retrieve(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir, "test_campaign")

            record = ExperimentRecord(
                experiment_id="exp_001",
                iteration=0,
                description="Test experiment",
                status="keep",
                timestamp=time.time(),
                metrics={"pearson_deg": 0.85, "mse_top20_deg": 0.12},
                p_values={"pearson_deg": 0.01},
                effect_sizes={"pearson_deg": 0.8},
            )
            tracker.log(record)

            assert len(tracker.records) == 1
            assert tracker.records[0].status == "keep"
            assert tracker.records[0].metrics["pearson_deg"] == 0.85

    def test_get_best_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir, "test_campaign")

            for i, val in enumerate([0.70, 0.85, 0.75, 0.90]):
                tracker.log(ExperimentRecord(
                    experiment_id=f"exp_{i:03d}",
                    iteration=i,
                    description=f"Experiment {i}",
                    status="keep",
                    timestamp=time.time(),
                    metrics={"pearson_deg": val},
                ))

            best = tracker.get_best_record("pearson_deg", "higher")
            assert best is not None
            assert best.metrics["pearson_deg"] == 0.90

    def test_tsv_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir, "test_campaign")
            tracker.log(ExperimentRecord(
                experiment_id="exp_001",
                iteration=0,
                description="Test",
                status="keep",
                timestamp=time.time(),
                metrics={"acc": 0.9},
            ))

            tsv_path = os.path.join(tmpdir, "test_campaign.tsv")
            assert os.path.exists(tsv_path)
            with open(tsv_path) as f:
                lines = f.readlines()
            assert len(lines) == 2  # header + 1 record

    def test_generate_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir, "test")
            for i in range(10):
                status = "keep" if i % 3 == 0 else "revert"
                tracker.log(ExperimentRecord(
                    experiment_id=f"exp_{i:03d}",
                    iteration=i,
                    description=f"Experiment {i}",
                    status=status,
                    timestamp=time.time(),
                    metrics={"acc": 0.7 + i * 0.01},
                ))

            summary = tracker.generate_summary()
            assert "Total experiments: 10" in summary
            assert "Kept:" in summary

    def test_plot_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir, "test")
            for i in range(5):
                tracker.log(ExperimentRecord(
                    experiment_id=f"exp_{i:03d}",
                    iteration=i,
                    description=f"Experiment {i}",
                    status="keep",
                    timestamp=time.time(),
                    metrics={"acc": 0.7 + i * 0.05},
                ))

            tracker.plot_metric_history("acc", "higher")
            assert os.path.exists(os.path.join(tmpdir, "plots", "acc_history.png"))


class TestExperimentResult:
    def test_metric_values(self):
        result = ExperimentResult(
            experiment_id="test",
            description="test",
            seed_results=[
                SeedResult(seed=0, metrics={"acc": 0.8}),
                SeedResult(seed=1, metrics={"acc": 0.85}),
                SeedResult(seed=2, metrics={"acc": 0.9}),
            ],
        )
        vals = result.metric_values("acc")
        assert len(vals) == 3
        np.testing.assert_allclose(vals, [0.8, 0.85, 0.9])

    def test_failed_seeds_excluded(self):
        result = ExperimentResult(
            experiment_id="test",
            description="test",
            seed_results=[
                SeedResult(seed=0, metrics={"acc": 0.8}),
                SeedResult(seed=1, metrics={}, success=False),
                SeedResult(seed=2, metrics={"acc": 0.9}),
            ],
        )
        assert result.num_successful == 2
        vals = result.metric_values("acc")
        assert len(vals) == 2

    def test_metric_mean_and_std(self):
        result = ExperimentResult(
            experiment_id="test",
            description="test",
            seed_results=[
                SeedResult(seed=i, metrics={"acc": v})
                for i, v in enumerate([0.8, 0.8, 0.8, 0.8, 0.8])
            ],
        )
        assert result.metric_mean("acc") == 0.8
        assert result.metric_std("acc") == 0.0


class TestMetricSpec:
    def test_higher_is_better(self):
        spec = MetricSpec("acc", MetricRole.PRIMARY, MetricDirection.HIGHER)
        assert spec.is_improvement(0.9, 0.8)
        assert not spec.is_improvement(0.7, 0.8)

    def test_lower_is_better(self):
        spec = MetricSpec("loss", MetricRole.PRIMARY, MetricDirection.LOWER)
        assert spec.is_improvement(0.5, 0.8)
        assert not spec.is_improvement(0.9, 0.8)

    def test_degradation_fraction(self):
        spec = MetricSpec("acc", MetricRole.GUARD, MetricDirection.HIGHER, guard_threshold=0.1)
        # 20% degradation
        frac = spec.degradation_fraction(0.64, 0.8)
        assert abs(frac - 0.2) < 1e-6
