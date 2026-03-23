"""Tests for the core autoresearch engine."""

import os
import tempfile
import time
from unittest.mock import patch, MagicMock

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
from engine.orchestrator import OrchestratorConfig
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


class TestOrchestratorConfig:
    def test_defaults(self):
        config = OrchestratorConfig()
        assert config.backend == "anthropic"
        assert config.vertex_project_id == ""
        assert config.vertex_region == "global"
        assert config.model == "claude-opus-4-6"

    def test_vertex_config(self):
        config = OrchestratorConfig(
            backend="vertex",
            vertex_project_id="my-project",
            vertex_region="europe-west1",
        )
        assert config.backend == "vertex"
        assert config.vertex_project_id == "my-project"
        assert config.vertex_region == "europe-west1"


class TestClientFactory:
    def test_anthropic_client_with_key(self):
        from engine.orchestrator import Orchestrator
        with tempfile.TemporaryDirectory() as tmpdir:
            domain_dir = os.path.join(tmpdir, "domain")
            os.makedirs(domain_dir)
            # Create minimal train.py and program.md
            with open(os.path.join(domain_dir, "train.py"), "w") as f:
                f.write("print('hello')")
            with open(os.path.join(domain_dir, "program.md"), "w") as f:
                f.write("")

            config = OrchestratorConfig()
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-key"}):
                orch = Orchestrator(
                    config=config,
                    domain_dir=domain_dir,
                    output_dir=tmpdir,
                    metric_specs=[],
                )
                import anthropic
                assert isinstance(orch.client, anthropic.Anthropic)

    def test_anthropic_client_without_key(self):
        from engine.orchestrator import Orchestrator
        with tempfile.TemporaryDirectory() as tmpdir:
            domain_dir = os.path.join(tmpdir, "domain")
            os.makedirs(domain_dir)
            with open(os.path.join(domain_dir, "train.py"), "w") as f:
                f.write("print('hello')")

            config = OrchestratorConfig()
            env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY not set"):
                    Orchestrator(
                        config=config,
                        domain_dir=domain_dir,
                        output_dir=tmpdir,
                        metric_specs=[],
                    )

    def test_vertex_client_without_project_id(self):
        from engine.orchestrator import Orchestrator
        with tempfile.TemporaryDirectory() as tmpdir:
            domain_dir = os.path.join(tmpdir, "domain")
            os.makedirs(domain_dir)
            with open(os.path.join(domain_dir, "train.py"), "w") as f:
                f.write("print('hello')")

            config = OrchestratorConfig(backend="vertex")
            env = {k: v for k, v in os.environ.items() if k != "VERTEX_PROJECT_ID"}
            with patch.dict(os.environ, env, clear=True):
                # Mock AnthropicVertex so we don't need the dependency
                mock_vertex = MagicMock()
                with patch.dict("sys.modules", {"anthropic": MagicMock(AnthropicVertex=mock_vertex)}):
                    with pytest.raises(RuntimeError, match="Vertex AI requires a GCP project ID"):
                        Orchestrator(
                            config=config,
                            domain_dir=domain_dir,
                            output_dir=tmpdir,
                            metric_specs=[],
                        )

    def test_vertex_client_with_config(self):
        from engine.orchestrator import Orchestrator
        with tempfile.TemporaryDirectory() as tmpdir:
            domain_dir = os.path.join(tmpdir, "domain")
            os.makedirs(domain_dir)
            with open(os.path.join(domain_dir, "train.py"), "w") as f:
                f.write("print('hello')")

            config = OrchestratorConfig(
                backend="vertex",
                vertex_project_id="test-project",
                vertex_region="us-east5",
            )
            mock_vertex_cls = MagicMock()
            # Patch AnthropicVertex on the anthropic module that orchestrator imports
            import anthropic as anthropic_mod
            with patch.object(anthropic_mod, "AnthropicVertex", mock_vertex_cls, create=True):
                orch = Orchestrator(
                    config=config,
                    domain_dir=domain_dir,
                    output_dir=tmpdir,
                    metric_specs=[],
                )
                mock_vertex_cls.assert_called_once_with(
                    project_id="test-project", region="us-east5"
                )


class TestComputeRunners:
    def test_local_runner_interface(self):
        from cli import _make_local_runner
        runner = _make_local_runner()
        assert callable(runner)

    def test_local_runner_with_broken_code(self):
        from cli import _make_local_runner
        with tempfile.TemporaryDirectory() as tmpdir:
            domain_dir = os.path.join(tmpdir, "domain")
            os.makedirs(domain_dir)
            with open(os.path.join(domain_dir, "train.py"), "w") as f:
                f.write("print('placeholder')")

            runner = _make_local_runner()
            results = runner(domain_dir, "raise Exception('broken')", [0, 1], 10)
            assert len(results) == 2
            assert not results[0].success
            assert not results[1].success

    def test_prescreen_runner_skips_modal_on_failure(self):
        from cli import _make_prescreen_runner
        from engine.metrics import SeedResult

        # Mock _make_modal_runner to track if Modal is called
        modal_called = []
        def mock_modal_runner(domain_dir, train_code, seeds, time_budget):
            modal_called.append(True)
            return [SeedResult(seed=s, metrics={"acc": 0.5}) for s in seeds]

        with patch("cli._make_modal_runner", return_value=mock_modal_runner):
            runner = _make_prescreen_runner("perturbation")

            with tempfile.TemporaryDirectory() as tmpdir:
                domain_dir = os.path.join(tmpdir, "domain")
                os.makedirs(domain_dir)
                with open(os.path.join(domain_dir, "train.py"), "w") as f:
                    f.write("print('placeholder')")

                results = runner(domain_dir, "raise Exception('broken')", [0, 1, 2], 10)
                assert len(results) == 3
                assert not results[0].success  # pre-screen failed
                assert not results[1].success  # skipped
                assert not results[2].success  # skipped
                assert len(modal_called) == 0  # Modal was never called


class TestRecommendComputeMode:
    def test_high_end_gpu_with_modal(self):
        from infra.colab import recommend_compute_mode
        result = recommend_compute_mode(
            {"has_gpu": True, "gpu_name": "NVIDIA H100", "vram_gb": 80.0},
            modal_ok=True,
        )
        assert result == "hybrid"

    def test_low_end_gpu_with_modal(self):
        from infra.colab import recommend_compute_mode
        result = recommend_compute_mode(
            {"has_gpu": True, "gpu_name": "Tesla T4", "vram_gb": 16.0},
            modal_ok=True,
        )
        assert result == "prescreen"

    def test_gpu_without_modal(self):
        from infra.colab import recommend_compute_mode
        result = recommend_compute_mode(
            {"has_gpu": True, "gpu_name": "Tesla T4", "vram_gb": 16.0},
            modal_ok=False,
        )
        assert result == "local"

    def test_no_gpu_with_modal(self):
        from infra.colab import recommend_compute_mode
        result = recommend_compute_mode(
            {"has_gpu": False, "gpu_name": "none", "vram_gb": 0.0},
            modal_ok=True,
        )
        assert result == "modal"
