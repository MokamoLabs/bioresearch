"""Tests for the validity harness in engine/loop.py: confirmation runs + locked test."""

import os
import tempfile

from engine.metrics import MetricSpec, MetricRole, MetricDirection, SeedResult


def _write_domain(tmpdir, baseline="BASELINE"):
    d = os.path.join(tmpdir, "domain")
    os.makedirs(d, exist_ok=True)
    for fname, content in (("train.py", baseline), ("program.md", "prog"), ("prepare.py", "# p")):
        with open(os.path.join(d, fname), "w") as f:
            f.write(content)
    return d


def test_locked_test_sets_and_restores_eval_split():
    from engine.loop import _run_locked_test, LoopConfig

    seen = {}

    def stub(domain_dir, train_code, seeds, time_budget):
        seen["during"] = os.environ.get("EVAL_SPLIT")
        return [SeedResult(seed=s, metrics={"score": 1.0}) for s in seeds]

    os.environ.pop("EVAL_SPLIT", None)
    cfg = LoopConfig(domain_dir="x", output_dir="y")
    res = _run_locked_test("code", [0, 1, 2], cfg, None, stub)

    assert seen["during"] == "test"                      # set during the run
    assert os.environ.get("EVAL_SPLIT") is None          # restored afterwards
    assert res.num_successful == 3


def test_confirmation_reverts_a_world_specific_win(monkeypatch):
    """A candidate that only wins on the original worlds must be reverted at confirmation."""
    from engine.loop import autoresearch_loop, LoopConfig
    from engine.orchestrator import Orchestrator, OrchestratorConfig

    specs = [MetricSpec("score", MetricRole.PRIMARY, MetricDirection.HIGHER)]

    def stub(domain_dir, train_code, seeds, time_budget):
        fresh = any(s >= 1000 for s in seeds)  # confirmation uses seeds + 1000
        if "WIN_EVERYWHERE" in train_code:
            base = 0.90
        elif "WIN_ORIGINAL_ONLY" in train_code:
            base = 0.50 if fresh else 0.90      # looks great, but only on original worlds
        else:  # baseline
            base = 0.50
        return [SeedResult(seed=s, metrics={"score": base + (s % 5) * 1e-4}) for s in seeds]

    codes = iter(["WIN_ORIGINAL_ONLY", "WIN_EVERYWHERE"])
    monkeypatch.setattr(Orchestrator, "propose_modification",
                        lambda self, ctx: ("hyp", next(codes)))

    with tempfile.TemporaryDirectory() as tmp:
        domain_dir = _write_domain(tmp)
        cfg = LoopConfig(
            domain_dir=domain_dir, output_dir=os.path.join(tmp, "out"),
            num_seeds=5, max_iterations=2, min_seeds_for_decision=3,
        )
        tracker = autoresearch_loop(specs, cfg, run_seeds_parallel=stub)

    decided = {r.experiment_id: r for r in tracker.records if r.experiment_id.startswith("iter_")}
    r1 = decided["iter_0001"]
    r2 = decided["iter_0002"]
    # Iter 1 looked like a win but must be reverted by confirmation on fresh worlds.
    assert r1.status == "revert"
    assert "confirmation" in r1.decision_reason.lower()
    # Iter 2 wins everywhere, so it survives confirmation and is kept.
    assert r2.status == "keep"


def test_confirmation_can_be_disabled(monkeypatch):
    """With confirm=False, a world-specific win is (wrongly) kept — proving confirmation matters."""
    from engine.loop import autoresearch_loop, LoopConfig
    from engine.orchestrator import Orchestrator

    specs = [MetricSpec("score", MetricRole.PRIMARY, MetricDirection.HIGHER)]

    def stub(domain_dir, train_code, seeds, time_budget):
        base = 0.90 if "WIN_ORIGINAL_ONLY" in train_code and all(s < 1000 for s in seeds) else \
               (0.90 if "WIN_ORIGINAL_ONLY" in train_code else 0.50)
        return [SeedResult(seed=s, metrics={"score": base + (s % 5) * 1e-4}) for s in seeds]

    monkeypatch.setattr(Orchestrator, "propose_modification",
                        lambda self, ctx: ("hyp", "WIN_ORIGINAL_ONLY"))

    with tempfile.TemporaryDirectory() as tmp:
        domain_dir = _write_domain(tmp)
        cfg = LoopConfig(
            domain_dir=domain_dir, output_dir=os.path.join(tmp, "out"),
            num_seeds=5, max_iterations=1, min_seeds_for_decision=3,
            confirm=False, locked_test=False,
        )
        tracker = autoresearch_loop(specs, cfg, run_seeds_parallel=stub)

    r1 = next(r for r in tracker.records if r.experiment_id == "iter_0001")
    assert r1.status == "keep"  # kept without confirmation (the winner's-curse failure mode)
