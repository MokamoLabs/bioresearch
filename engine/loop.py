"""
Core autoresearch loop with multi-metric evaluation and guard rails.

This is the domain-agnostic experiment loop. Domains provide:
- A train.py that the agent modifies
- A prepare.py with frozen evaluation
- Metric specifications (PRIMARY, GUARD, BONUS, DIAGNOSTIC)
- A program.md with agent constraints

The loop:
1. Asks the orchestrator (Claude) to propose a modification
2. Runs the experiment across multiple seeds (via Modal or local)
3. Evaluates against baseline using statistical tests
4. Keeps or reverts based on multi-metric guard rail logic
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from engine.metrics import (
    ExperimentResult,
    SeedResult,
    MetricSpec,
    evaluate_experiment,
)
from engine.orchestrator import Orchestrator, OrchestratorConfig
from engine.tracker import ExperimentTracker, ExperimentRecord


@dataclass
class LoopConfig:
    domain_dir: str
    output_dir: str
    num_seeds: int = 5
    max_iterations: int = 100
    min_seeds_for_decision: int = 3
    alpha: float = 0.05
    min_effect_size: float = 0.3
    time_budget_per_seed: int = 600  # seconds
    orchestrator_config: Optional[OrchestratorConfig] = None


RunExperimentFn = Callable[[str, str, int, int], SeedResult]
"""
Function signature: (domain_dir, train_code, seed, time_budget) -> SeedResult

This is the function that actually runs training. It can be:
- A local function that runs train.py directly
- A Modal remote function that dispatches to GPUs
"""


def run_local_experiment(domain_dir: str, train_code: str, seed: int, time_budget: int) -> SeedResult:
    """
    Run a single experiment locally by writing train.py and executing it.
    This is the fallback when Modal is not available.
    """
    import subprocess
    import json

    domain_path = Path(domain_dir)
    train_path = domain_path / "train.py"

    # Save original code so we can restore it after
    original_code = train_path.read_text() if train_path.exists() else ""

    # Write the modified train code
    train_path.write_text(train_code)

    env = os.environ.copy()
    env["SEED"] = str(seed)
    env["TIME_BUDGET"] = str(time_budget)

    t0 = time.time()
    try:
        # Use the same Python interpreter, not "uv run python"
        result = subprocess.run(
            [sys.executable, str(train_path)],
            capture_output=True,
            text=True,
            timeout=time_budget + 120,
            cwd=str(domain_path.parent.parent),  # project root for correct imports
            env=env,
        )

        train_seconds = time.time() - t0

        if result.returncode != 0:
            return SeedResult(
                seed=seed,
                metrics={},
                train_seconds=train_seconds,
                success=False,
                error=result.stderr[-2000:] if result.stderr else "Unknown error",
            )

        # Parse metrics from stdout (expect JSON on last line)
        metrics = _parse_metrics(result.stdout)

        if not metrics:
            return SeedResult(
                seed=seed,
                metrics={},
                train_seconds=train_seconds,
                success=False,
                error=f"No metrics found in output. Last 500 chars: {result.stdout[-500:]}",
            )

        # Extract peak VRAM if available
        peak_vram = metrics.pop("peak_vram_mb", 0.0)

        return SeedResult(
            seed=seed,
            metrics=metrics,
            train_seconds=train_seconds,
            peak_vram_mb=peak_vram,
            success=True,
        )

    except subprocess.TimeoutExpired:
        return SeedResult(
            seed=seed,
            metrics={},
            train_seconds=time.time() - t0,
            success=False,
            error="Timeout exceeded",
        )
    except Exception as e:
        return SeedResult(
            seed=seed,
            metrics={},
            train_seconds=time.time() - t0,
            success=False,
            error=str(e),
        )
    finally:
        # Restore original train.py — the orchestrator manages the canonical code
        if original_code:
            train_path.write_text(original_code)


def _parse_metrics(stdout: str) -> dict[str, float]:
    """
    Parse metrics from stdout. Supports two formats:
    1. JSON dict on a line: {"metric_name": value, ...}
    2. Key-value lines: metric_name: value (after a --- separator)
    """
    import json

    metrics = {}

    # Try JSON first (last line that looks like JSON)
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    return {k: float(v) for k, v in parsed.items() if isinstance(v, (int, float))}
            except (json.JSONDecodeError, ValueError):
                pass

    # Fall back to key-value parsing (after --- separator)
    in_results = False
    for line in stdout.splitlines():
        line = line.strip()
        if line == "---":
            in_results = True
            continue
        if in_results and ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            try:
                metrics[key] = float(val)
            except ValueError:
                pass

    return metrics


def _validate_metrics(metrics: dict[str, float], metric_specs: list[MetricSpec]) -> list[str]:
    """Check that all required metrics are present. Returns list of warnings."""
    from engine.metrics import MetricRole
    warnings = []
    for spec in metric_specs:
        if spec.role in (MetricRole.PRIMARY, MetricRole.GUARD) and spec.name not in metrics:
            warnings.append(f"Required metric '{spec.name}' missing from experiment output")
    return warnings


def autoresearch_loop(
    metric_specs: list[MetricSpec],
    loop_config: LoopConfig,
    run_fn: Optional[RunExperimentFn] = None,
    knowledge_fn: Optional[Callable[[], str]] = None,
    run_seeds_parallel: Optional[Callable[[str, str, list[int], int], list[SeedResult]]] = None,
):
    """
    Main autoresearch loop.

    Args:
        metric_specs: List of metric specifications (PRIMARY, GUARD, etc.)
        loop_config: Configuration for the loop
        run_fn: Function to run a single experiment (seed). Used if run_seeds_parallel is None.
        knowledge_fn: Optional function returning biological knowledge context
        run_seeds_parallel: Optional function to run all seeds in parallel (e.g., via Modal)
    """
    if run_fn is None:
        run_fn = run_local_experiment

    orch_config = loop_config.orchestrator_config or OrchestratorConfig()

    orchestrator = Orchestrator(
        config=orch_config,
        domain_dir=loop_config.domain_dir,
        output_dir=loop_config.output_dir,
        metric_specs=metric_specs,
        knowledge_fn=knowledge_fn,
    )

    # Run baseline first
    print("=" * 60)
    print("Running baseline experiment...")
    print("=" * 60)

    baseline = _run_multi_seed(
        domain_dir=loop_config.domain_dir,
        train_code=orchestrator.current_code,
        seeds=list(range(loop_config.num_seeds)),
        time_budget=loop_config.time_budget_per_seed,
        run_fn=run_fn,
        run_seeds_parallel=run_seeds_parallel,
        experiment_id="baseline",
        description="Baseline (unmodified train.py)",
    )

    print(f"Baseline: {baseline.num_successful}/{len(baseline.seed_results)} seeds succeeded")
    for spec in metric_specs:
        mean = baseline.metric_mean(spec.name)
        std = baseline.metric_std(spec.name)
        print(f"  {spec.name}: {mean:.6f} +/- {std:.6f}")

    if baseline.num_successful < loop_config.min_seeds_for_decision:
        raise RuntimeError(
            f"Baseline failed: only {baseline.num_successful}/{loop_config.num_seeds} seeds succeeded. "
            "Fix train.py before starting the loop."
        )

    # Validate that baseline produces the metrics we need
    if baseline.num_successful > 0:
        sample_metrics = baseline.successful_seeds[0].metrics
        warnings = _validate_metrics(sample_metrics, metric_specs)
        for w in warnings:
            print(f"WARNING: {w}")

    current_baseline = baseline

    # Log baseline
    orchestrator.tracker.log(ExperimentRecord(
        experiment_id="baseline",
        iteration=0,
        description="Baseline (unmodified train.py)",
        status="keep",
        timestamp=time.time(),
        metrics={spec.name: baseline.metric_mean(spec.name) for spec in metric_specs},
        num_seeds_success=baseline.num_successful,
        num_seeds_total=len(baseline.seed_results),
    ))
    orchestrator.iteration = 1

    # Main loop
    for iteration in range(loop_config.max_iterations):
        print(f"\n{'=' * 60}")
        print(f"Iteration {iteration + 1}/{loop_config.max_iterations}")
        print("=" * 60)

        try:
            # 1. Get context and propose modification
            context = orchestrator.get_context()
            hypothesis, new_code = orchestrator.propose_modification(context)
            print(f"Hypothesis: {hypothesis}")

            # 2. Run experiment across seeds (using new_code, not touching orchestrator state yet)
            candidate = _run_multi_seed(
                domain_dir=loop_config.domain_dir,
                train_code=new_code,
                seeds=list(range(loop_config.num_seeds)),
                time_budget=loop_config.time_budget_per_seed,
                run_fn=run_fn,
                run_seeds_parallel=run_seeds_parallel,
                experiment_id=f"iter_{iteration+1:04d}",
                description=hypothesis,
            )

            print(f"Results: {candidate.num_successful}/{len(candidate.seed_results)} seeds succeeded")
            for spec in metric_specs:
                mean = candidate.metric_mean(spec.name)
                std = candidate.metric_std(spec.name)
                b_mean = current_baseline.metric_mean(spec.name)
                print(f"  {spec.name}: {mean:.6f} +/- {std:.6f} (baseline: {b_mean:.6f})")

            # 3. Evaluate
            decision = evaluate_experiment(
                baseline=current_baseline,
                candidate=candidate,
                metric_specs=metric_specs,
                alpha=loop_config.alpha,
                min_effect_size=loop_config.min_effect_size,
                min_seeds=loop_config.min_seeds_for_decision,
            )

            # 4. Keep or revert
            action = "KEEP" if decision.keep else "REVERT"
            print(f"Decision: {action} -- {decision.reason}")

            if decision.keep:
                # Only write the new code if we're keeping it
                orchestrator.apply_modification(new_code)
                current_baseline = candidate

            orchestrator.handle_decision(decision, candidate)

            # Update plots periodically
            if (iteration + 1) % 5 == 0:
                metric_names = [s.name for s in metric_specs]
                directions = {s.name: s.direction.value for s in metric_specs}
                orchestrator.tracker.plot_all_metrics(metric_names, directions)
                orchestrator.tracker.plot_experiment_overview()

        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving state...")
            break
        except Exception as e:
            print(f"Error in iteration {iteration + 1}: {e}")
            traceback.print_exc()
            # Log crash — do NOT touch orchestrator code state
            orchestrator.tracker.log(ExperimentRecord(
                experiment_id=f"iter_{iteration+1:04d}",
                iteration=orchestrator.iteration,
                description=f"CRASH: {str(e)[:200]}",
                status="crash",
                timestamp=time.time(),
            ))
            orchestrator.iteration += 1

    # Final summary
    print("\n" + "=" * 60)
    print("Campaign complete!")
    print("=" * 60)
    summary = orchestrator.tracker.generate_summary()
    print(summary)

    # Final plots
    metric_names = [s.name for s in metric_specs]
    directions = {s.name: s.direction.value for s in metric_specs}
    orchestrator.tracker.plot_all_metrics(metric_names, directions)
    orchestrator.tracker.plot_experiment_overview()

    return orchestrator.tracker


def _run_multi_seed(
    domain_dir: str,
    train_code: str,
    seeds: list[int],
    time_budget: int,
    run_fn: RunExperimentFn,
    run_seeds_parallel: Optional[Callable] = None,
    experiment_id: str = "",
    description: str = "",
) -> ExperimentResult:
    """Run an experiment across multiple seeds, optionally in parallel."""

    if run_seeds_parallel is not None:
        seed_results = run_seeds_parallel(domain_dir, train_code, seeds, time_budget)
    else:
        seed_results = []
        for seed in seeds:
            print(f"  Running seed {seed}...", end=" ", flush=True)
            result = run_fn(domain_dir, train_code, seed, time_budget)
            status = "OK" if result.success else f"FAIL: {result.error[:80] if result.error else 'unknown'}"
            print(status)
            seed_results.append(result)

    return ExperimentResult(
        experiment_id=experiment_id,
        description=description,
        seed_results=seed_results,
        code_diff="",
    )
