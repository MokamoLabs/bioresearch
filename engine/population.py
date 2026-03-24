"""
Population-based search with tournament selection.

Runs K agents in parallel, each exploring a different design direction.
Every N iterations, bottom 25% adopt top 25%'s code (tournament selection).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from engine.metrics import (
    ExperimentResult,
    SeedResult,
    MetricSpec,
    MetricRole,
    evaluate_experiment,
)
from engine.orchestrator import Orchestrator, OrchestratorConfig
from engine.tracker import ExperimentTracker, ExperimentRecord
from engine.loop import RunExperimentFn, _run_multi_seed, run_local_experiment


@dataclass
class PopulationConfig:
    num_agents: int = 4
    tournament_interval: int = 10  # every N iterations
    domain_dir: str = ""
    output_dir: str = ""
    num_seeds: int = 5
    max_iterations: int = 100
    min_seeds_for_decision: int = 3
    alpha: float = 0.10
    min_effect_size: float = 0.15
    time_budget_per_seed: int = 600
    orchestrator_config: Optional[OrchestratorConfig] = None


@dataclass
class AgentSlot:
    agent_id: int
    orchestrator: Orchestrator
    current_baseline: Optional[ExperimentResult] = None
    best_primary_score: float = float("-inf")
    iteration_count: int = 0


class PopulationSearch:
    """
    Population-based autoresearch: K agents search in parallel,
    periodically sharing top architectures via tournament selection.
    """

    def __init__(
        self,
        metric_specs: list[MetricSpec],
        config: PopulationConfig,
        run_fn: Optional[RunExperimentFn] = None,
        knowledge_fn: Optional[Callable[[], str]] = None,
        run_seeds_parallel: Optional[Callable] = None,
    ):
        self.metric_specs = metric_specs
        self.config = config
        self.run_fn = run_fn or run_local_experiment
        self.knowledge_fn = knowledge_fn
        self.run_seeds_parallel = run_seeds_parallel

        self.primary_metric = next(
            s for s in metric_specs if s.role == MetricRole.PRIMARY
        )

        orch_config = config.orchestrator_config or OrchestratorConfig()

        # Create K agent slots
        self.agents: list[AgentSlot] = []
        for i in range(config.num_agents):
            agent_output_dir = str(Path(config.output_dir) / f"agent_{i}")
            orchestrator = Orchestrator(
                config=copy.deepcopy(orch_config),
                domain_dir=config.domain_dir,
                output_dir=agent_output_dir,
                metric_specs=metric_specs,
                knowledge_fn=knowledge_fn,
            )
            self.agents.append(AgentSlot(
                agent_id=i,
                orchestrator=orchestrator,
            ))

        self.global_tracker = ExperimentTracker(
            config.output_dir, "population_global"
        )

    def run(self):
        """Run the population-based search."""
        # First: run baseline for all agents (shared)
        print("=" * 60)
        print(f"Population search: {self.config.num_agents} agents")
        print("Running shared baseline...")
        print("=" * 60)

        baseline = _run_multi_seed(
            domain_dir=self.config.domain_dir,
            train_code=self.agents[0].orchestrator.current_code,
            seeds=list(range(self.config.num_seeds)),
            time_budget=self.config.time_budget_per_seed,
            run_fn=self.run_fn,
            run_seeds_parallel=self.run_seeds_parallel,
            experiment_id="baseline",
            description="Shared baseline",
        )

        if baseline.num_successful < self.config.min_seeds_for_decision:
            raise RuntimeError(
                f"Baseline failed: only {baseline.num_successful}/{self.config.num_seeds} seeds succeeded. "
                "Fix train.py before starting population search."
            )

        baseline_score = baseline.metric_mean(self.primary_metric.name)
        print(f"Baseline {self.primary_metric.name}: {baseline_score:.6f}")

        for agent in self.agents:
            agent.current_baseline = copy.deepcopy(baseline)
            agent.best_primary_score = baseline_score

        # Main loop
        for iteration in range(self.config.max_iterations):
            print(f"\n{'=' * 60}")
            print(f"Population iteration {iteration + 1}/{self.config.max_iterations}")
            print("=" * 60)

            # Run each agent
            for agent in self.agents:
                print(f"\n--- Agent {agent.agent_id} ---")
                self._run_agent_iteration(agent)

            # Tournament selection
            if (iteration + 1) % self.config.tournament_interval == 0:
                self._tournament_selection()

        # Final summary
        print("\n" + "=" * 60)
        print("Population search complete!")
        print("=" * 60)

        for agent in self.agents:
            print(f"\nAgent {agent.agent_id}:")
            print(f"  Best {self.primary_metric.name}: {agent.best_primary_score:.6f}")
            print(f"  Iterations: {agent.iteration_count}")
            summary = agent.orchestrator.tracker.generate_summary()
            for line in summary.split("\n"):
                print(f"  {line}")

    def _run_agent_iteration(self, agent: AgentSlot):
        """Run a single iteration for one agent."""
        try:
            context = agent.orchestrator.get_context()
            hypothesis, new_code = agent.orchestrator.propose_modification(context)
            print(f"  Hypothesis: {hypothesis}")

            candidate = _run_multi_seed(
                domain_dir=self.config.domain_dir,
                train_code=new_code,
                seeds=list(range(self.config.num_seeds)),
                time_budget=self.config.time_budget_per_seed,
                run_fn=self.run_fn,
                run_seeds_parallel=self.run_seeds_parallel,
                experiment_id=f"agent{agent.agent_id}_iter{agent.iteration_count+1:04d}",
                description=hypothesis,
            )

            decision = evaluate_experiment(
                baseline=agent.current_baseline,
                candidate=candidate,
                metric_specs=self.metric_specs,
                alpha=self.config.alpha,
                min_effect_size=self.config.min_effect_size,
                min_seeds=self.config.min_seeds_for_decision,
                paired=True,
            )

            action = "KEEP" if decision.keep else "REVERT"
            print(f"  Decision: {action}")

            agent.orchestrator.apply_modification(new_code)
            agent.orchestrator.handle_decision(decision, candidate)

            if decision.keep:
                agent.current_baseline = candidate
                score = candidate.metric_mean(self.primary_metric.name)
                if self.primary_metric.is_improvement(score, agent.best_primary_score):
                    agent.best_primary_score = score

            agent.iteration_count += 1

        except Exception as e:
            print(f"  Error: {e}")
            agent.iteration_count += 1

    def _tournament_selection(self):
        """Bottom 25% adopt top 25%'s code."""
        print("\n--- Tournament Selection ---")

        # Sort agents by best score
        sorted_agents = sorted(
            self.agents,
            key=lambda a: a.best_primary_score,
            reverse=(self.primary_metric.direction.value == "higher"),
        )

        n = len(sorted_agents)
        top_k = max(1, n // 4)
        bottom_k = max(1, n // 4)

        top_agents = sorted_agents[:top_k]
        bottom_agents = sorted_agents[-bottom_k:]

        for i, bottom in enumerate(bottom_agents):
            donor = top_agents[i % len(top_agents)]
            print(
                f"  Agent {bottom.agent_id} (score={bottom.best_primary_score:.6f}) "
                f"<- Agent {donor.agent_id} (score={donor.best_primary_score:.6f})"
            )
            # Copy the donor's best code (deep copy to avoid shared references)
            donor_code = copy.deepcopy(donor.orchestrator.best_code)
            bottom.orchestrator._write_train(donor_code)
            bottom.orchestrator.best_code = donor_code
            bottom.orchestrator.baseline_code = donor_code
            bottom.current_baseline = copy.deepcopy(donor.current_baseline)
            # Don't reset best_primary_score — let them improve from the new baseline

        print("Tournament selection complete.")
