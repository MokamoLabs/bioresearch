"""
Claude prompt builder + keep/revert state machine.

The orchestrator manages the conversation with Claude, building prompts that include:
- The current state of train.py
- Experiment history and insights
- Knowledge packets from the bio knowledge base
- Guard rail status
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import anthropic

from engine.metrics import (
    ExperimentResult,
    EvaluationDecision,
    MetricSpec,
    ComparisonResult,
)
from engine.tracker import ExperimentTracker, ExperimentRecord


class AgentState(Enum):
    INIT = "init"
    PROPOSING = "proposing"
    RUNNING = "running"
    EVALUATING = "evaluating"
    KEEPING = "keeping"
    REVERTING = "reverting"


@dataclass
class OrchestratorConfig:
    model: str = "claude-opus-4-6"
    max_tokens: int = 16384
    temperature: float = 0.7
    train_file: str = "train.py"
    program_file: str = "program.md"
    max_prompt_tokens: int = 100000  # approximate budget for the prompt
    max_history_items: int = 50
    backend: str = "anthropic"  # "anthropic" or "vertex"
    vertex_project_id: str = ""
    vertex_region: str = "global"


@dataclass
class AgentContext:
    """All the context the agent needs for its next proposal."""
    iteration: int
    current_code: str
    baseline_code: str
    metric_specs: list[MetricSpec]
    experiment_history: list[ExperimentRecord]
    last_decision: Optional[EvaluationDecision] = None
    knowledge_packet: str = ""
    agent_id: int = 0


class Orchestrator:
    """
    Manages the Claude agent that modifies train.py.

    State machine:
    INIT -> PROPOSING -> RUNNING -> EVALUATING -> KEEPING/REVERTING -> PROPOSING -> ...
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        domain_dir: str,
        output_dir: str,
        metric_specs: list[MetricSpec],
        knowledge_fn=None,
    ):
        self.config = config
        self.domain_dir = Path(domain_dir)
        self.output_dir = Path(output_dir)
        self.metric_specs = metric_specs
        self.knowledge_fn = knowledge_fn

        self.state = AgentState.INIT
        self.iteration = 0

        self.train_path = self.domain_dir / config.train_file
        self.program_path = self.domain_dir / config.program_file

        self.current_code = self._read_file(self.train_path)
        self.baseline_code = self.current_code
        self.best_code = self.current_code

        self.client = self._create_client(config)
        self.tracker = ExperimentTracker(str(self.output_dir))

    def _create_client(self, config: OrchestratorConfig):
        """Create the appropriate Anthropic client based on backend config."""
        if config.backend == "vertex":
            return self._create_vertex_client(config)
        return self._create_anthropic_client()

    def _create_anthropic_client(self):
        """Create a direct Anthropic API client."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Set it with:\n"
                "  export ANTHROPIC_API_KEY=your-key-here"
            )
        return anthropic.Anthropic()

    def _create_vertex_client(self, config: OrchestratorConfig):
        """Create an Anthropic client via Google Cloud Vertex AI."""
        try:
            from anthropic import AnthropicVertex
        except ImportError:
            raise RuntimeError(
                "Vertex AI support requires anthropic[vertex]. Install with:\n"
                "  pip install 'anthropic[vertex]'"
            )

        project_id = config.vertex_project_id or os.environ.get("VERTEX_PROJECT_ID", "")
        region = config.vertex_region or os.environ.get("VERTEX_REGION", "us-east5")

        if not project_id:
            raise RuntimeError(
                "Vertex AI requires a GCP project ID. Set it via:\n"
                "  OrchestratorConfig(vertex_project_id='my-project') or\n"
                "  export VERTEX_PROJECT_ID=my-project"
            )

        return AnthropicVertex(project_id=project_id, region=region)

    def _read_file(self, path: Path) -> str:
        return path.read_text() if path.exists() else ""

    def _write_train(self, code: str):
        self.train_path.write_text(code)
        self.current_code = code

    def build_system_prompt(self) -> str:
        program = self._read_file(self.program_path)
        metric_desc = self._format_metric_specs()
        prepare_code = self._read_file(self.domain_dir / "prepare.py")

        return f"""You are an autonomous biology ML researcher. Your job is to modify train.py to improve the model's performance on the evaluation metrics.

## Program Constraints
{program}

## Metric Specifications
{metric_desc}

## How Evaluation Works
- Each experiment runs across multiple seeds. Each seed uses the SAME dataset but a different 90% random subsample of training data.
- Your modification is compared to the baseline using a **paired one-sided t-test** (p < 0.10) and paired Cohen's d (> 0.15).
- Genuine improvements that consistently help across seeds WILL be detected and kept.
- Introduce stochastic elements (e.g., weight initialization, dropout, data augmentation) controlled by the SEED variable for robust evaluation.

## Rules
1. You ONLY modify train.py. The evaluation harness in prepare.py is FROZEN.
2. Each experiment runs for a fixed time budget across multiple seeds.
3. Your changes are kept only if they produce statistically significant improvement.
4. You must output ONLY the complete, modified train.py content between <train_py> and </train_py> tags.
5. Before the code, briefly explain your hypothesis in 1-2 sentences between <hypothesis> and </hypothesis> tags.
6. Think carefully about what might work. Consider the biological domain knowledge provided.
7. Make one focused change per iteration. Do not combine multiple unrelated ideas.
8. The code must be complete and runnable. Do not use placeholders or TODOs.
9. Preserve the output format: metrics must be printed as JSON on the last line of stdout.
10. Keep the seed-controlled training subsample logic (rng = np.random.RandomState(SEED); subsample 90% of training data). This ensures meaningful statistical evaluation.

## Exploration Strategy
- Track what you've tried. Avoid repeating the same family of approaches.
- If regularization variants haven't worked after 3 attempts, switch to a completely different direction.
- The data has gene pathway structure, expression-dependent effects, and cell-type-specific responses.
  Models that capture these patterns will outperform the linear baseline.
- Consider: MLP for nonlinear expression-delta mapping, cell-type conditioning,
  pathway-aware features, attention over genes, ensemble methods.

## Evaluation Code (READ-ONLY reference — do NOT modify prepare.py)
Study this to understand the data structure, split strategy, and how metrics are computed:
```python
{prepare_code}
```
"""

    def _format_metric_specs(self) -> str:
        lines = []
        for spec in self.metric_specs:
            role = spec.role.value.upper()
            direction = "higher is better" if spec.direction.value == "higher" else "lower is better"
            guard_info = ""
            if spec.guard_threshold is not None:
                guard_info = f", max degradation={spec.guard_threshold:.0%}"
            lines.append(f"- {spec.name} [{role}] ({direction}{guard_info})")
        return "\n".join(lines)

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 chars per token for code/English."""
        return len(text) // 4

    def _categorize_experiments(self, history: list) -> dict[str, list[str]]:
        """Group experiments by approach category based on description keywords."""
        categories: dict[str, list[str]] = {}
        for rec in history:
            desc = rec.description.lower()
            if any(w in desc for w in ["shrink", "ridge", "regulariz", "l1", "l2", "weight decay", "dropout"]):
                cat = "regularization"
            elif any(w in desc for w in ["neural", "mlp", "hidden", "layer", "deep", "nonlinear"]):
                cat = "neural_network"
            elif any(w in desc for w in ["attention", "transformer", "self-attention"]):
                cat = "attention"
            elif any(w in desc for w in ["ensemble", "bagging", "boost", "averaging"]):
                cat = "ensemble"
            elif any(w in desc for w in ["graph", "gnn", "network", "pathway"]):
                cat = "graph"
            elif any(w in desc for w in ["feature", "engineer", "augment", "cell type", "cell-type"]):
                cat = "feature_engineering"
            else:
                cat = "other"
            categories.setdefault(cat, []).append(rec.status)
        return categories

    def build_user_prompt(self, context: AgentContext) -> str:
        parts = []
        token_budget = self.config.max_prompt_tokens

        parts.append(f"## Iteration {context.iteration}")

        # Current code (always included — this is the most important context)
        code_section = f"\n## Current train.py\n```python\n{context.current_code}\n```"
        parts.append(code_section)
        token_budget -= self._estimate_tokens(code_section)

        # Knowledge packet (high priority)
        if context.knowledge_packet and token_budget > 2000:
            knowledge_section = f"\n## Biological Knowledge\n{context.knowledge_packet}"
            knowledge_tokens = self._estimate_tokens(knowledge_section)
            if knowledge_tokens < token_budget * 0.3:  # Cap at 30% of remaining budget
                parts.append(knowledge_section)
                token_budget -= knowledge_tokens

        # Last decision (high priority)
        if context.last_decision and token_budget > 1000:
            dec = context.last_decision
            dec_parts = [f"\n## Last Decision: {'KEEP' if dec.keep else 'REVERT'}"]
            dec_parts.append(f"Reason: {dec.reason}")
            for comp in dec.all_comparisons:
                dec_parts.append(
                    f"  {comp.metric_name}: {comp.baseline_mean:.6f} -> {comp.candidate_mean:.6f} "
                    f"(p={comp.p_value:.4f}, d={comp.effect_size:.3f})"
                )
            dec_section = "\n".join(dec_parts)
            parts.append(dec_section)
            token_budget -= self._estimate_tokens(dec_section)

        # Experiment history (adaptive — include as many as fit, with reasons)
        if context.experiment_history and token_budget > 500:
            history_lines = ["\n## Recent Experiment History"]
            max_items = min(self.config.max_history_items, len(context.experiment_history))
            recent = context.experiment_history[-max_items:]
            for rec in recent:
                metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in rec.metrics.items())
                reason_str = f" | Reason: {rec.decision_reason[:80]}" if rec.decision_reason else ""
                line = f"- [{rec.status}] {rec.description[:100]} | {metrics_str}{reason_str}"
                line_tokens = self._estimate_tokens(line)
                if token_budget - line_tokens < 200:
                    break
                history_lines.append(line)
                token_budget -= line_tokens
            if len(history_lines) > 1:
                parts.append("\n".join(history_lines))

        # Approach categorization and diversity nudge
        if context.experiment_history and token_budget > 500:
            categories = self._categorize_experiments(context.experiment_history)
            if categories:
                cat_lines = ["\n## Tried Approach Categories"]
                for cat, statuses in sorted(categories.items()):
                    kept = statuses.count("keep")
                    reverted = statuses.count("revert")
                    cat_lines.append(f"- {cat}: {len(statuses)} attempts ({kept} kept, {reverted} reverted)")
                all_categories = {"regularization", "neural_network", "attention", "ensemble", "graph", "feature_engineering"}
                unexplored = all_categories - set(categories.keys())
                if unexplored:
                    cat_lines.append(f"-> Consider unexplored categories: {', '.join(sorted(unexplored))}")
                cat_section = "\n".join(cat_lines)
                parts.append(cat_section)
                token_budget -= self._estimate_tokens(cat_section)

            # Diversity instruction after consecutive reverts
            recent_reverts = 0
            for rec in reversed(context.experiment_history):
                if rec.status == "revert":
                    recent_reverts += 1
                else:
                    break
            if recent_reverts >= 5:
                parts.append(
                    f"\n## IMPORTANT: Diversity Required\n"
                    f"The last {recent_reverts} experiments were all reverted. "
                    "You MUST try a fundamentally different approach. Consider:\n"
                    "- A completely different model architecture (MLP, attention, GNN)\n"
                    "- Using biological knowledge priors\n"
                    "- Novel feature engineering (gene interactions, pathway features)\n"
                    "- Conditioning on cell type metadata\n"
                    "- Ensemble methods\n"
                    "Do NOT try another variation of regularization or shrinkage."
                )

        parts.append(
            "\nNow propose your next modification to train.py. "
            "Output your hypothesis and the complete modified train.py."
        )

        return "\n".join(parts)

    def propose_modification(self, context: AgentContext) -> tuple[str, str]:
        """
        Ask Claude for a new train.py modification.
        Returns (hypothesis, new_code).
        """
        self.state = AgentState.PROPOSING

        system = self.build_system_prompt()
        user = self.build_user_prompt(context)

        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )

        text = response.content[0].text

        hypothesis = self._extract_tag(text, "hypothesis") or "No hypothesis provided"
        new_code = self._extract_tag(text, "train_py") or ""

        if not new_code.strip():
            raise ValueError(
                "Agent did not produce valid train.py code. "
                f"Response started with: {text[:300]}..."
            )

        return hypothesis, new_code

    def _extract_tag(self, text: str, tag: str) -> Optional[str]:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        end = text.find(end_tag)
        if start == -1 or end == -1:
            return None
        return text[start + len(start_tag):end].strip()

    def apply_modification(self, code: str):
        """Write the new code to train.py (only called on KEEP)."""
        self.state = AgentState.RUNNING
        self._write_train(code)

    def handle_decision(self, decision: EvaluationDecision, experiment: ExperimentResult):
        """Keep or revert based on the evaluation decision."""
        self.state = AgentState.EVALUATING

        if decision.keep:
            self.state = AgentState.KEEPING
            self.best_code = self.current_code
            self.baseline_code = self.current_code
        else:
            self.state = AgentState.REVERTING
            # Code was never written on revert (apply_modification not called)

        # Log to tracker
        record = ExperimentRecord(
            experiment_id=experiment.experiment_id,
            iteration=self.iteration,
            description=experiment.description,
            status="keep" if decision.keep else "revert",
            timestamp=time.time(),
            code_diff=experiment.code_diff,
            metrics={
                comp.metric_name: comp.candidate_mean
                for comp in decision.all_comparisons
            },
            p_values={
                comp.metric_name: comp.p_value
                for comp in decision.all_comparisons
            },
            effect_sizes={
                comp.metric_name: comp.effect_size
                for comp in decision.all_comparisons
            },
            decision_reason=decision.reason,
            mean_train_seconds=sum(r.train_seconds for r in experiment.successful_seeds) / max(1, experiment.num_successful),
            mean_peak_vram_mb=sum(r.peak_vram_mb for r in experiment.successful_seeds) / max(1, experiment.num_successful),
            num_seeds_success=experiment.num_successful,
            num_seeds_total=len(experiment.seed_results),
        )
        self.tracker.log(record)
        self.iteration += 1

    def get_context(self) -> AgentContext:
        """Build the current agent context."""
        knowledge = ""
        if self.knowledge_fn:
            try:
                knowledge = self.knowledge_fn()
            except Exception as e:
                knowledge = f"(Knowledge retrieval failed: {e})"

        last_decision = None
        if self.tracker.records:
            last_rec = self.tracker.records[-1]
            comparisons = []
            for name in last_rec.metrics:
                comparisons.append(ComparisonResult(
                    metric_name=name,
                    baseline_mean=0.0,
                    candidate_mean=last_rec.metrics[name],
                    t_statistic=0.0,
                    p_value=last_rec.p_values.get(name, 1.0),
                    effect_size=last_rec.effect_sizes.get(name, 0.0),
                    ci_low=0.0,
                    ci_high=0.0,
                    is_improvement=last_rec.status == "keep",
                    is_significant=last_rec.p_values.get(name, 1.0) < 0.05,
                    is_meaningful=last_rec.effect_sizes.get(name, 0.0) > 0.3,
                ))
            last_decision = EvaluationDecision(
                keep=last_rec.status == "keep",
                reason=last_rec.decision_reason,
                primary_comparisons=[],
                guard_violations=[],
                all_comparisons=comparisons,
            )

        return AgentContext(
            iteration=self.iteration,
            current_code=self.current_code,
            baseline_code=self.baseline_code,
            metric_specs=self.metric_specs,
            experiment_history=self.tracker.records,
            last_decision=last_decision,
            knowledge_packet=knowledge,
        )
