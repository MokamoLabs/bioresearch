"""
Experiment logging: TSV log, metric plots, and LLM insight reports.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ExperimentRecord:
    experiment_id: str
    iteration: int
    description: str
    status: str  # "keep", "revert", "crash"
    timestamp: float
    code_diff: str = ""
    # Metric summaries (mean across seeds)
    metrics: dict[str, float] = field(default_factory=dict)
    # Per-seed raw values
    seed_metrics: dict[str, list[float]] = field(default_factory=dict)
    # Statistical test results
    p_values: dict[str, float] = field(default_factory=dict)
    effect_sizes: dict[str, float] = field(default_factory=dict)
    # Decision reason
    decision_reason: str = ""
    # Resource usage
    mean_train_seconds: float = 0.0
    mean_peak_vram_mb: float = 0.0
    num_seeds_success: int = 0
    num_seeds_total: int = 0


class ExperimentTracker:
    """Tracks all experiments in a campaign, writes TSV logs, generates plots."""

    def __init__(self, output_dir: str, campaign_name: str = "campaign"):
        self.output_dir = Path(output_dir)
        self.campaign_name = campaign_name
        self.records: list[ExperimentRecord] = []
        self._setup_dirs()

    def _setup_dirs(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        self.tsv_path = self.output_dir / f"{self.campaign_name}.tsv"
        self.json_path = self.output_dir / f"{self.campaign_name}.json"

        if not self.tsv_path.exists():
            with open(self.tsv_path, "w", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow([
                    "iteration", "experiment_id", "status", "description",
                    "decision_reason", "metrics_json", "p_values_json",
                    "effect_sizes_json", "num_seeds", "timestamp",
                ])

    def log(self, record: ExperimentRecord):
        self.records.append(record)
        self._append_tsv(record)
        self._save_json()

    def _append_tsv(self, record: ExperimentRecord):
        with open(self.tsv_path, "a", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow([
                record.iteration,
                record.experiment_id,
                record.status,
                record.description,
                record.decision_reason,
                json.dumps(record.metrics),
                json.dumps(record.p_values),
                json.dumps(record.effect_sizes),
                f"{record.num_seeds_success}/{record.num_seeds_total}",
                f"{record.timestamp:.0f}",
            ])

    def _save_json(self):
        data = [asdict(r) for r in self.records]
        with open(self.json_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load_existing(self):
        """Reload records from the JSON log file."""
        if self.json_path.exists():
            with open(self.json_path) as f:
                data = json.load(f)
            self.records = []
            for d in data:
                # Remove fields that aren't ExperimentRecord constructor args
                d.pop("seed_metrics", None)
                self.records.append(ExperimentRecord(**d))

    def get_best_record(self, metric_name: str, direction: str = "higher") -> Optional[ExperimentRecord]:
        kept = [r for r in self.records if r.status == "keep"]
        if not kept:
            return None
        if direction == "higher":
            return max(kept, key=lambda r: r.metrics.get(metric_name, float("-inf")))
        return min(kept, key=lambda r: r.metrics.get(metric_name, float("inf")))

    def plot_metric_history(self, metric_name: str, direction: str = "higher"):
        kept = [r for r in self.records if r.status == "keep" and metric_name in r.metrics]
        if not kept:
            return

        iterations = [r.iteration for r in kept]
        values = [r.metrics[metric_name] for r in kept]

        # Running best
        running_best = []
        best = float("-inf") if direction == "higher" else float("inf")
        for v in values:
            if direction == "higher":
                best = max(best, v)
            else:
                best = min(best, v)
            running_best.append(best)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(iterations, values, "o-", alpha=0.5, label=metric_name, markersize=3)
        ax.plot(iterations, running_best, "-", color="red", linewidth=2, label=f"Best {metric_name}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(metric_name)
        ax.set_title(f"{self.campaign_name}: {metric_name} over iterations")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.plots_dir / f"{metric_name}_history.png", dpi=150)
        plt.close(fig)

    def plot_all_metrics(self, metric_names: list[str], directions: Optional[dict[str, str]] = None):
        directions = directions or {}
        for name in metric_names:
            self.plot_metric_history(name, directions.get(name, "higher"))

    def plot_experiment_overview(self):
        """Plot status distribution and timing."""
        if not self.records:
            return

        statuses = [r.status for r in self.records]
        status_counts = {}
        for s in statuses:
            status_counts[s] = status_counts.get(s, 0) + 1

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Status pie chart
        ax = axes[0]
        colors = {"keep": "#2ecc71", "revert": "#e74c3c", "crash": "#95a5a6"}
        labels = list(status_counts.keys())
        sizes = list(status_counts.values())
        c = [colors.get(l, "#3498db") for l in labels]
        ax.pie(sizes, labels=labels, colors=c, autopct="%1.0f%%")
        ax.set_title("Experiment Outcomes")

        # Timing
        ax = axes[1]
        times = [r.mean_train_seconds for r in self.records if r.mean_train_seconds > 0]
        if times:
            ax.hist(times, bins=20, color="#3498db", alpha=0.7)
            ax.set_xlabel("Training time (s)")
            ax.set_ylabel("Count")
            ax.set_title("Training Time Distribution")

        fig.tight_layout()
        fig.savefig(self.plots_dir / "overview.png", dpi=150)
        plt.close(fig)

    def generate_summary(self) -> str:
        """Generate a text summary of the campaign."""
        total = len(self.records)
        kept = sum(1 for r in self.records if r.status == "keep")
        reverted = sum(1 for r in self.records if r.status == "revert")
        crashed = sum(1 for r in self.records if r.status == "crash")

        lines = [
            f"Campaign: {self.campaign_name}",
            f"Total experiments: {total}",
            f"  Kept: {kept} ({100*kept/total:.0f}%)" if total > 0 else "  Kept: 0",
            f"  Reverted: {reverted}",
            f"  Crashed: {crashed}",
        ]

        # Best metrics from kept experiments
        if kept > 0:
            kept_records = [r for r in self.records if r.status == "keep"]
            all_metric_names = set()
            for r in kept_records:
                all_metric_names.update(r.metrics.keys())
            if all_metric_names:
                lines.append("\nBest kept metrics:")
                for name in sorted(all_metric_names):
                    vals = [r.metrics[name] for r in kept_records if name in r.metrics]
                    lines.append(f"  {name}: best={max(vals):.6f}, last={vals[-1]:.6f}")

        return "\n".join(lines)
