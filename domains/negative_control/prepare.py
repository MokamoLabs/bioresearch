"""
FROZEN evaluation harness for the NEGATIVE-CONTROL task.

This task looks like an ordinary binary-prediction problem, but the labels are drawn
independently of the features (see the hidden _generator.py). It exists to CALIBRATE the
search loop: running the loop here measures how often it "keeps" a change that cannot
possibly be a real improvement — i.e. the false-positive rate.

A defining diagnostic: the oracle ceiling equals the floor (zero achievable headroom).
Any apparent gain is noise.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from domains.base import CeilingReport, DataAdapter, Splits
from domains.negative_control._generator import GeneratedNullWorld, generate_world

SEED = int(os.environ.get("SEED", "42"))
WORLD_SEED = int(os.environ.get("WORLD_SEED", str(SEED)))
TIME_BUDGET = int(os.environ.get("TIME_BUDGET", "600"))


@dataclass
class NullDataset:
    features: np.ndarray
    labels: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    splits: Optional[Splits] = None
    ceiling: Optional[dict] = None

    @property
    def n_samples(self) -> int:
        return self.features.shape[0]


_SCHEMA_DOC = """\
Prediction dataset schema (fields on the object returned by load_data):

  features  float32 [n_samples, n_features]  input features
  labels    float32 [n_samples]              binary target (0/1)
  train_idx / val_idx / test_idx             split indices (val_idx = selection split)

Objective (frozen, see evaluate): accuracy = fraction of correctly classified samples on
the selection split. Improve it however you can.
"""


def _dataset_from_world(world: GeneratedNullWorld) -> NullDataset:
    splits = Splits(world.train_idx, world.select_idx, world.test_idx, meta=world.split_meta)
    splits.validate(n_samples=world.features.shape[0])
    return NullDataset(
        features=world.features,
        labels=world.labels,
        train_idx=world.train_idx,
        val_idx=world.select_idx,
        test_idx=world.test_idx,
        splits=splits,
    )


def load_data(world_seed: Optional[int] = None) -> NullDataset:
    return SyntheticAdapter().load(WORLD_SEED if world_seed is None else world_seed)


def evaluate(predictions: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Accuracy from probabilistic or class predictions (thresholded at 0.5)."""
    preds = (np.asarray(predictions).ravel() >= 0.5).astype(np.float32)
    labels = np.asarray(labels).ravel()
    return {"accuracy": float((preds == labels).mean()) if len(labels) else 0.0}


class SyntheticAdapter(DataAdapter):
    """Signal-free control task. Oracle == floor (no achievable headroom)."""

    is_synthetic = True

    def __init__(self):
        self.name = "negative_control:synthetic"

    def load(self, world_seed: int = 0) -> NullDataset:
        return _dataset_from_world(generate_world(world_seed))

    def describe_schema(self) -> str:
        return _SCHEMA_DOC

    def oracle_ceiling(self, world_seed: int = 0, split: str = "select") -> dict:
        world = generate_world(world_seed)
        idx = world.select_idx if split == "select" else world.test_idx
        labels = world.labels[idx]
        base = float(world.labels[world.train_idx].mean())
        # With y independent of X, the best possible predictor IS the base-rate/majority
        # predictor — so oracle and floor are the same. Zero headroom, by design.
        majority = 1.0 if base >= 0.5 else 0.0
        acc = float((labels == majority).mean())
        report = CeilingReport(metric="accuracy", floor=acc, oracle=acc, higher_is_better=True)
        return {"accuracy": report}


def print_metrics(metrics: dict[str, float]) -> None:
    print("---")
    for k, v in sorted(metrics.items()):
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    dataset = load_data()
    print(f"NegativeControl: {dataset.n_samples} samples, {dataset.features.shape[1]} features")
    rep = SyntheticAdapter().oracle_ceiling(WORLD_SEED)["accuracy"]
    print(f"accuracy ceiling: floor={rep.floor:.4f} -> oracle={rep.oracle:.4f} "
          f"(headroom={rep.oracle - rep.floor:.4f}; should be ~0)")
