"""
HIDDEN generator for the NEGATIVE-CONTROL task.

⚠️  Never shown to the agent — and deliberately so. The label is drawn INDEPENDENTLY of
    the features, so there is no learnable signal at all. The agent is told (via the schema
    doc) that this is an ordinary "predict y from X" task, and it will try in good faith.

The point is measurement: run the exact same search loop on this domain and the fraction of
iterations it "keeps" is the loop's empirical FALSE-POSITIVE RATE. Every campaign can then
report "N confirmed improvements on the real task vs an M% false-positive rate here", which
is the difference between a research engine and a noise amplifier.

Multi-world: each world_seed draws fresh independent (X, y).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

_SPLIT_SEED_OFFSET = 100_000


@dataclass
class GeneratedNullWorld:
    features: np.ndarray           # [n, n_features] — pure noise
    labels: np.ndarray             # [n] binary — INDEPENDENT of features
    train_idx: np.ndarray
    select_idx: np.ndarray
    test_idx: np.ndarray
    base_rate: float = 0.5
    split_meta: dict[str, Any] = field(default_factory=dict)


def generate_world(world_seed: int, n_samples: int = 1500, n_features: int = 50) -> GeneratedNullWorld:
    """Draw one world of independent (X, y): there is nothing to learn, by construction."""
    rng = np.random.RandomState(world_seed)

    # Structured-looking but signal-free features.
    features = rng.randn(n_samples, n_features).astype(np.float32)
    # Labels drawn independently of the features (slightly imbalanced, like real data).
    base_rate = 0.5
    labels = (rng.rand(n_samples) < base_rate).astype(np.float32)

    split_rng = np.random.RandomState(world_seed + _SPLIT_SEED_OFFSET)
    idx = split_rng.permutation(n_samples)
    n_tr, n_va = int(n_samples * 0.7), int(n_samples * 0.15)
    train_idx = np.sort(idx[:n_tr])
    select_idx = np.sort(idx[n_tr:n_tr + n_va])
    test_idx = np.sort(idx[n_tr + n_va:])

    return GeneratedNullWorld(
        features=features,
        labels=labels,
        train_idx=train_idx,
        select_idx=select_idx,
        test_idx=test_idx,
        base_rate=float(labels.mean()),
        split_meta={"n_features": n_features},
    )
