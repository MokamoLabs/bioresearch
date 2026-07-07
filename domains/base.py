"""
Shared task/data contracts for all research domains.

A *domain* defines a task: how to load data (real or synthetic), how it splits into
train / select / test, and — for synthetic tasks whose generator we own — the oracle
ceiling that bounds achievable performance.

This module encodes the invariants that make "the metric improved" trustworthy:

  * **Split integrity.** train / select / test are disjoint. `select` (a.k.a. val) is
    the ONLY split used for keep/revert decisions. `test` is touched once, on a
    committed model, on data the selection loop never saw.

  * **Generator hiding.** The agent is shown `DataAdapter.describe_schema()` (a loader
    doc), never the generative mechanism. Synthetic generators live in a separate
    module (`domains/<d>/_generator.py`) that is never injected into the prompt.

  * **Multi-world synthetic.** `DataAdapter.load(world_seed=...)` resamples the
    generative parameters per world, so generalization is tested across independent
    data realizations rather than across row-subsamples of a single fixed draw.

  * **Honest headroom.** `CeilingReport` expresses a score as the fraction of the
    achievable gap between a trivial floor and the oracle-optimal predictor, so a raw
    number like "0.35 pearson" becomes "42% of achievable headroom captured".

Domain dataset classes (e.g. `PerturbationDataset`) carry a `.splits: Splits` and,
for synthetic worlds, a `.ceiling: dict[str, CeilingReport] | None`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class Splits:
    """Disjoint train / select / test index sets over a dataset.

    `select` is used for keep/revert decisions during the search; `test` is the locked
    holdout, evaluated once on a committed model. Keeping these separate is what stops
    the loop from ratcheting on selection-set noise.
    """

    train_idx: np.ndarray
    select_idx: np.ndarray  # a.k.a. "val": used for keep/revert decisions
    test_idx: np.ndarray    # locked holdout: evaluated once, never during selection
    meta: dict[str, Any] = field(default_factory=dict)  # composition info (seen/unseen, etc.)

    def validate(self, n_samples: Optional[int] = None) -> None:
        """Raise if the three splits overlap or fall out of range. Cheap leakage guard."""
        tr = set(int(i) for i in self.train_idx)
        se = set(int(i) for i in self.select_idx)
        te = set(int(i) for i in self.test_idx)
        if tr & se:
            raise ValueError(f"train/select overlap: {len(tr & se)} shared indices")
        if tr & te:
            raise ValueError(f"train/test overlap: {len(tr & te)} shared indices")
        if se & te:
            raise ValueError(f"select/test overlap: {len(se & te)} shared indices")
        if n_samples is not None:
            allidx = tr | se | te
            if allidx and (min(allidx) < 0 or max(allidx) >= n_samples):
                raise ValueError(
                    f"split index out of range for n_samples={n_samples} "
                    f"(min={min(allidx)}, max={max(allidx)})"
                )

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": int(len(self.train_idx)),
            "select": int(len(self.select_idx)),
            "test": int(len(self.test_idx)),
        }


@dataclass(frozen=True)
class CeilingReport:
    """Achievability bounds for one metric on a synthetic task whose generator we own.

    floor:   score of a trivial predictor (e.g. global mean / random chance).
    oracle:  score of the Bayes-optimal predictor — one that knows the full generative
             mechanism but cannot see the irreducible per-sample noise. This is the best
             any model could do; the remaining gap to a perfect score is pure noise.

    `fraction_captured` is direction-agnostic: it works whether higher or lower is
    better, because `oracle - floor` carries the sign.
    """

    metric: str
    floor: float
    oracle: float
    higher_is_better: bool = True

    def fraction_captured(self, score: float) -> float:
        """Position of `score` on the floor(0.0) -> oracle(1.0) axis.

        Values <0 mean worse than the trivial floor; >1 means the model beat the oracle
        bound (usually a sign of test leakage or an over-optimistic oracle estimate).
        """
        denom = self.oracle - self.floor
        if abs(denom) < 1e-12:
            return 0.0
        return float((score - self.floor) / denom)

    def summarize(self, score: float) -> str:
        return (
            f"{self.metric}: {score:.4f}  "
            f"[floor={self.floor:.4f}, oracle={self.oracle:.4f}, "
            f"captured={self.fraction_captured(score) * 100:.0f}%]"
        )


class DataAdapter(ABC):
    """Seam between a domain's model code and its data source (synthetic or real).

    Concrete adapters:
      * `SyntheticAdapter` — draws from a hidden generator; `world_seed` picks the world.
      * `RealAdapter` — loads a published benchmark; `world_seed` is accepted but ignored
        (there is one real world) so the interface is uniform.
    """

    #: short identifier, e.g. "perturbation:synthetic" or "perturbation:norman_2019"
    name: str = "unnamed"
    #: whether this adapter is a synthetic generator (True) or real data (False)
    is_synthetic: bool = True

    @abstractmethod
    def load(self, world_seed: int = 0) -> Any:
        """Return the domain dataset for a given world.

        For synthetic adapters, `world_seed` selects an independent draw of the
        generative parameters. For real adapters it is ignored.
        """

    def describe_schema(self) -> str:
        """Agent-facing description of the data (fields, shapes, split semantics).

        This is what goes into the prompt in place of the raw data module. For synthetic
        tasks it MUST NOT reveal the generative mechanism — only the observable schema.
        """
        return (
            f"Data adapter '{self.name}' (synthetic={self.is_synthetic}). "
            "No schema description provided."
        )

    def oracle_ceiling(self, world_seed: int = 0) -> Optional[dict[str, CeilingReport]]:
        """metric -> CeilingReport for synthetic tasks whose generator we own.

        Returns None for real data (no known oracle) or when not computed.
        """
        return None
