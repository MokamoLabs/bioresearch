"""
HIDDEN generative model for the synthetic ADMET task.

⚠️  Never shown to the agent. The original synthetic molecules task drew fingerprints
    and labels INDEPENDENTLY (`rng.randn` features, `rng.randint` labels), so there was
    no learnable relationship at all — `composite_admet` floated at ~0.65 on pure noise
    and every "improvement" the loop kept was a false positive.

This generator fixes that: every molecule has a latent chemical vector `z`. The observable
fingerprint is a linear mixing of `z` plus noise (so the signal is recoverable but spread
across all bits, as with real fingerprints), and each endpoint label is a function of `z`
with BOTH a linear part and a nonlinear part. A linear model recovers the linear part; the
remaining nonlinear + multi-task structure is the headroom a smarter model can capture.

Because we own the generator we also emit the noise-free label (`labels_clean`), the
Bayes-optimal prediction used to report the achievable ceiling.

Multi-world: `world_seed` resamples the mixing matrix, the per-endpoint weights, and the
train/select/test partition, so cross-seed variance measures generalization across draws.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

D_LATENT = 16          # latent chemical dimensions
FP_NOISE = 0.25        # observation noise on fingerprints
LABEL_NOISE = 0.5      # irreducible noise on regression labels
MISSING_FRAC = 0.10    # fraction of missing labels per endpoint
_SPLIT_SEED_OFFSET = 100_000


@dataclass
class GeneratedMolWorld:
    smiles: list[str]
    fingerprints: np.ndarray       # observed features [n, fp_dim] — agent-visible
    labels: np.ndarray             # observed labels [n, n_endpoints] (may contain NaN)
    labels_clean: np.ndarray       # noise-free labels — ceiling only, hidden
    train_idx: np.ndarray
    select_idx: np.ndarray
    test_idx: np.ndarray
    split_meta: dict[str, Any] = field(default_factory=dict)


def _random_split(n: int, world_seed: int):
    rng = np.random.RandomState(world_seed + _SPLIT_SEED_OFFSET)
    idx = rng.permutation(n)
    n_tr, n_va = int(n * 0.7), int(n * 0.15)
    return (
        np.sort(idx[:n_tr]),
        np.sort(idx[n_tr:n_tr + n_va]),
        np.sort(idx[n_tr + n_va:]),
    )


def generate_world(
    world_seed: int,
    endpoint_types: list[str],
    n_samples: int = 2000,
    fp_dim: int = 256,
    d_latent: int = D_LATENT,
) -> GeneratedMolWorld:
    """Draw one world of the synthetic ADMET task with genuine fingerprint->label signal."""
    rng = np.random.RandomState(world_seed)
    n_endpoints = len(endpoint_types)

    # Latent chemical descriptors per molecule.
    z = rng.randn(n_samples, d_latent).astype(np.float64)

    # Observable fingerprints: a linear encoding of the latent + observation noise.
    # Signal is recoverable (z ~ pinv(A) @ fp) but spread across all bits.
    mix = rng.randn(fp_dim, d_latent) * (1.0 / np.sqrt(d_latent))
    # numpy 1.26 + Apple Accelerate emits spurious divide/overflow/invalid warnings from
    # matmul even on finite inputs; suppress locally (outputs are verified finite).
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        fingerprints = (z @ mix.T + rng.randn(n_samples, fp_dim) * FP_NOISE).astype(np.float32)

    labels = np.zeros((n_samples, n_endpoints), dtype=np.float32)
    labels_clean = np.zeros((n_samples, n_endpoints), dtype=np.float32)

    for j, etype in enumerate(endpoint_types):
        # Linear part (shared latent factors => endpoints are correlated => multi-task helps).
        w = rng.randn(d_latent)
        # Nonlinear part: one pairwise interaction + one saturating term. This is the
        # headroom a linear baseline cannot capture.
        i1, i2, i3 = rng.randint(0, d_latent, size=3)
        a, b = rng.randn() * 0.8, rng.randn() * 0.8
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            score = z @ w + a * (z[:, i1] * z[:, i2]) + b * np.tanh(z[:, i3])
        score = (score - score.mean()) / (score.std() + 1e-8)

        if etype == "classification":
            p = 1.0 / (1.0 + np.exp(-score))
            labels_clean[:, j] = p.astype(np.float32)
            labels[:, j] = (rng.rand(n_samples) < p).astype(np.float32)
        else:
            labels_clean[:, j] = score.astype(np.float32)
            labels[:, j] = (score + rng.randn(n_samples) * LABEL_NOISE).astype(np.float32)

        # Inject missingness into the OBSERVED labels only.
        missing = rng.choice(n_samples, int(n_samples * MISSING_FRAC), replace=False)
        labels[missing, j] = np.nan

    # Placeholder SMILES (features are the fingerprints; these are only identifiers).
    smiles = [f"MOL_{i:05d}" for i in range(n_samples)]

    train_idx, select_idx, test_idx = _random_split(n_samples, world_seed)
    return GeneratedMolWorld(
        smiles=smiles,
        fingerprints=fingerprints,
        labels=labels,
        labels_clean=labels_clean,
        train_idx=train_idx,
        select_idx=select_idx,
        test_idx=test_idx,
        split_meta={"n_endpoints": n_endpoints, "d_latent": d_latent},
    )
