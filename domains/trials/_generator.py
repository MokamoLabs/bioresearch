"""
HIDDEN generative model for the synthetic clinical-trial task.

⚠️  Never shown to the agent. The original synthetic trials task put signal in only the
    first 5 (linear) features, so an XGBoost baseline saturated it and cross-seed AUROC
    swung from 0.50 (chance) to 0.66 — mostly noise.

This generator gives each trial a latent drug descriptor whose effect on success is
BOTH linear and nonlinear, mixed across all `n_features` observable features, and combined
with realistic phase base-rates and an enrollment effect. A tree/linear baseline captures
much of it; the nonlinear + interaction structure is the headroom.

`labels_clean` (the true success probability) is emitted for the achievable-ceiling report.
Multi-world: `world_seed` resamples the mixing, the outcome weights, and the phase/enroll
effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

D_LATENT = 8
FEATURE_NOISE = 0.25
# Realistic per-phase base success rates.
PHASE_BASE_RATE = {1: 0.65, 2: 0.35, 3: 0.60}


@dataclass
class GeneratedTrialWorld:
    drug_smiles: list[str]
    target_names: list[list[str]]
    indications: list[str]
    phases: list[int]
    enrollment: list[int]
    features: np.ndarray           # observed [n_trials, n_features] — agent-visible
    feature_names: list[str]
    labels: np.ndarray             # observed 0/1 outcomes
    labels_clean: np.ndarray       # true success probability — ceiling only, hidden
    train_idx: np.ndarray
    select_idx: np.ndarray
    test_idx: np.ndarray
    split_meta: dict[str, Any] = field(default_factory=dict)


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def generate_world(
    world_seed: int,
    n_trials: int = 1000,
    n_features: int = 100,
    d_latent: int = D_LATENT,
) -> GeneratedTrialWorld:
    """Draw one world of the synthetic trial-outcome task with genuine, learnable signal."""
    rng = np.random.RandomState(world_seed)

    # Latent drug descriptor per trial.
    z = rng.randn(n_trials, d_latent)

    # Observable features: linear mixing of the latent drug descriptor + noise.
    mix = rng.randn(n_features, d_latent) * (1.0 / np.sqrt(d_latent))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        features = (z @ mix.T + rng.randn(n_trials, n_features) * FEATURE_NOISE).astype(np.float32)

    phases = rng.choice([1, 2, 3], n_trials, p=[0.3, 0.4, 0.3]).tolist()
    enrollment = rng.randint(50, 5000, n_trials)

    # Outcome model: linear + nonlinear latent effect, phase base rate, enrollment effect.
    w = rng.randn(d_latent)
    i1, i2, i3 = rng.randint(0, d_latent, size=3)
    a, b = rng.randn() * 0.9, rng.randn() * 0.9
    enroll_z = (np.log(enrollment) - np.log(enrollment).mean()) / (np.log(enrollment).std() + 1e-8)
    enroll_coef = rng.randn() * 0.3

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        drug_effect = z @ w + a * (z[:, i1] * z[:, i2]) + b * np.tanh(z[:, i3])
    drug_effect = (drug_effect - drug_effect.mean()) / (drug_effect.std() + 1e-8)

    phase_offset = np.array([_logit(PHASE_BASE_RATE[p]) for p in phases])
    logit = phase_offset + 1.1 * drug_effect + enroll_coef * enroll_z
    prob = 1.0 / (1.0 + np.exp(-logit))

    labels = (rng.rand(n_trials) < prob).astype(np.float32)
    labels_clean = prob.astype(np.float32)

    drug_smiles = [f"DRUG_{i:05d}" for i in range(n_trials)]
    target_names = [[f"TARGET_{rng.randint(0, 50)}"] for _ in range(n_trials)]
    indications = [rng.choice(["NSCLC", "breast_cancer", "AML", "melanoma", "glioblastoma"])
                   for _ in range(n_trials)]
    feature_names = [f"feat_{i}" for i in range(n_features)]

    # Temporal split: contiguous blocks (train before val before test), as the frozen
    # evaluation contract expects. Data is i.i.d. so contiguous slicing is unbiased.
    train_idx = np.arange(0, int(n_trials * 0.6))
    select_idx = np.arange(int(n_trials * 0.6), int(n_trials * 0.8))
    test_idx = np.arange(int(n_trials * 0.8), n_trials)

    return GeneratedTrialWorld(
        drug_smiles=drug_smiles,
        target_names=target_names,
        indications=indications,
        phases=phases,
        enrollment=enrollment.tolist(),
        features=features,
        feature_names=feature_names,
        labels=labels,
        labels_clean=labels_clean,
        train_idx=train_idx,
        select_idx=select_idx,
        test_idx=test_idx,
        split_meta={"success_rate": float(labels.mean()), "d_latent": d_latent},
    )
