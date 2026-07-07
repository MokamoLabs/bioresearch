"""
HIDDEN generative model for the synthetic perturbation task.

⚠️  This module is NEVER shown to the agent. It owns the data-generating mechanism,
    which is exactly what the agent must *discover* rather than transcribe. Keeping it
    out of `prepare.py` (the file embedded in the prompt) is what turns the perturbation
    task from an answer-leaked inversion game into genuine research.

Because we own the generator, we can also emit the *noise-free* post-perturbation
expression (`pert_expr_oracle`). That is the Bayes-optimal prediction — a model that
knew the full mechanism but not the irreducible per-cell noise — and it defines the
achievable ceiling reported by `prepare.SyntheticAdapter.oracle_ceiling`.

Multi-world design
------------------
`generate_world(world_seed)` draws an independent realization of *every* generative
parameter (pathway assignments, gene means, per-perturbation targets and effects, the
train/select/test partition). Different `world_seed`s are different biological "worlds".
The search loop runs one world per experiment seed, so cross-seed variance measures
generalization across data realizations — not resampling of a single fixed draw.

The four complexity layers (kept from the original design, now hidden):
  1. gene pathway structure (correlated genes, secondary effects)
  2. expression-dependent modulation (nonlinear ctrl -> effect mapping)
  3. cell-type-specific response scaling (K562 vs HeLa)
  4. shared pathway-level effect components
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Split composition (fractions of perturbations).
FRAC_TRAIN_ONLY = 0.5   # all cells in train
FRAC_SEEN_SPLIT = 0.2   # cells divided 70/15/15 across train/select/test
# remaining 0.3 -> unseen perturbations (all cells in select or test), features provided

# Cell-type response scaling (layer 3).
CELL_TYPE_SCALE = {"K562": 1.0, "HeLa": 0.6}

# Offset applied to world_seed for the split RNG so the partition is decorrelated
# from the data-generating RNG stream but still deterministic per world.
_SPLIT_SEED_OFFSET = 100_000


@dataclass
class GeneratedWorld:
    """One realization of the synthetic perturbation universe.

    `pert_expr` is what the agent observes (includes per-cell noise). `pert_expr_oracle`
    is the deterministic, noise-free signal — used ONLY to compute the achievable
    ceiling, never exposed to the agent.
    """

    ctrl_expr: np.ndarray
    pert_expr: np.ndarray          # observed (with per-cell noise) — agent-visible
    pert_expr_oracle: np.ndarray   # deterministic (noise-free) — ceiling only, hidden
    pert_names: list[str]
    pert_types: list[str]
    cell_types: list[str]
    gene_names: list[str]
    pert_features: dict[str, dict]
    gene_pathway: np.ndarray
    n_pathways: int
    train_idx: np.ndarray
    select_idx: np.ndarray
    test_idx: np.ndarray
    split_meta: dict[str, Any] = field(default_factory=dict)


def _hybrid_split(pert_names: list[str], world_seed: int):
    """Partition cells into train / select / test using the hybrid strategy.

    50% of perturbations are train-only, 20% are seen-split (their cells divided across
    all three sets), 30% are unseen (cells only in select or test). Returns index arrays
    plus a composition summary.
    """
    rng = np.random.RandomState(world_seed + _SPLIT_SEED_OFFSET)
    unique_perts = sorted(set(pert_names))
    rng.shuffle(unique_perts)

    n_train_only = int(len(unique_perts) * FRAC_TRAIN_ONLY)
    n_seen_split = int(len(unique_perts) * FRAC_SEEN_SPLIT)

    train_only = set(unique_perts[:n_train_only])
    seen_split = list(unique_perts[n_train_only:n_train_only + n_seen_split])
    unseen = unique_perts[n_train_only + n_seen_split:]
    n_val_unseen = len(unseen) // 2
    val_unseen = set(unseen[:n_val_unseen])
    test_unseen = set(unseen[n_val_unseen:])

    train_idx, select_idx, test_idx = [], [], []

    for i, pname in enumerate(pert_names):
        if pname in train_only:
            train_idx.append(i)

    for pname in seen_split:
        cell_idx = [i for i, p in enumerate(pert_names) if p == pname]
        rng.shuffle(cell_idx)
        n = len(cell_idx)
        n_tr, n_va = int(n * 0.7), int(n * 0.15)
        train_idx.extend(cell_idx[:n_tr])
        select_idx.extend(cell_idx[n_tr:n_tr + n_va])
        test_idx.extend(cell_idx[n_tr + n_va:])

    for i, pname in enumerate(pert_names):
        if pname in val_unseen:
            select_idx.append(i)
        elif pname in test_unseen:
            test_idx.append(i)

    meta = {
        "n_train_only_perts": len(train_only),
        "n_seen_split_perts": len(seen_split),
        "n_unseen_select_perts": len(val_unseen),
        "n_unseen_test_perts": len(test_unseen),
    }
    return (
        np.array(sorted(train_idx)),
        np.array(sorted(select_idx)),
        np.array(sorted(test_idx)),
        meta,
    )


def generate_world(
    world_seed: int,
    n_genes: int = 5000,
    n_perts: int = 20,
    n_cells_per_pert: int = 50,
) -> GeneratedWorld:
    """Draw one independent world of the synthetic perturbation task."""
    rng = np.random.RandomState(world_seed)
    n_samples = n_perts * n_cells_per_pert

    # --- Layer 1: pathway structure ---
    n_pathways = min(8, max(2, n_genes // 12))
    gene_pathway = np.zeros(n_genes, dtype=int)
    for i, gene_idx in enumerate(rng.permutation(n_genes)):
        gene_pathway[gene_idx] = i % n_pathways

    pathway_shared_effects = {pw: rng.randn(n_genes) * 0.5 for pw in range(n_pathways)}

    # Control expression with pathway-correlated baseline offsets.
    gene_means = rng.exponential(1.5, n_genes)
    pathway_offsets = rng.randn(n_pathways) * 0.5
    for g in range(n_genes):
        gene_means[g] += abs(pathway_offsets[gene_pathway[g]])
    ctrl_expr = rng.poisson(np.maximum(gene_means, 0.1), (n_samples, n_genes)).astype(np.float32)

    # Cell-type assignment per cell (layer 3).
    cell_types = []
    for _p in range(n_perts):
        for _c in range(n_cells_per_pert):
            cell_types.append("K562" if rng.rand() > 0.3 else "HeLa")

    pert_names: list[str] = []
    pert_types: list[str] = []
    pert_expr = ctrl_expr.copy()
    pert_expr_oracle = ctrl_expr.copy()
    pert_features: dict[str, dict] = {}

    for p in range(n_perts):
        start, end = p * n_cells_per_pert, (p + 1) * n_cells_per_pert
        pname = f"PERT_{p:03d}"

        # Direct targets (layer 1) and their effects.
        n_primary = rng.randint(3, min(15, n_genes))
        primary_genes = rng.choice(n_genes, n_primary, replace=False)
        primary_effect = rng.randn(n_primary) * 2.0
        primary_set = set(primary_genes.tolist())

        # Secondary targets: same-pathway genes, propagated at 0.3x (layer 1).
        primary_pathways = set(gene_pathway[g] for g in primary_genes)
        secondary_genes = np.array(
            [g for g in range(n_genes)
             if g not in primary_set and gene_pathway[g] in primary_pathways],
            dtype=int,
        )
        secondary_effect = np.zeros(len(secondary_genes))
        for sg_idx, sg in enumerate(secondary_genes):
            pw = gene_pathway[sg]
            in_pw = [gene_pathway[pg] == pw for pg in primary_genes]
            if any(in_pw):
                secondary_effect[sg_idx] = np.mean(primary_effect[in_pw]) * 0.3

        # Shared pathway-level component (layer 4).
        primary_pw = gene_pathway[primary_genes[0]]
        shared_primary = pathway_shared_effects[primary_pw][primary_genes] * 0.4
        shared_secondary = (
            pathway_shared_effects[primary_pw][secondary_genes] * 0.2
            if len(secondary_genes) > 0 else np.zeros(0)
        )

        # Features exposed for ALL perturbations (the key to generalizing to unseen perts).
        pert_features[pname] = {
            "target_genes": primary_genes.copy(),
            "pathway": int(primary_pw),
        }

        for cell_idx in range(start, end):
            ct_scale = CELL_TYPE_SCALE[cell_types[cell_idx]]

            # Expression-dependent modulation (layer 2) for primary genes.
            ctrl_vals = ctrl_expr[cell_idx, primary_genes]
            modulation = 1.0 + 0.5 * np.tanh(
                (ctrl_vals - gene_means[primary_genes]) / (gene_means[primary_genes] + 1e-6)
            )
            total_primary = (primary_effect + shared_primary) * modulation * ct_scale
            pert_expr[cell_idx, primary_genes] += total_primary
            pert_expr_oracle[cell_idx, primary_genes] += total_primary

            if len(secondary_genes) > 0:
                ctrl_sec = ctrl_expr[cell_idx, secondary_genes]
                mod_sec = 1.0 + 0.3 * np.tanh(
                    (ctrl_sec - gene_means[secondary_genes]) / (gene_means[secondary_genes] + 1e-6)
                )
                total_secondary = (secondary_effect + shared_secondary) * mod_sec * ct_scale
                pert_expr[cell_idx, secondary_genes] += total_secondary
                pert_expr_oracle[cell_idx, secondary_genes] += total_secondary

            # Irreducible per-cell noise — added ONLY to the observed expression.
            affected = (
                np.concatenate([primary_genes, secondary_genes])
                if len(secondary_genes) > 0 else primary_genes
            )
            pert_expr[cell_idx, affected] += rng.randn(len(affected)) * 0.3

        pert_names.extend([pname] * n_cells_per_pert)
        pert_types.extend(["gene"] * n_cells_per_pert)

    gene_names = [f"GENE_{i:04d}" for i in range(n_genes)]
    train_idx, select_idx, test_idx, split_meta = _hybrid_split(pert_names, world_seed)

    return GeneratedWorld(
        ctrl_expr=ctrl_expr,
        pert_expr=pert_expr,
        pert_expr_oracle=pert_expr_oracle,
        pert_names=pert_names,
        pert_types=pert_types,
        cell_types=cell_types,
        gene_names=gene_names,
        pert_features=pert_features,
        gene_pathway=gene_pathway,
        n_pathways=n_pathways,
        train_idx=train_idx,
        select_idx=select_idx,
        test_idx=test_idx,
        split_meta=split_meta,
    )
