"""
Perturbation prediction model — MUTABLE.

This file is what the autoresearch agent modifies.
Starting point: linear baseline (known to match or beat deep learning methods).

The agent will modify this to discover better architectures.

Usage: python domains/perturbation/train.py
Outputs metrics as JSON on the last line of stdout.
"""

import json
import os
import sys
import time

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from domains.perturbation.prepare import load_data, evaluate, N_TOP_DEGS

SEED = int(os.environ.get("SEED", "42"))
TIME_BUDGET = int(os.environ.get("TIME_BUDGET", "600"))

np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Model: Linear baseline
# ---------------------------------------------------------------------------

class LinearPerturbModel:
    """
    Linear perturbation prediction model.

    For each seen perturbation, uses the mean delta (pert - ctrl) from training.
    For unseen perturbations, falls back to the global mean delta.

    This baseline does NOT use:
    - Perturbation features (target genes, pathway) for unseen perts
    - Expression-dependent modulation (nonlinear ctrl->effect mapping)
    - Cell-type conditioning

    These are available in the dataset and represent opportunities for improvement.
    """

    def __init__(self, n_genes: int):
        self.n_genes = n_genes
        self.pert_deltas: dict[str, np.ndarray] = {}
        self.global_mean_delta: np.ndarray | None = None

    def fit(self, ctrl_expr: np.ndarray, pert_expr: np.ndarray, pert_names: list[str]):
        """Fit the model: compute mean perturbation effect per perturbation."""
        deltas = pert_expr - ctrl_expr
        self.global_mean_delta = deltas.mean(axis=0)

        for pname in set(pert_names):
            mask = np.array([p == pname for p in pert_names])
            if mask.sum() > 0:
                self.pert_deltas[pname] = deltas[mask].mean(axis=0)

    def predict(self, ctrl_expr: np.ndarray, pert_names: list[str]) -> np.ndarray:
        """Predict post-perturbation expression."""
        predictions = np.zeros_like(ctrl_expr)
        for i, pname in enumerate(pert_names):
            delta = self.pert_deltas.get(pname, self.global_mean_delta)
            if delta is None:
                delta = np.zeros(self.n_genes)
            predictions[i] = ctrl_expr[i] + delta
        return predictions


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    # Load data
    dataset = load_data("synthetic")  # Start with synthetic; change to "norman_2019" for real data
    print(f"Dataset: {dataset.n_samples} samples, {dataset.n_genes} genes")
    print(f"Train: {len(dataset.train_idx)}, Val: {len(dataset.val_idx)}, Test: {len(dataset.test_idx)}")

    # Available but unused by baseline:
    #   dataset.pert_features  — dict: pert_name -> {"target_genes": ndarray, "pathway": int}
    #   dataset.gene_pathway   — ndarray: gene_idx -> pathway_id
    #   dataset.n_pathways     — int: number of pathways
    #   dataset.cell_types     — list: cell type per sample ("K562" or "HeLa")

    # Subsample training data for seed-controlled variance.
    # Each seed uses a 90% random subsample of training data, giving natural
    # variance across seeds while keeping the evaluation set fixed.
    rng = np.random.RandomState(SEED)
    n_subsample = int(len(dataset.train_idx) * 0.9)
    subsample_idx = rng.choice(dataset.train_idx, n_subsample, replace=False)
    train_ctrl = dataset.ctrl_expr[subsample_idx]
    train_pert = dataset.pert_expr[subsample_idx]
    train_names = [dataset.pert_names[i] for i in subsample_idx]

    model = LinearPerturbModel(n_genes=dataset.n_genes)
    model.fit(train_ctrl, train_pert, train_names)

    train_time = time.time() - t_start
    print(f"Training time: {train_time:.1f}s")

    # Evaluate on validation set
    val_ctrl = dataset.ctrl_expr[dataset.val_idx]
    val_pert = dataset.pert_expr[dataset.val_idx]
    val_names = [dataset.pert_names[i] for i in dataset.val_idx]
    val_cell_types = [dataset.cell_types[i] for i in dataset.val_idx]

    predictions = model.predict(val_ctrl, val_names)

    metrics = evaluate(
        predictions=predictions,
        ground_truth=val_pert,
        pert_names=val_names,
        deg_indices=dataset.deg_indices,
        cell_types=val_cell_types,
        ctrl_expr=val_ctrl,
    )

    metrics["train_seconds"] = train_time
    metrics["peak_vram_mb"] = 0.0  # CPU model, no VRAM

    # Print metrics (parsed by the engine)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
