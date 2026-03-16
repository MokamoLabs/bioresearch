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

    For each perturbation, learns a linear map from control expression
    to perturbation effect (delta = pert - ctrl).

    This is the baseline that deep learning hasn't beaten (Nature Methods 2025).
    """

    def __init__(self, n_genes: int, reg_strength: float = 1.0):
        self.n_genes = n_genes
        self.reg_strength = reg_strength
        self.pert_embeddings: dict[str, np.ndarray] = {}  # pert_name -> mean delta
        self.global_mean_delta: np.ndarray | None = None

    def fit(self, ctrl_expr: np.ndarray, pert_expr: np.ndarray, pert_names: list[str]):
        """Fit the model: compute mean perturbation effect per perturbation."""
        deltas = pert_expr - ctrl_expr
        self.global_mean_delta = deltas.mean(axis=0)

        for pname in set(pert_names):
            mask = np.array([p == pname for p in pert_names])
            if mask.sum() > 0:
                self.pert_embeddings[pname] = deltas[mask].mean(axis=0)

    def predict(self, ctrl_expr: np.ndarray, pert_names: list[str]) -> np.ndarray:
        """Predict post-perturbation expression."""
        predictions = np.zeros_like(ctrl_expr)
        for i, pname in enumerate(pert_names):
            delta = self.pert_embeddings.get(pname, self.global_mean_delta)
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

    # Fit model on training data
    train_ctrl = dataset.ctrl_expr[dataset.train_idx]
    train_pert = dataset.pert_expr[dataset.train_idx]
    train_names = [dataset.pert_names[i] for i in dataset.train_idx]

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
