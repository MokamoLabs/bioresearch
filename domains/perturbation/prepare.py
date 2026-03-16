"""
FROZEN evaluation harness for perturbation prediction.

DO NOT MODIFY THIS FILE. It defines the data loading, preprocessing,
and evaluation metrics for the perturbation prediction task.

Task: Given a cell type and perturbation (gene knockout/overexpression or drug),
predict the post-perturbation gene expression profile.

Data: scPerturb (via pertpy) and Tahoe-100M (Arc Institute)
Metrics:
    PRIMARY:   pearson_deg     (Pearson r on top-20 DEGs)
    GUARD:     mse_top20_deg   (MSE on top-20 DEGs, must not degrade >10%)
    GUARD:     direction_acc   (up/down direction accuracy, must stay >0.7)
    BONUS:     cross_context   (generalization gap to unseen cell types)
    DIAG:      pearson_all, calibration
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Attempt to import bio-specific libraries, fall back gracefully
try:
    import scanpy as sc
    import anndata as ad
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False

try:
    import pertpy as pt
    HAS_PERTPY = True
except ImportError:
    HAS_PERTPY = False


# ---------------------------------------------------------------------------
# Constants (FROZEN)
# ---------------------------------------------------------------------------

SEED = int(os.environ.get("SEED", "42"))
TIME_BUDGET = int(os.environ.get("TIME_BUDGET", "600"))
DATA_DIR = os.environ.get("DATA_DIR", os.path.expanduser("~/.cache/bioresearch/perturbation"))
N_TOP_DEGS = 20
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
N_GENES = 5000  # top variable genes


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class PerturbationDataset:
    """Preprocessed perturbation dataset ready for ML."""
    # Gene expression matrices (cells x genes)
    ctrl_expr: np.ndarray       # control expression
    pert_expr: np.ndarray       # perturbed expression
    # Perturbation labels
    pert_names: list[str]       # perturbation names
    pert_types: list[str]       # "gene" or "drug"
    cell_types: list[str]       # cell type per sample
    gene_names: list[str]       # gene names (columns)
    # DEG indices per perturbation
    deg_indices: dict[str, np.ndarray]  # pert_name -> indices of top DEGs
    # Split indices
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    @property
    def n_genes(self) -> int:
        return self.ctrl_expr.shape[1]

    @property
    def n_samples(self) -> int:
        return self.ctrl_expr.shape[0]


def load_data(dataset_name: str = "norman_2019", n_genes: int = N_GENES) -> PerturbationDataset:
    """
    Load a perturbation dataset.

    Supported datasets:
    - norman_2019: CRISPRa perturbations in K562 cells (via pertpy)
    - tahoe_sample: 1M stratified subsample from Tahoe-100M
    - synthetic: Small synthetic dataset for testing
    """
    if dataset_name == "synthetic":
        return _make_synthetic_dataset(n_genes)

    if not HAS_SCANPY or not HAS_PERTPY:
        print("scanpy and pertpy required for real datasets. Falling back to synthetic.")
        return _make_synthetic_dataset(n_genes)

    cache_path = Path(DATA_DIR) / f"{dataset_name}_processed.npz"
    if cache_path.exists():
        return _load_cached(cache_path)

    if dataset_name == "norman_2019":
        return _load_norman(n_genes, cache_path)
    elif dataset_name == "tahoe_sample":
        return _load_tahoe_sample(n_genes, cache_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'norman_2019', 'tahoe_sample', or 'synthetic'.")


def _make_synthetic_dataset(n_genes: int = 100, n_perts: int = 20, n_cells_per_pert: int = 50) -> PerturbationDataset:
    """Create a small synthetic dataset for testing."""
    rng = np.random.RandomState(SEED)
    n_samples = n_perts * n_cells_per_pert

    # Generate control expression
    gene_means = rng.exponential(1.0, n_genes)
    ctrl_expr = rng.poisson(gene_means, (n_samples, n_genes)).astype(np.float32)

    # Generate perturbation effects
    pert_names_list = []
    pert_types_list = []
    cell_types_list = []
    pert_expr = np.zeros_like(ctrl_expr)
    deg_indices = {}

    for p in range(n_perts):
        start = p * n_cells_per_pert
        end = start + n_cells_per_pert
        pname = f"PERT_{p:03d}"

        # Each perturbation affects a random subset of genes
        n_affected = rng.randint(5, min(30, n_genes))
        affected = rng.choice(n_genes, n_affected, replace=False)
        effect = rng.randn(n_affected) * 2.0

        pert_expr[start:end] = ctrl_expr[start:end].copy()
        pert_expr[start:end, affected] += effect

        deg_indices[pname] = affected[:N_TOP_DEGS] if len(affected) >= N_TOP_DEGS else affected

        for _ in range(n_cells_per_pert):
            pert_names_list.append(pname)
            pert_types_list.append("gene")
            cell_types_list.append("K562" if rng.rand() > 0.3 else "HeLa")

    gene_names = [f"GENE_{i:04d}" for i in range(n_genes)]

    # Split by perturbation (not by cell)
    unique_perts = list(set(pert_names_list))
    rng.shuffle(unique_perts)
    n_train = int(len(unique_perts) * TRAIN_SPLIT)
    n_val = int(len(unique_perts) * VAL_SPLIT)
    train_perts = set(unique_perts[:n_train])
    val_perts = set(unique_perts[n_train:n_train + n_val])
    test_perts = set(unique_perts[n_train + n_val:])

    train_idx = np.array([i for i, p in enumerate(pert_names_list) if p in train_perts])
    val_idx = np.array([i for i, p in enumerate(pert_names_list) if p in val_perts])
    test_idx = np.array([i for i, p in enumerate(pert_names_list) if p in test_perts])

    return PerturbationDataset(
        ctrl_expr=ctrl_expr,
        pert_expr=pert_expr,
        pert_names=pert_names_list,
        pert_types=pert_types_list,
        cell_types=cell_types_list,
        gene_names=gene_names,
        deg_indices=deg_indices,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )


def _load_norman(n_genes: int, cache_path: Path) -> PerturbationDataset:
    """Load Norman 2019 CRISPRa dataset via pertpy."""
    print("Loading Norman 2019 dataset...")
    adata = pt.dt.norman_2019()

    # Preprocess
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_genes)
    adata = adata[:, adata.var.highly_variable].copy()

    # Separate control and perturbed
    is_ctrl = adata.obs["gene_program"].isna() | (adata.obs["gene_program"] == "ctrl")
    ctrl_mean = adata[is_ctrl].X.toarray().mean(axis=0) if hasattr(adata[is_ctrl].X, 'toarray') else adata[is_ctrl].X.mean(axis=0)

    # Build dataset
    pert_mask = ~is_ctrl
    pert_adata = adata[pert_mask]
    n_samples = pert_adata.n_obs

    ctrl_expr = np.tile(ctrl_mean, (n_samples, 1)).astype(np.float32)
    pert_expr = pert_adata.X.toarray().astype(np.float32) if hasattr(pert_adata.X, 'toarray') else pert_adata.X.astype(np.float32)

    pert_names = list(pert_adata.obs.get("perturbation", pert_adata.obs.index))
    cell_types = list(pert_adata.obs.get("cell_type", ["K562"] * n_samples))
    gene_names = list(pert_adata.var_names)

    # Compute DEGs per perturbation
    deg_indices = _compute_degs(ctrl_expr, pert_expr, pert_names, n_top=N_TOP_DEGS)

    # Split by perturbation
    unique_perts = list(set(pert_names))
    rng = np.random.RandomState(SEED)
    rng.shuffle(unique_perts)
    n_train = int(len(unique_perts) * TRAIN_SPLIT)
    n_val = int(len(unique_perts) * VAL_SPLIT)
    train_perts = set(unique_perts[:n_train])
    val_perts = set(unique_perts[n_train:n_train + n_val])

    train_idx = np.array([i for i, p in enumerate(pert_names) if p in train_perts])
    val_idx = np.array([i for i, p in enumerate(pert_names) if p in val_perts])
    test_idx = np.array([i for i, p in enumerate(pert_names) if p not in train_perts and p not in val_perts])

    dataset = PerturbationDataset(
        ctrl_expr=ctrl_expr,
        pert_expr=pert_expr,
        pert_names=pert_names,
        pert_types=["gene"] * n_samples,
        cell_types=cell_types,
        gene_names=gene_names,
        deg_indices=deg_indices,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    _save_cached(dataset, cache_path)
    return dataset


def _load_tahoe_sample(n_genes: int, cache_path: Path) -> PerturbationDataset:
    """Load a stratified subsample from Tahoe-100M."""
    print("Loading Tahoe-100M subsample...")
    print("Note: Full Tahoe-100M loading requires downloading from Arc Institute.")
    print("Falling back to synthetic dataset for now.")
    return _make_synthetic_dataset(n_genes)


def _compute_degs(
    ctrl_expr: np.ndarray,
    pert_expr: np.ndarray,
    pert_names: list[str],
    n_top: int = 20,
) -> dict[str, np.ndarray]:
    """Compute top DEGs per perturbation by mean absolute fold change."""
    deg_indices = {}
    unique_perts = set(pert_names)

    for pname in unique_perts:
        mask = np.array([p == pname for p in pert_names])
        if mask.sum() < 2:
            continue

        ctrl_mean = ctrl_expr[mask].mean(axis=0)
        pert_mean = pert_expr[mask].mean(axis=0)
        diff = np.abs(pert_mean - ctrl_mean)
        top_idx = np.argsort(diff)[-n_top:][::-1]
        deg_indices[pname] = top_idx

    return deg_indices


def _save_cached(dataset: PerturbationDataset, path: Path):
    """Save processed dataset to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ctrl_expr=dataset.ctrl_expr,
        pert_expr=dataset.pert_expr,
        pert_names=dataset.pert_names,
        pert_types=dataset.pert_types,
        cell_types=dataset.cell_types,
        gene_names=dataset.gene_names,
        train_idx=dataset.train_idx,
        val_idx=dataset.val_idx,
        test_idx=dataset.test_idx,
        deg_indices_json=json.dumps({k: v.tolist() for k, v in dataset.deg_indices.items()}),
    )
    print(f"Cached processed dataset to {path}")


def _load_cached(path: Path) -> PerturbationDataset:
    """Load cached processed dataset."""
    data = np.load(path, allow_pickle=True)
    deg_indices = {k: np.array(v) for k, v in json.loads(str(data["deg_indices_json"])).items()}
    return PerturbationDataset(
        ctrl_expr=data["ctrl_expr"],
        pert_expr=data["pert_expr"],
        pert_names=list(data["pert_names"]),
        pert_types=list(data["pert_types"]),
        cell_types=list(data["cell_types"]),
        gene_names=list(data["gene_names"]),
        deg_indices=deg_indices,
        train_idx=data["train_idx"],
        val_idx=data["val_idx"],
        test_idx=data["test_idx"],
    )


# ---------------------------------------------------------------------------
# Evaluation (FROZEN — DO NOT MODIFY)
# ---------------------------------------------------------------------------

def evaluate(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    pert_names: list[str],
    deg_indices: dict[str, np.ndarray],
    cell_types: list[str] | None = None,
    ctrl_expr: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Evaluate perturbation predictions.

    Args:
        predictions: Predicted post-perturbation expression (n_samples x n_genes)
        ground_truth: True post-perturbation expression (n_samples x n_genes)
        pert_names: Perturbation name for each sample
        deg_indices: Dict mapping perturbation name to DEG indices
        cell_types: Optional cell type labels (for cross-context eval)
        ctrl_expr: Optional control expression (for direction accuracy)

    Returns:
        Dict of metric values
    """
    metrics = {}

    # 1. Pearson correlation on top-20 DEGs (PRIMARY)
    pearson_degs = []
    for pname in set(pert_names):
        if pname not in deg_indices:
            continue
        mask = np.array([p == pname for p in pert_names])
        degs = deg_indices[pname]
        if len(degs) == 0:
            continue

        pred_mean = predictions[mask][:, degs].mean(axis=0)
        true_mean = ground_truth[mask][:, degs].mean(axis=0)

        if np.std(pred_mean) < 1e-10 or np.std(true_mean) < 1e-10:
            continue
        r = np.corrcoef(pred_mean, true_mean)[0, 1]
        if not np.isnan(r):
            pearson_degs.append(r)

    metrics["pearson_deg"] = float(np.mean(pearson_degs)) if pearson_degs else 0.0

    # 2. MSE on top-20 DEGs (GUARD)
    mse_degs = []
    for pname in set(pert_names):
        if pname not in deg_indices:
            continue
        mask = np.array([p == pname for p in pert_names])
        degs = deg_indices[pname]
        if len(degs) == 0:
            continue

        pred_mean = predictions[mask][:, degs].mean(axis=0)
        true_mean = ground_truth[mask][:, degs].mean(axis=0)
        mse = np.mean((pred_mean - true_mean) ** 2)
        mse_degs.append(mse)

    metrics["mse_top20_deg"] = float(np.mean(mse_degs)) if mse_degs else float("inf")

    # 3. Direction accuracy (GUARD) — requires control expression
    if ctrl_expr is not None:
        direction_accs = []
        for pname in set(pert_names):
            if pname not in deg_indices:
                continue
            mask = np.array([p == pname for p in pert_names])
            degs = deg_indices[pname]
            if len(degs) == 0:
                continue

            ctrl_mean = ctrl_expr[mask][:, degs].mean(axis=0)
            pred_mean = predictions[mask][:, degs].mean(axis=0)
            true_mean = ground_truth[mask][:, degs].mean(axis=0)

            true_dir = np.sign(true_mean - ctrl_mean)
            pred_dir = np.sign(pred_mean - ctrl_mean)

            nonzero = true_dir != 0
            if nonzero.sum() > 0:
                acc = (true_dir[nonzero] == pred_dir[nonzero]).mean()
                direction_accs.append(acc)

        metrics["direction_acc"] = float(np.mean(direction_accs)) if direction_accs else 0.5

    # 4. Pearson on all genes (DIAGNOSTIC)
    all_pearson = []
    for i in range(predictions.shape[0]):
        pred = predictions[i]
        true = ground_truth[i]
        if np.std(pred) < 1e-10 or np.std(true) < 1e-10:
            continue
        r = np.corrcoef(pred, true)[0, 1]
        if not np.isnan(r):
            all_pearson.append(r)
    metrics["pearson_all"] = float(np.mean(all_pearson)) if all_pearson else 0.0

    # 5. Cross-context generalization (BONUS) — requires cell type info
    if cell_types is not None:
        unique_types = list(set(cell_types))
        if len(unique_types) > 1:
            per_type_pearson = {}
            for ct in unique_types:
                ct_mask = np.array([c == ct for c in cell_types])
                ct_preds = predictions[ct_mask]
                ct_truth = ground_truth[ct_mask]
                rs = []
                for i in range(ct_preds.shape[0]):
                    if np.std(ct_preds[i]) < 1e-10 or np.std(ct_truth[i]) < 1e-10:
                        continue
                    r = np.corrcoef(ct_preds[i], ct_truth[i])[0, 1]
                    if not np.isnan(r):
                        rs.append(r)
                if rs:
                    per_type_pearson[ct] = np.mean(rs)

            if len(per_type_pearson) > 1:
                vals = list(per_type_pearson.values())
                metrics["cross_context"] = float(max(vals) - min(vals))  # smaller gap = better

    return metrics


def print_metrics(metrics: dict[str, float]):
    """Print metrics in standard format."""
    print("---")
    for key, value in sorted(metrics.items()):
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    print("Loading synthetic dataset for testing...")
    dataset = load_data("synthetic")
    print(f"Dataset: {dataset.n_samples} samples, {dataset.n_genes} genes")
    print(f"Train: {len(dataset.train_idx)}, Val: {len(dataset.val_idx)}, Test: {len(dataset.test_idx)}")

    # Test evaluation with random predictions
    rng = np.random.RandomState(42)
    predictions = dataset.ctrl_expr + rng.randn(*dataset.pert_expr.shape) * 0.1
    metrics = evaluate(
        predictions[dataset.test_idx],
        dataset.pert_expr[dataset.test_idx],
        [dataset.pert_names[i] for i in dataset.test_idx],
        dataset.deg_indices,
        cell_types=[dataset.cell_types[i] for i in dataset.test_idx],
        ctrl_expr=dataset.ctrl_expr[dataset.test_idx],
    )
    print_metrics(metrics)
