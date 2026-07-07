"""
FROZEN evaluation harness for perturbation prediction.

DO NOT MODIFY THIS FILE. It defines the data loading, preprocessing,
and evaluation metrics for the perturbation prediction task.

Task: Given a cell type and perturbation (gene knockout/overexpression or drug),
predict the post-perturbation gene expression profile.

Data: scPerturb (via pertpy) and Tahoe-100M (Arc Institute)
Metrics:
    PRIMARY:   pearson_deg     (per-cell Pearson r on top-20 DEGs)
    GUARD:     mse_top20_deg   (per-cell MSE on top-20 DEGs, must not degrade >10%)
    GUARD:     direction_acc   (up/down direction accuracy, must stay >0.7)
    BONUS:     cross_context   (generalization gap across cell types)
    DIAG:      pearson_all, calibration
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from domains.base import CeilingReport, DataAdapter, Splits
from domains.perturbation._generator import GeneratedWorld, generate_world, _hybrid_split

# Attempt to import single-cell libraries, fall back gracefully. We catch broad
# Exceptions (not just ImportError) because some optional stacks import but raise at
# import time under version skew (e.g. pertpy -> jax vs numpy). The real Norman loader
# needs only scanpy + anndata and downloads the h5ad directly (no pertpy).
try:
    import scanpy as sc
    import anndata as ad
    HAS_SCANPY = True
except Exception:
    HAS_SCANPY = False


# ---------------------------------------------------------------------------
# Constants (FROZEN)
# ---------------------------------------------------------------------------

SEED = int(os.environ.get("SEED", "42"))
# The synthetic task is MULTI-WORLD: each experiment SEED selects an independent data
# realization (see the hidden domains/perturbation/_generator.py). There is no single
# fixed data seed anymore — cross-seed variance now measures generalization across
# worlds, not resampling of one fixed draw. WORLD_SEED can override for diagnostics.
WORLD_SEED = int(os.environ.get("WORLD_SEED", str(SEED)))
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
    # Perturbation features for generalization
    pert_features: dict[str, dict] = field(default_factory=dict)
    # pert_name -> {"target_genes": ndarray, "pathway": int}
    gene_pathway: np.ndarray = field(default_factory=lambda: np.array([]))
    # gene_idx -> pathway_id
    n_pathways: int = 0
    # Validity-harness additions (populated for synthetic worlds):
    splits: Optional[Splits] = None        # explicit train/select/test contract
    ceiling: Optional[dict] = None         # metric -> CeilingReport (achievable headroom)

    @property
    def n_genes(self) -> int:
        return self.ctrl_expr.shape[1]

    @property
    def n_samples(self) -> int:
        return self.ctrl_expr.shape[0]


def load_data(
    dataset_name: str = "synthetic",
    n_genes: int = N_GENES,
    world_seed: Optional[int] = None,
) -> PerturbationDataset:
    """
    Load a perturbation dataset via the appropriate DataAdapter.

    Supported datasets:
    - synthetic:   multi-world synthetic task (default). `world_seed` selects the world;
                   it defaults to the experiment SEED so each loop seed is a fresh world.
    - norman_2019: CRISPRa perturbations in K562 cells (via pertpy) — real data (Phase 3).
    - tahoe_sample: stratified subsample from Tahoe-100M — real data (Phase 3).
    """
    ws = WORLD_SEED if world_seed is None else world_seed

    if dataset_name == "synthetic":
        return SyntheticAdapter(n_genes=n_genes).load(ws)

    if not HAS_SCANPY:
        print("scanpy + anndata required for real datasets. Falling back to synthetic.")
        return SyntheticAdapter(n_genes=n_genes).load(ws)

    cache_path = Path(DATA_DIR) / f"{dataset_name}_processed.npz"
    if cache_path.exists():
        return _load_cached(cache_path)

    if dataset_name in ("norman_2019", "tahoe_sample"):
        return RealAdapter(dataset_name=dataset_name, n_genes=n_genes).load(ws)
    raise ValueError(
        f"Unknown dataset: {dataset_name}. Use 'synthetic', 'norman_2019', or 'tahoe_sample'."
    )


# Agent-facing schema description. This is what the prompt should show INSTEAD of the
# data-generating code — it describes the observable data and the objective, but NOT the
# generative mechanism (which lives, hidden, in _generator.py). Keeping the mechanism out
# is what makes this a discovery task rather than a transcription task.
_SCHEMA_DOC = """\
Perturbation dataset schema (fields available on the object returned by load_data):

  ctrl_expr     float32 [n_cells, n_genes]  control (pre-perturbation) expression
  pert_expr     float32 [n_cells, n_genes]  observed post-perturbation expression (TARGET)
  pert_names    list[str]                   perturbation id per cell (e.g. "PERT_003")
  cell_types    list[str]                   "K562" or "HeLa" per cell
  pert_features dict[pert -> {target_genes: int[], pathway: int}]
                                            provided for ALL perturbations, including
                                            UNSEEN ones — this is how you generalize
  gene_pathway  int[n_genes]                pathway id per gene
  deg_indices   dict[pert -> int[20]]       top-20 differentially expressed genes per pert
  train_idx / val_idx / test_idx            cell indices per split (val_idx == select split)

Splits are hybrid: some perturbations are train-only, some are 'seen' (their cells are
divided across train/val/test), and some are UNSEEN (their cells appear only in val/test).
Unseen perturbations still expose target_genes and pathway.

Objective (frozen, see `evaluate`): pearson_deg = per-cell Pearson correlation between the
PREDICTED delta (pred - ctrl) and the TRUE delta (truth - ctrl) over each perturbation's
top-20 DEGs. You are scored on capturing the *pattern of change per cell*, not on the
baseline expression. How control level, cell type, target genes and pathway structure map
to the delta is for your model to learn from the training split — it is not disclosed here.
"""


def _dataset_from_world(world: GeneratedWorld) -> PerturbationDataset:
    """Assemble the agent-visible dataset from a generated world.

    DEGs are computed from the OBSERVED expression (part of the frozen eval definition).
    The world's noise-free oracle signal is not carried into the dataset; it is used only
    by SyntheticAdapter.oracle_ceiling to compute achievable headroom.
    """
    deg_indices = _compute_degs(world.ctrl_expr, world.pert_expr, world.pert_names, n_top=N_TOP_DEGS)
    splits = Splits(world.train_idx, world.select_idx, world.test_idx, meta=world.split_meta)
    splits.validate(n_samples=world.ctrl_expr.shape[0])
    return PerturbationDataset(
        ctrl_expr=world.ctrl_expr,
        pert_expr=world.pert_expr,
        pert_names=world.pert_names,
        pert_types=world.pert_types,
        cell_types=world.cell_types,
        gene_names=world.gene_names,
        deg_indices=deg_indices,
        train_idx=world.train_idx,
        val_idx=world.select_idx,   # `val_idx` retained as an alias for the select split
        test_idx=world.test_idx,
        pert_features=world.pert_features,
        gene_pathway=world.gene_pathway,
        n_pathways=world.n_pathways,
        splits=splits,
    )


def _make_synthetic_dataset(
    n_genes: int = N_GENES, world_seed: Optional[int] = None, **_legacy
) -> PerturbationDataset:
    """Build a synthetic dataset for one world (thin wrapper over the hidden generator).

    `world_seed` selects the data realization; it defaults to WORLD_SEED (the experiment
    SEED), so each of the loop's seeds trains/evaluates on an independent world.
    """
    ws = WORLD_SEED if world_seed is None else world_seed
    return _dataset_from_world(generate_world(ws, n_genes=n_genes))


class SyntheticAdapter(DataAdapter):
    """Multi-world synthetic perturbation data with a computable oracle ceiling."""

    is_synthetic = True

    def __init__(self, n_genes: int = N_GENES):
        self.n_genes = n_genes
        self.name = "perturbation:synthetic"

    def load(self, world_seed: int = 0) -> PerturbationDataset:
        return _make_synthetic_dataset(n_genes=self.n_genes, world_seed=world_seed)

    def describe_schema(self) -> str:
        return _SCHEMA_DOC

    def oracle_ceiling(self, world_seed: int = 0, split: str = "select") -> dict:
        """Return metric -> CeilingReport (floor and Bayes-optimal oracle) for one world.

        We regenerate the world to access its hidden noise-free signal, then score the
        oracle predictor (noise-free expression) and a trivial floor predictor
        (global mean delta) through the frozen `evaluate`.
        """
        world = generate_world(world_seed, n_genes=self.n_genes)
        deg_indices = _compute_degs(world.ctrl_expr, world.pert_expr, world.pert_names, n_top=N_TOP_DEGS)
        idx = world.select_idx if split == "select" else world.test_idx

        ctrl = world.ctrl_expr[idx]
        truth = world.pert_expr[idx]
        names = [world.pert_names[i] for i in idx]
        cts = [world.cell_types[i] for i in idx]

        # Oracle: the deterministic (noise-free) post-perturbation expression.
        m_oracle = evaluate(world.pert_expr_oracle[idx], truth, names, deg_indices,
                            cell_types=cts, ctrl_expr=ctrl)
        # Floor: predict the global mean delta (from train) for every cell.
        global_delta = (
            world.pert_expr[world.train_idx] - world.ctrl_expr[world.train_idx]
        ).mean(axis=0)
        m_floor = evaluate(ctrl + global_delta, truth, names, deg_indices,
                           cell_types=cts, ctrl_expr=ctrl)

        reports = {}
        for metric, higher in (("pearson_deg", True), ("direction_acc", True),
                               ("mse_top20_deg", False)):
            if metric in m_oracle and metric in m_floor:
                reports[metric] = CeilingReport(
                    metric=metric,
                    floor=float(m_floor[metric]),
                    oracle=float(m_oracle[metric]),
                    higher_is_better=higher,
                )
        return reports


class RealAdapter(DataAdapter):
    """Real perturbation benchmarks (Norman 2019 / Replogle). Fully wired in Phase 3."""

    is_synthetic = False

    def __init__(self, dataset_name: str = "norman_2019", n_genes: int = N_GENES):
        self.dataset_name = dataset_name
        self.n_genes = n_genes
        self.name = f"perturbation:{dataset_name}"

    def load(self, world_seed: int = 0) -> PerturbationDataset:
        # Real data is a single world; world_seed is accepted for a uniform interface.
        cache_path = Path(DATA_DIR) / f"{self.dataset_name}_processed.npz"
        if cache_path.exists():
            return _load_cached(cache_path)
        if self.dataset_name == "norman_2019":
            return _load_norman(self.n_genes, cache_path)
        if self.dataset_name == "tahoe_sample":
            return _load_tahoe_sample(self.n_genes, cache_path)
        raise ValueError(f"Unknown real dataset: {self.dataset_name}")


NORMAN_URL = "https://exampledata.scverse.org/pertpy/norman_2019.h5ad"


def _download_h5ad(url: str, dest: Path):
    """Download an .h5ad with a browser UA (the scverse host 403s the default urllib UA)."""
    import shutil
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def _load_norman(n_genes: int, cache_path: Path) -> PerturbationDataset:
    """Load the REAL Norman 2019 CRISPRa dataset (K562) directly from its h5ad.

    pertpy is deliberately NOT imported (its jax dependency conflicts with numpy 1.x); we
    download the same h5ad pertpy would and read it with anndata. The result matches the
    synthetic contract — a hybrid seen/unseen perturbation split, and REAL per-perturbation
    target genes taken from the CRISPR guide identities as pert_features (the feature a model
    needs to generalize to unseen perturbations).
    """
    import anndata as ad
    from collections import defaultdict

    h5 = Path(DATA_DIR) / "norman_2019.h5ad"
    if not h5.exists():
        _download_h5ad(NORMAN_URL, h5)
    print(f"Loading Norman 2019 from {h5} ...")
    adata = ad.read_h5ad(h5)

    # Expression is already log-normalized; select highly-variable genes for tractability.
    sc.pp.highly_variable_genes(adata, n_top_genes=min(n_genes, adata.n_vars))
    adata = adata[:, adata.var.highly_variable].copy()
    genes = list(map(str, adata.var_names))
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    obs = adata.obs
    pert_all = obs["perturbation_name"].astype(str).values
    guide_all = obs["guide_ids"].astype(str).values
    X = adata.X
    is_ctrl = pert_all == "control"

    # Control baseline: mean control expression (sparse-friendly), tiled per perturbed cell.
    ctrl_mean = np.asarray(X[is_ctrl].mean(axis=0)).ravel().astype(np.float32)

    # Cap perturbed cells per perturbation for tractability; drop perts with too few cells.
    rng = np.random.RandomState(0)
    by_pert = defaultdict(list)
    for i in np.where(~is_ctrl)[0]:
        by_pert[pert_all[i]].append(int(i))
    CAP = 50
    keep: list[int] = []
    for _p, cells in by_pert.items():
        if len(cells) < 10:
            continue
        rng.shuffle(cells)
        keep.extend(cells[:CAP])
    keep = np.array(sorted(keep))

    sub = X[keep]
    pert_expr = (sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub)).astype(np.float32)
    n_samples = len(keep)
    ctrl_expr = np.tile(ctrl_mean, (n_samples, 1)).astype(np.float32)
    pert_names = [pert_all[i] for i in keep]
    cell_types = ["K562"] * n_samples  # Norman 2019 is entirely K562

    # Real per-perturbation target genes from the CRISPR guide identities (e.g. "KLF1,MAP2K6").
    guide_by_pert: dict[str, str] = {}
    for i in keep:
        guide_by_pert.setdefault(pert_all[i], guide_all[i])
    pert_features = {}
    for p in set(pert_names):
        gi = guide_by_pert.get(p, "")
        targets = [gene_to_idx[g] for g in gi.replace("+", ",").split(",") if g in gene_to_idx]
        pert_features[p] = {"target_genes": np.array(targets, dtype=int), "pathway": -1}

    deg_indices = _compute_degs(ctrl_expr, pert_expr, pert_names, n_top=N_TOP_DEGS)

    train_idx, select_idx, test_idx, split_meta = _hybrid_split(pert_names, world_seed=0)
    splits = Splits(train_idx, select_idx, test_idx,
                    meta={**split_meta, "dataset": "norman_2019"})
    splits.validate(n_samples=n_samples)

    dataset = PerturbationDataset(
        ctrl_expr=ctrl_expr,
        pert_expr=pert_expr,
        pert_names=pert_names,
        pert_types=["gene"] * n_samples,
        cell_types=cell_types,
        gene_names=genes,
        deg_indices=deg_indices,
        train_idx=train_idx,
        val_idx=select_idx,   # `val_idx` retained as an alias for the select split
        test_idx=test_idx,
        pert_features=pert_features,
        gene_pathway=np.full(len(genes), -1, dtype=int),
        n_pathways=0,
        splits=splits,
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

    # Serialize pert_features: convert ndarray values to lists for JSON
    pf_serializable = {}
    for pname, feats in dataset.pert_features.items():
        pf_serializable[pname] = {
            "target_genes": feats["target_genes"].tolist() if isinstance(feats["target_genes"], np.ndarray) else feats["target_genes"],
            "pathway": int(feats["pathway"]),
        }

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
        pert_features_json=json.dumps(pf_serializable),
        gene_pathway=dataset.gene_pathway,
        n_pathways=np.array([dataset.n_pathways]),
    )
    print(f"Cached processed dataset to {path}")


def _load_cached(path: Path) -> PerturbationDataset:
    """Load cached processed dataset."""
    data = np.load(path, allow_pickle=True)
    deg_indices = {k: np.array(v) for k, v in json.loads(str(data["deg_indices_json"])).items()}

    # Load perturbation features if available
    pert_features = {}
    if "pert_features_json" in data:
        pf_raw = json.loads(str(data["pert_features_json"]))
        for pname, feats in pf_raw.items():
            pert_features[pname] = {
                "target_genes": np.array(feats["target_genes"]),
                "pathway": feats["pathway"],
            }

    gene_pathway = data["gene_pathway"] if "gene_pathway" in data else np.array([])
    n_pathways = int(data["n_pathways"][0]) if "n_pathways" in data else 0

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
        pert_features=pert_features,
        gene_pathway=gene_pathway,
        n_pathways=n_pathways,
        splits=Splits(data["train_idx"], data["val_idx"], data["test_idx"], meta={"cached": True}),
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

    All DEG metrics are computed PER-CELL then averaged. This means models
    that capture cell-level variation (expression-dependent effects, cell-type
    conditioning) are properly rewarded.

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

    # 1. Per-cell Pearson on DELTA (predicted effect vs true effect) on top-20 DEGs (PRIMARY)
    # This measures whether the model captures the perturbation effect pattern,
    # not just the baseline expression. ctrl_expr is required for delta computation.
    pearson_degs = []
    for pname in set(pert_names):
        if pname not in deg_indices:
            continue
        mask = np.array([p == pname for p in pert_names])
        degs = deg_indices[pname]
        if len(degs) == 0:
            continue

        for i in np.where(mask)[0]:
            if ctrl_expr is not None:
                pred_delta = predictions[i, degs] - ctrl_expr[i, degs]
                true_delta = ground_truth[i, degs] - ctrl_expr[i, degs]
            else:
                pred_delta = predictions[i, degs]
                true_delta = ground_truth[i, degs]
            if np.std(pred_delta) < 1e-10 or np.std(true_delta) < 1e-10:
                continue
            r = np.corrcoef(pred_delta, true_delta)[0, 1]
            if not np.isnan(r):
                pearson_degs.append(r)

    metrics["pearson_deg"] = float(np.mean(pearson_degs)) if pearson_degs else 0.0

    # 2. Per-cell MSE on top-20 DEGs (GUARD)
    mse_degs = []
    for pname in set(pert_names):
        if pname not in deg_indices:
            continue
        mask = np.array([p == pname for p in pert_names])
        degs = deg_indices[pname]
        if len(degs) == 0:
            continue

        for i in np.where(mask)[0]:
            mse = float(np.mean((predictions[i, degs] - ground_truth[i, degs]) ** 2))
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

            # Per-perturbation direction accuracy (averaged over cells would be too noisy)
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
    print(f"Perturbation features: {len(dataset.pert_features)} perturbations")
    print(f"Gene pathways: {dataset.n_pathways} pathways")

    # Show split composition
    train_perts = set(dataset.pert_names[i] for i in dataset.train_idx)
    val_perts = set(dataset.pert_names[i] for i in dataset.val_idx)
    test_perts = set(dataset.pert_names[i] for i in dataset.test_idx)
    seen_val = train_perts & val_perts
    seen_test = train_perts & test_perts
    unseen_val = val_perts - train_perts
    unseen_test = test_perts - train_perts
    print(f"Val: {len(seen_val)} seen perts + {len(unseen_val)} unseen perts")
    print(f"Test: {len(seen_test)} seen perts + {len(unseen_test)} unseen perts")

    # Achievable-headroom report (floor -> Bayes-optimal oracle) for this world.
    print("\nAchievable ceiling (select split):")
    ceiling = SyntheticAdapter(n_genes=dataset.n_genes).oracle_ceiling(world_seed=WORLD_SEED)
    for metric, report in ceiling.items():
        print(f"  {metric}: floor={report.floor:.4f} -> oracle={report.oracle:.4f}")

    # Test evaluation with baseline predictions, reported against the ceiling.
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
