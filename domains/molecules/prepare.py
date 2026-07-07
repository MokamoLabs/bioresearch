"""
FROZEN evaluation harness for ADMET prediction.

DO NOT MODIFY THIS FILE. It defines the data loading, preprocessing,
and evaluation metrics for the molecular ADMET prediction task.

Task: Given a SMILES string, predict 22-endpoint ADMET profile.
Also: generative model that produces molecules matching a target ADMET profile.

Data: TDC ADMET Benchmark (22 datasets, standardized splits)
Metrics:
    PRIMARY:   composite_admet  (clinically-weighted average across 22 endpoints)
    GUARD:     per_endpoint     (no endpoint degrades >15% from baseline)
    PRIMARY2:  gen_profile_match (generation: fraction meeting all target criteria)
    GUARD:     gen_diversity     (>0.7 Tanimoto diversity)
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from domains.base import CeilingReport, DataAdapter, Splits
from domains.molecules._generator import GeneratedMolWorld, generate_world

SEED = int(os.environ.get("SEED", "42"))
# Synthetic ADMET is multi-world: each experiment SEED selects an independent data
# realization (see the hidden domains/molecules/_generator.py).
WORLD_SEED = int(os.environ.get("WORLD_SEED", str(SEED)))
TIME_BUDGET = int(os.environ.get("TIME_BUDGET", "600"))
DATA_DIR = os.environ.get("DATA_DIR", os.path.expanduser("~/.cache/bioresearch/molecules"))

# TDC ADMET endpoints with clinical importance weights
ADMET_ENDPOINTS = {
    # Absorption
    "Caco2_Wang": {"type": "regression", "weight": 1.0, "metric": "mae"},
    "Bioavailability_Ma": {"type": "classification", "weight": 1.0, "metric": "auroc"},
    "Lipophilicity_AstraZeneca": {"type": "regression", "weight": 1.0, "metric": "mae"},
    "Solubility_AqSolDB": {"type": "regression", "weight": 1.0, "metric": "mae"},
    "HIA_Hou": {"type": "classification", "weight": 1.0, "metric": "auroc"},
    "Pgp_Broccatelli": {"type": "classification", "weight": 1.0, "metric": "auroc"},
    # Distribution
    "BBB_Martins": {"type": "classification", "weight": 1.0, "metric": "auroc"},
    "PPBR_AZ": {"type": "regression", "weight": 1.0, "metric": "mae"},
    "VDss_Lombardo": {"type": "regression", "weight": 1.0, "metric": "spearman"},
    # Metabolism
    "CYP2D6_Veith": {"type": "classification", "weight": 1.0, "metric": "auprc"},
    "CYP3A4_Veith": {"type": "classification", "weight": 1.0, "metric": "auprc"},
    "CYP2C9_Veith": {"type": "classification", "weight": 1.0, "metric": "auprc"},
    "CYP2D6_Substrate": {"type": "classification", "weight": 1.0, "metric": "auprc"},
    "CYP3A4_Substrate": {"type": "classification", "weight": 1.0, "metric": "auprc"},
    # Excretion
    "Half_Life_Obach": {"type": "regression", "weight": 1.0, "metric": "spearman"},
    "Clearance_Hepatocyte_AZ": {"type": "regression", "weight": 1.5, "metric": "spearman"},
    "Clearance_Microsome_AZ": {"type": "regression", "weight": 1.5, "metric": "spearman"},
    # Toxicity (2x weight for safety-critical endpoints)
    "hERG": {"type": "classification", "weight": 2.0, "metric": "auroc"},
    "AMES": {"type": "classification", "weight": 1.5, "metric": "auroc"},
    "DILI": {"type": "classification", "weight": 2.0, "metric": "auroc"},
    "LD50_Zhu": {"type": "regression", "weight": 1.5, "metric": "mae"},
    "Skin_Reaction": {"type": "classification", "weight": 1.0, "metric": "auroc"},
}

N_ENDPOINTS = len(ADMET_ENDPOINTS)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class MoleculeDataset:
    """Preprocessed ADMET dataset."""
    smiles: list[str]
    # Labels: n_samples x n_endpoints (NaN for missing)
    labels: np.ndarray
    endpoint_names: list[str]
    endpoint_types: list[str]  # "regression" or "classification"
    # Molecular fingerprints (pre-computed)
    fingerprints: np.ndarray  # n_samples x fp_dim
    # Split indices
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    # Validity-harness additions (populated for synthetic worlds):
    splits: Optional[Splits] = None
    ceiling: Optional[dict] = None


_ENDPOINT_NAMES = list(ADMET_ENDPOINTS.keys())
_ENDPOINT_TYPES = [ADMET_ENDPOINTS[e]["type"] for e in _ENDPOINT_NAMES]

# Agent-facing schema description (shown in the prompt instead of the data generator).
_SCHEMA_DOC = """\
ADMET dataset schema (fields on the object returned by load_data):

  fingerprints  float32 [n_molecules, fp_dim]  molecular feature vectors (INPUT)
  labels        float32 [n_molecules, 22]       endpoint values; NaN where missing (TARGET)
  endpoint_names list[str]                       the 22 ADMET endpoint names
  endpoint_types list[str]                       "classification" or "regression" per endpoint
  smiles        list[str]                        molecule identifiers
  train_idx / val_idx / test_idx                 split indices (val_idx = selection split)

Objective (frozen, see `evaluate`): composite_admet = clinically-weighted average over
endpoints of AUROC (classification) and rank correlation clipped at chance (regression).
Predicting the per-endpoint mean scores ~0 on that composite — you are rewarded only for
genuinely ranking molecules. Endpoints share latent structure, so multi-task modelling
and nonlinear feature maps help; the exact input->label relationship is for you to learn.
"""


def _dataset_from_world(world: GeneratedMolWorld) -> MoleculeDataset:
    splits = Splits(world.train_idx, world.select_idx, world.test_idx, meta=world.split_meta)
    splits.validate(n_samples=world.fingerprints.shape[0])
    return MoleculeDataset(
        smiles=world.smiles,
        labels=world.labels,
        endpoint_names=_ENDPOINT_NAMES,
        endpoint_types=_ENDPOINT_TYPES,
        fingerprints=world.fingerprints,
        train_idx=world.train_idx,
        val_idx=world.select_idx,   # `val_idx` retained as an alias for the select split
        test_idx=world.test_idx,
        splits=splits,
    )


def load_data(use_tdc: bool = True, world_seed: Optional[int] = None) -> MoleculeDataset:
    """Load ADMET data via the appropriate adapter.

    use_tdc=False -> multi-world synthetic (default for the loop/CI); `world_seed` selects
    the world and defaults to the experiment SEED. use_tdc=True -> real TDC data (Phase 3),
    falling back to synthetic if TDC/rdkit are unavailable.
    """
    ws = WORLD_SEED if world_seed is None else world_seed
    if not use_tdc:
        return SyntheticAdapter().load(ws)

    cache_path = Path(DATA_DIR) / "admet_processed.npz"
    if cache_path.exists():
        return _load_cached(cache_path)
    try:
        return RealAdapter().load(ws)
    except Exception as e:
        print(f"TDC loading failed ({e}), using synthetic data.")
        return SyntheticAdapter().load(ws)


def _make_synthetic_dataset(n_samples: int = 2000, fp_dim: int = 256,
                            world_seed: Optional[int] = None, **_legacy) -> MoleculeDataset:
    """Build a synthetic ADMET dataset for one world (thin wrapper over the hidden generator)."""
    ws = WORLD_SEED if world_seed is None else world_seed
    world = generate_world(ws, _ENDPOINT_TYPES, n_samples=n_samples, fp_dim=fp_dim)
    return _dataset_from_world(world)


class SyntheticAdapter(DataAdapter):
    """Multi-world synthetic ADMET data with genuine input->label signal and a ceiling."""

    is_synthetic = True

    def __init__(self, fp_dim: int = 256):
        self.fp_dim = fp_dim
        self.name = "molecules:synthetic"

    def load(self, world_seed: int = 0) -> MoleculeDataset:
        return _make_synthetic_dataset(fp_dim=self.fp_dim, world_seed=world_seed)

    def describe_schema(self) -> str:
        return _SCHEMA_DOC

    def oracle_ceiling(self, world_seed: int = 0, split: str = "select") -> dict:
        """composite_admet floor (per-endpoint mean predictor) and oracle (noise-free labels)."""
        world = generate_world(world_seed, _ENDPOINT_TYPES)
        idx = world.select_idx if split == "select" else world.test_idx
        labels = world.labels[idx]

        m_oracle = evaluate(world.labels_clean[idx], labels, _ENDPOINT_NAMES, _ENDPOINT_TYPES)
        floor_vec = np.nan_to_num(np.nanmean(world.labels[world.train_idx], axis=0))
        floor_pred = np.tile(floor_vec, (len(idx), 1)).astype(np.float32)
        m_floor = evaluate(floor_pred, labels, _ENDPOINT_NAMES, _ENDPOINT_TYPES)

        reports = {}
        if "composite_admet" in m_oracle and "composite_admet" in m_floor:
            reports["composite_admet"] = CeilingReport(
                metric="composite_admet",
                floor=float(m_floor["composite_admet"]),
                oracle=float(m_oracle["composite_admet"]),
                higher_is_better=True,
            )
        return reports


class RealAdapter(DataAdapter):
    """Real TDC ADMET benchmark (22 endpoints). Fully wired in Phase 3."""

    is_synthetic = False

    def __init__(self):
        self.name = "molecules:tdc_admet"

    def load(self, world_seed: int = 0) -> MoleculeDataset:
        cache_path = Path(DATA_DIR) / "admet_processed.npz"
        if cache_path.exists():
            return _load_cached(cache_path)
        return _load_tdc(cache_path)


def _load_tdc(cache_path: Path) -> MoleculeDataset:
    """Load all 22 TDC ADMET endpoints."""
    from tdc.benchmark_group import admet_group

    group = admet_group(path=DATA_DIR)
    endpoint_names = list(ADMET_ENDPOINTS.keys())
    endpoint_types = [ADMET_ENDPOINTS[e]["type"] for e in endpoint_names]

    all_smiles = set()
    endpoint_data = {}

    for endpoint_name in endpoint_names:
        try:
            benchmark = group.get(endpoint_name)
            train, val = benchmark["train_val"], benchmark["test"]
            all_smiles.update(train["Drug"].tolist())
            all_smiles.update(val["Drug"].tolist())
            endpoint_data[endpoint_name] = {
                "train": dict(zip(train["Drug"].tolist(), train["Y"].tolist())),
                "test": dict(zip(val["Drug"].tolist(), val["Y"].tolist())),
            }
        except Exception as e:
            print(f"  Warning: Could not load {endpoint_name}: {e}")

    smiles_list = sorted(all_smiles)
    smiles_to_idx = {s: i for i, s in enumerate(smiles_list)}
    n_samples = len(smiles_list)

    labels = np.full((n_samples, N_ENDPOINTS), np.nan, dtype=np.float32)
    for j, endpoint_name in enumerate(endpoint_names):
        if endpoint_name in endpoint_data:
            for split_data in [endpoint_data[endpoint_name]["train"], endpoint_data[endpoint_name]["test"]]:
                for smi, val in split_data.items():
                    if smi in smiles_to_idx:
                        try:
                            labels[smiles_to_idx[smi], j] = float(val)
                        except (TypeError, ValueError):
                            pass

    # Real Morgan fingerprints (rdkit).
    fingerprints = _compute_fingerprints(smiles_list)

    # Scaffold split: whole Bemis-Murcko scaffolds are assigned to a single split, so the
    # test set contains chemical scaffolds never seen in training — the standard "hard"
    # ADMET generalization split, not a random shuffle.
    train_idx, select_idx, test_idx = _scaffold_split(smiles_list)
    splits = Splits(train_idx, select_idx, test_idx,
                    meta={"split": "scaffold", "n_molecules": n_samples})
    splits.validate(n_samples=n_samples)

    dataset = MoleculeDataset(
        smiles=smiles_list,
        labels=labels,
        endpoint_names=endpoint_names,
        endpoint_types=endpoint_types,
        fingerprints=fingerprints,
        train_idx=train_idx,
        val_idx=select_idx,   # `val_idx` retained as an alias for the select split
        test_idx=test_idx,
        splits=splits,
    )

    _save_cached(dataset, cache_path)
    return dataset


def _scaffold_split(smiles_list: list[str], frac_train: float = 0.7,
                    frac_select: float = 0.15):
    """Bemis-Murcko scaffold split (DeepChem-style): largest scaffold groups fill train
    first, so held-out splits contain unseen scaffolds. Deterministic (real data = 1 world).
    """
    from collections import defaultdict
    from rdkit.Chem.Scaffolds import MurckoScaffold

    groups: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=False)
        except Exception:
            scaf = ""
        groups[scaf].append(i)

    ordered = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
    n = len(smiles_list)
    n_train, n_select = int(n * frac_train), int(n * frac_select)
    train: list[int] = []
    select: list[int] = []
    test: list[int] = []
    for g in ordered:
        if len(train) + len(g) <= n_train:
            train.extend(g)
        elif len(select) + len(g) <= n_select:
            select.extend(g)
        else:
            test.extend(g)
    return (np.array(sorted(train)), np.array(sorted(select)), np.array(sorted(test)))


def _compute_fingerprints(smiles_list: list[str], fp_dim: int = 2048) -> np.ndarray:
    """Compute Morgan fingerprints."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        fps = np.zeros((len(smiles_list), fp_dim), dtype=np.float32)
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=fp_dim)
                fps[i] = np.array(fp)
        return fps
    except ImportError:
        # Fallback: hash-based fingerprints
        rng = np.random.RandomState(42)
        fps = np.zeros((len(smiles_list), fp_dim), dtype=np.float32)
        for i, smi in enumerate(smiles_list):
            for j, c in enumerate(smi.encode()):
                fps[i, (c * 31 + j) % fp_dim] = 1.0
        return fps


def _save_cached(dataset: MoleculeDataset, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        smiles=dataset.smiles,
        labels=dataset.labels,
        endpoint_names=dataset.endpoint_names,
        endpoint_types=dataset.endpoint_types,
        fingerprints=dataset.fingerprints,
        train_idx=dataset.train_idx,
        val_idx=dataset.val_idx,
        test_idx=dataset.test_idx,
    )


def _load_cached(path: Path) -> MoleculeDataset:
    data = np.load(path, allow_pickle=True)
    return MoleculeDataset(
        smiles=list(data["smiles"]),
        labels=data["labels"],
        endpoint_names=list(data["endpoint_names"]),
        endpoint_types=list(data["endpoint_types"]),
        fingerprints=data["fingerprints"],
        train_idx=data["train_idx"],
        val_idx=data["val_idx"],
        test_idx=data["test_idx"],
        splits=Splits(data["train_idx"], data["val_idx"], data["test_idx"], meta={"cached": True}),
    )


# ---------------------------------------------------------------------------
# Evaluation (FROZEN — DO NOT MODIFY)
# ---------------------------------------------------------------------------

def evaluate(
    predictions: np.ndarray,
    labels: np.ndarray,
    endpoint_names: list[str],
    endpoint_types: list[str],
) -> dict[str, float]:
    """
    Evaluate ADMET predictions.

    Args:
        predictions: n_samples x n_endpoints
        labels: n_samples x n_endpoints (may contain NaN)
        endpoint_names: list of endpoint names
        endpoint_types: list of "regression" or "classification"

    Returns:
        Dict with per-endpoint and composite metrics
    """
    from scipy import stats as sp_stats

    metrics = {}
    weighted_sum = 0.0
    total_weight = 0.0

    for j, (name, etype) in enumerate(zip(endpoint_names, endpoint_types)):
        mask = ~np.isnan(labels[:, j])
        if mask.sum() < 10:
            continue

        pred = predictions[mask, j]
        true = labels[mask, j]
        weight = ADMET_ENDPOINTS.get(name, {}).get("weight", 1.0)

        if etype == "classification":
            # AUROC
            from sklearn.metrics import roc_auc_score, average_precision_score
            try:
                if len(np.unique(true)) > 1:
                    auroc = roc_auc_score(true, pred)
                    metrics[f"{name}_auroc"] = auroc
                    weighted_sum += auroc * weight
                    total_weight += weight
                else:
                    metrics[f"{name}_auroc"] = 0.5
            except Exception:
                metrics[f"{name}_auroc"] = 0.5
        else:
            # MAE and Spearman
            mae = np.mean(np.abs(pred - true))
            metrics[f"{name}_mae"] = mae

            try:
                with warnings.catch_warnings():
                    # A constant prediction column (e.g. the floor predictor) makes
                    # Spearman undefined; that is expected and handled as rho=0.
                    warnings.simplefilter("ignore")
                    rho, _ = sp_stats.spearmanr(pred, true)
                if np.isnan(rho):
                    rho = 0.0
                metrics[f"{name}_spearman"] = rho
                # Composite uses rank correlation clipped at chance (0). A mean-predictor
                # scores 0 here. The previous `1 - mae/range` term rewarded predicting the
                # mean, which made composite_admet ~0.65 even on pure-noise data — the
                # artifact that let the loop "keep" false positives on the null task.
                weighted_sum += max(0.0, rho) * weight
                total_weight += weight
            except Exception:
                metrics[f"{name}_spearman"] = 0.0

    # Composite ADMET score
    metrics["composite_admet"] = weighted_sum / total_weight if total_weight > 0 else 0.0

    return metrics


def evaluate_generation(
    generated_smiles: list[str],
    target_profile: dict[str, float],
    prediction_model=None,
) -> dict[str, float]:
    """
    Evaluate molecular generation quality.

    Args:
        generated_smiles: List of generated SMILES strings
        target_profile: Target ADMET profile {endpoint: value}
        prediction_model: ADMET prediction model (for profile matching)

    Returns:
        Dict with generation metrics
    """
    metrics = {}

    # Validity
    try:
        from rdkit import Chem
        valid = [s for s in generated_smiles if Chem.MolFromSmiles(s) is not None]
        metrics["gen_validity"] = len(valid) / len(generated_smiles) if generated_smiles else 0.0
    except ImportError:
        valid = generated_smiles
        metrics["gen_validity"] = 1.0

    # Uniqueness
    unique = set(valid)
    metrics["gen_uniqueness"] = len(unique) / len(valid) if valid else 0.0

    # Diversity (Tanimoto)
    if len(valid) > 1:
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, DataStructs
            fps = []
            for s in list(unique)[:500]:  # cap for speed
                mol = Chem.MolFromSmiles(s)
                if mol:
                    fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

            if len(fps) > 1:
                sims = []
                for i in range(min(100, len(fps))):
                    for j in range(i + 1, min(100, len(fps))):
                        sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
                metrics["gen_diversity"] = 1 - np.mean(sims) if sims else 0.0
            else:
                metrics["gen_diversity"] = 0.0
        except ImportError:
            metrics["gen_diversity"] = 0.5
    else:
        metrics["gen_diversity"] = 0.0

    # Profile match (requires prediction model)
    if prediction_model is not None and target_profile and valid:
        matches = 0
        for smi in valid:
            pred = prediction_model.predict_single(smi)
            all_match = all(
                abs(pred.get(k, 0) - v) < 0.5
                for k, v in target_profile.items()
            )
            if all_match:
                matches += 1
        metrics["gen_profile_match"] = matches / len(valid)
    else:
        metrics["gen_profile_match"] = 0.0

    return metrics


if __name__ == "__main__":
    print("Loading synthetic ADMET dataset for testing...")
    dataset = load_data(use_tdc=False)
    print(f"Dataset: {len(dataset.smiles)} molecules, {N_ENDPOINTS} endpoints")

    # Random predictions should now score ~0 on the composite (chance-anchored).
    rng = np.random.RandomState(42)
    predictions = rng.randn(len(dataset.test_idx), N_ENDPOINTS).astype(np.float32)
    labels = dataset.labels[dataset.test_idx]
    metrics = evaluate(predictions, labels, dataset.endpoint_names, dataset.endpoint_types)
    print(f"Composite ADMET (random predictions): {metrics['composite_admet']:.4f}")

    ceiling = SyntheticAdapter().oracle_ceiling(world_seed=WORLD_SEED)
    if "composite_admet" in ceiling:
        c = ceiling["composite_admet"]
        print(f"Composite ADMET ceiling: floor={c.floor:.4f} -> oracle={c.oracle:.4f}")
