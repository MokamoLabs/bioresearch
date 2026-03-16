"""
FROZEN evaluation harness for clinical trial outcome prediction.

DO NOT MODIFY THIS FILE.

Task: Given drug (SMILES), targets, indication, trial design parameters,
predict whether the clinical trial will succeed.

Data: TrialBench, TDC Clinical Trial Benchmark
Metrics:
    PRIMARY:   auroc            (discrimination)
    GUARD:     calibration_ece  (must stay <0.15)
    GUARD:     net_value        (must be >0: success=+$1B, failure=-$200M)
    BONUS:     lift_at_10       (concentration of successes in top 10%)
    DIAG:      phase_stratified_auroc
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

SEED = int(os.environ.get("SEED", "42"))
TIME_BUDGET = int(os.environ.get("TIME_BUDGET", "600"))
DATA_DIR = os.environ.get("DATA_DIR", os.path.expanduser("~/.cache/bioresearch/trials"))

# Economic parameters
SUCCESS_VALUE = 1_000_000_000    # $1B for successful trial
FAILURE_COST = -200_000_000      # -$200M for failed trial
TRIAL_COST = 50_000_000          # $50M base trial cost


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class TrialDataset:
    """Clinical trial dataset."""
    # Trial features
    drug_smiles: list[str]
    target_names: list[list[str]]    # list of targets per trial
    indications: list[str]           # disease indication
    phases: list[int]                # trial phase (1, 2, 3)
    enrollment: list[int]           # number of patients
    # Feature matrix (pre-computed)
    features: np.ndarray             # n_trials x n_features
    feature_names: list[str]
    # Labels
    labels: np.ndarray               # 1 = success, 0 = failure
    # Split indices (temporal split)
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def load_data(use_tdc: bool = True) -> TrialDataset:
    """Load clinical trial data."""
    cache_path = Path(DATA_DIR) / "trials_processed.npz"
    if cache_path.exists():
        return _load_cached(cache_path)

    if use_tdc:
        try:
            return _load_tdc(cache_path)
        except (ImportError, Exception) as e:
            print(f"TDC loading failed ({e}), using synthetic data.")

    return _make_synthetic_dataset()


def _make_synthetic_dataset(n_trials: int = 1000, n_features: int = 100) -> TrialDataset:
    """Create synthetic clinical trial data."""
    rng = np.random.RandomState(SEED)

    phases = rng.choice([1, 2, 3], n_trials, p=[0.3, 0.4, 0.3]).tolist()
    # Success rates by phase (realistic)
    phase_success_rates = {1: 0.65, 2: 0.35, 3: 0.60}

    labels = np.zeros(n_trials, dtype=np.float32)
    features = rng.randn(n_trials, n_features).astype(np.float32)
    drug_smiles = [f"C{'C' * rng.randint(1, 15)}O" for _ in range(n_trials)]
    target_names = [[f"TARGET_{rng.randint(0, 50)}"] for _ in range(n_trials)]
    indications = [rng.choice(["NSCLC", "breast_cancer", "AML", "melanoma", "glioblastoma"]) for _ in range(n_trials)]
    enrollment = rng.randint(50, 5000, n_trials).tolist()

    for i in range(n_trials):
        base_rate = phase_success_rates[phases[i]]
        # Features influence success probability
        logit = np.log(base_rate / (1 - base_rate)) + 0.3 * features[i, :5].sum()
        prob = 1 / (1 + np.exp(-logit))
        labels[i] = 1.0 if rng.rand() < prob else 0.0

    feature_names = [f"feat_{i}" for i in range(n_features)]

    # Temporal split (simulate time ordering)
    train_idx = np.arange(0, int(n_trials * 0.6))
    val_idx = np.arange(int(n_trials * 0.6), int(n_trials * 0.8))
    test_idx = np.arange(int(n_trials * 0.8), n_trials)

    return TrialDataset(
        drug_smiles=drug_smiles,
        target_names=target_names,
        indications=indications,
        phases=phases,
        enrollment=enrollment,
        features=features,
        feature_names=feature_names,
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )


def _load_tdc(cache_path: Path) -> TrialDataset:
    """Load from TDC Clinical Trial Benchmark."""
    from tdc.single_pred import Trial

    print("Loading clinical trial data from TDC...")

    # TDC provides trial outcome prediction datasets
    # Try multiple available datasets
    for dataset_name in ["Phase1", "Phase2", "Phase3", "Approval"]:
        try:
            data = Trial(name=dataset_name, path=DATA_DIR)
            df = data.get_data()
            print(f"  Loaded TDC {dataset_name}: {len(df)} trials")

            # Extract features from the dataframe
            drug_smiles = df["Drug"].tolist() if "Drug" in df.columns else ["C" * 5] * len(df)
            labels = df["Y"].values.astype(np.float32) if "Y" in df.columns else np.zeros(len(df), dtype=np.float32)

            # Build feature matrix from available columns
            feature_cols = [c for c in df.columns if c not in ("Drug", "Y", "Drug_ID")]
            if feature_cols:
                features = df[feature_cols].select_dtypes(include=[np.number]).fillna(0).values.astype(np.float32)
            else:
                # Fallback: use fingerprints as features
                try:
                    from domains.molecules.prepare import _compute_fingerprints
                    features = _compute_fingerprints(drug_smiles, fp_dim=256)
                except Exception:
                    rng = np.random.RandomState(SEED)
                    features = rng.randn(len(df), 100).astype(np.float32)

            n = len(df)
            target_names = [["UNKNOWN"]] * n
            indications = ["UNKNOWN"] * n
            phases = [int(dataset_name.replace("Phase", "")) if "Phase" in dataset_name else 3] * n
            enrollment = [300] * n
            feature_names = feature_cols if feature_cols else [f"feat_{i}" for i in range(features.shape[1])]

            # Temporal split (by index order, simulating time)
            train_idx = np.arange(0, int(n * 0.6))
            val_idx = np.arange(int(n * 0.6), int(n * 0.8))
            test_idx = np.arange(int(n * 0.8), n)

            dataset = TrialDataset(
                drug_smiles=drug_smiles,
                target_names=target_names,
                indications=indications,
                phases=phases,
                enrollment=enrollment,
                features=features,
                feature_names=feature_names,
                labels=labels,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
            )
            _save_cached(dataset, cache_path)
            return dataset

        except Exception as e:
            print(f"  Could not load TDC {dataset_name}: {e}")
            continue

    print("  All TDC datasets failed. Using synthetic data.")
    return _make_synthetic_dataset()


def _save_cached(dataset: TrialDataset, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        drug_smiles=dataset.drug_smiles,
        target_names_json=json.dumps(dataset.target_names),
        indications=dataset.indications,
        phases=dataset.phases,
        enrollment=dataset.enrollment,
        features=dataset.features,
        feature_names=dataset.feature_names,
        labels=dataset.labels,
        train_idx=dataset.train_idx,
        val_idx=dataset.val_idx,
        test_idx=dataset.test_idx,
    )


def _load_cached(path: Path) -> TrialDataset:
    data = np.load(path, allow_pickle=True)
    return TrialDataset(
        drug_smiles=list(data["drug_smiles"]),
        target_names=json.loads(str(data["target_names_json"])),
        indications=list(data["indications"]),
        phases=list(data["phases"]),
        enrollment=list(data["enrollment"]),
        features=data["features"],
        feature_names=list(data["feature_names"]),
        labels=data["labels"],
        train_idx=data["train_idx"],
        val_idx=data["val_idx"],
        test_idx=data["test_idx"],
    )


# ---------------------------------------------------------------------------
# Evaluation (FROZEN — DO NOT MODIFY)
# ---------------------------------------------------------------------------

def evaluate(
    predictions: np.ndarray,
    labels: np.ndarray,
    phases: list[int] | None = None,
) -> dict[str, float]:
    """
    Evaluate clinical trial predictions.

    Args:
        predictions: Predicted success probabilities (n_trials,)
        labels: True outcomes, 0 or 1 (n_trials,)
        phases: Optional trial phase labels

    Returns:
        Dict of metric values
    """
    from sklearn.metrics import roc_auc_score

    metrics = {}

    # 1. AUROC (PRIMARY)
    try:
        if len(np.unique(labels)) > 1:
            metrics["auroc"] = float(roc_auc_score(labels, predictions))
        else:
            metrics["auroc"] = 0.5
    except Exception:
        metrics["auroc"] = 0.5

    # 2. Calibration ECE (GUARD)
    metrics["calibration_ece"] = _expected_calibration_error(predictions, labels)

    # 3. Net economic value (GUARD)
    metrics["net_value"] = _net_economic_value(predictions, labels)

    # 4. Lift at 10% (BONUS)
    metrics["lift_at_10"] = _lift_at_k(predictions, labels, k_frac=0.1)

    # 5. Phase-stratified AUROC (DIAGNOSTIC)
    if phases is not None:
        for phase in [1, 2, 3]:
            mask = np.array([p == phase for p in phases])
            if mask.sum() > 10 and len(np.unique(labels[mask])) > 1:
                try:
                    metrics[f"auroc_phase{phase}"] = float(
                        roc_auc_score(labels[mask], predictions[mask])
                    )
                except Exception:
                    metrics[f"auroc_phase{phase}"] = 0.5

    return metrics


def _expected_calibration_error(predictions: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = predictions[mask].mean()
        ece += mask.sum() / len(predictions) * abs(bin_acc - bin_conf)
    return float(ece)


def _net_economic_value(predictions: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    """
    Net economic value: sum of (decision outcome) - (base cost).
    Positive = model advice makes money on average.
    """
    go_decisions = predictions >= threshold
    n_go = go_decisions.sum()
    if n_go == 0:
        return 0.0

    # For trials we "go" on: compute P&L
    go_labels = labels[go_decisions]
    successes = go_labels.sum()
    failures = n_go - successes

    total_value = successes * SUCCESS_VALUE + failures * FAILURE_COST - n_go * TRIAL_COST

    # Normalize per trial
    return float(total_value / n_go)


def _lift_at_k(predictions: np.ndarray, labels: np.ndarray, k_frac: float = 0.1) -> float:
    """Lift at top k%: how concentrated are successes in the top predictions?"""
    n = len(predictions)
    k = max(1, int(n * k_frac))

    top_k_idx = np.argsort(predictions)[-k:]
    top_k_success_rate = labels[top_k_idx].mean()
    overall_success_rate = labels.mean()

    if overall_success_rate == 0:
        return 1.0
    return float(top_k_success_rate / overall_success_rate)


if __name__ == "__main__":
    print("Loading synthetic trial dataset for testing...")
    dataset = load_data(use_tdc=False)
    print(f"Dataset: {len(dataset.labels)} trials, {dataset.features.shape[1]} features")
    print(f"Success rate: {dataset.labels.mean():.1%}")

    # Test with random predictions
    rng = np.random.RandomState(42)
    predictions = rng.rand(len(dataset.test_idx))
    labels = dataset.labels[dataset.test_idx]
    phases = [dataset.phases[i] for i in dataset.test_idx]
    metrics = evaluate(predictions, labels, phases)
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")
