"""
Clinical trial prediction model — MUTABLE.

This file is what the autoresearch agent modifies.
Starting point: XGBoost on engineered features + cross-project ADMET/perturbation features.

The chain: molecule (SMILES) -> ADMET profile (from AutoMol) ->
           perturbation signature (from AutoPerturb) -> trial success prediction

Usage: python domains/trials/train.py
Outputs metrics as JSON on the last line of stdout.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from domains.trials.prepare import load_data, evaluate

SEED = int(os.environ.get("SEED", "42"))
TIME_BUDGET = int(os.environ.get("TIME_BUDGET", "600"))

np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Cross-project feature extraction (the molecule -> cell -> patient chain)
# ---------------------------------------------------------------------------

def extract_admet_features(drug_smiles: list[str]) -> np.ndarray | None:
    """
    Extract ADMET profile features from AutoMol.
    Returns n_trials x n_admet_features matrix, or None if unavailable.
    """
    try:
        from domains.molecules.prepare import _compute_fingerprints, ADMET_ENDPOINTS
        from domains.molecules.train import FingerprintMLPModel
        from domains.molecules.prepare import load_data as load_mol_data

        mol_data = load_mol_data(use_tdc=False)

        # Train the ADMET model on molecule data
        model = FingerprintMLPModel(
            n_endpoints=len(mol_data.endpoint_names),
            endpoint_types=mol_data.endpoint_types,
            fp_dim=mol_data.fingerprints.shape[1],
        )
        model.fit(mol_data.fingerprints[mol_data.train_idx], mol_data.labels[mol_data.train_idx])

        # Compute fingerprints and predict ADMET for trial drugs
        fps = _compute_fingerprints(drug_smiles, fp_dim=mol_data.fingerprints.shape[1])
        admet_preds = model.predict(fps)

        print(f"  ADMET features: {admet_preds.shape[1]} endpoints for {len(drug_smiles)} drugs")
        return admet_preds

    except Exception as e:
        print(f"  ADMET features unavailable: {e}")
        return None


def extract_perturbation_features(drug_smiles: list[str], target_names: list[list[str]]) -> np.ndarray | None:
    """
    Extract perturbation signature features from AutoPerturb.
    Maps drug targets to perturbation effects.
    Returns n_trials x n_pert_features matrix, or None if unavailable.
    """
    try:
        from domains.perturbation.prepare import load_data as load_pert_data
        from domains.perturbation.train import LinearPerturbModel

        pert_data = load_pert_data("synthetic")

        # Train perturbation model
        model = LinearPerturbModel(n_genes=pert_data.n_genes)
        model.fit(
            pert_data.ctrl_expr[pert_data.train_idx],
            pert_data.pert_expr[pert_data.train_idx],
            [pert_data.pert_names[i] for i in pert_data.train_idx],
        )

        # For each trial, look up perturbation effects of its targets
        # Use top-10 PCA components of the predicted expression change as features
        n_trials = len(drug_smiles)
        n_pert_features = 10

        pert_features = np.zeros((n_trials, n_pert_features), dtype=np.float32)
        ctrl_mean = pert_data.ctrl_expr.mean(axis=0)

        for i, targets in enumerate(target_names):
            deltas = []
            for target in targets:
                if target in model.pert_embeddings:
                    deltas.append(model.pert_embeddings[target])
            if deltas:
                # Average perturbation effect across targets
                avg_delta = np.mean(deltas, axis=0)
                # Take top absolute effects as features
                top_idx = np.argsort(np.abs(avg_delta))[-n_pert_features:]
                pert_features[i] = avg_delta[top_idx]
            elif model.global_mean_delta is not None:
                top_idx = np.argsort(np.abs(model.global_mean_delta))[-n_pert_features:]
                pert_features[i] = model.global_mean_delta[top_idx]

        print(f"  Perturbation features: {n_pert_features} dims for {n_trials} trials")
        return pert_features

    except Exception as e:
        print(f"  Perturbation features unavailable: {e}")
        return None


# ---------------------------------------------------------------------------
# Model: Gradient Boosted Trees with cross-project features
# ---------------------------------------------------------------------------

class TrialPredictionModel:
    """
    Clinical trial success prediction with cross-project feature integration.

    Uses gradient boosted trees on:
    1. Base trial features (from prepare.py)
    2. ADMET profiles (from AutoMol, if available)
    3. Perturbation signatures (from AutoPerturb, if available)
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 5, learning_rate: float = 0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None

    def fit(self, features: np.ndarray, labels: np.ndarray):
        """Fit the model."""
        try:
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=SEED,
                use_label_encoder=False,
                eval_metric="logloss",
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=SEED,
            )

        self.model.fit(features, labels)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict success probabilities."""
        return self.model.predict_proba(features)[:, 1]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    dataset = load_data(use_tdc=False)
    print(f"Dataset: {len(dataset.labels)} trials, {dataset.features.shape[1]} base features")
    print(f"Success rate: {dataset.labels.mean():.1%}")

    # --- Cross-project feature extraction ---
    print("Extracting cross-project features...")
    admet_features = extract_admet_features(dataset.drug_smiles)
    pert_features = extract_perturbation_features(dataset.drug_smiles, dataset.target_names)

    # Build combined feature matrix
    feature_parts = [dataset.features]
    if admet_features is not None:
        feature_parts.append(admet_features)
    if pert_features is not None:
        feature_parts.append(pert_features)

    combined_features = np.hstack(feature_parts)
    n_extra = combined_features.shape[1] - dataset.features.shape[1]
    print(f"Combined features: {combined_features.shape[1]} ({dataset.features.shape[1]} base + {n_extra} cross-project)")

    # --- Train model ---
    train_features = combined_features[dataset.train_idx]
    train_labels = dataset.labels[dataset.train_idx]

    model = TrialPredictionModel()
    model.fit(train_features, train_labels)

    train_time = time.time() - t_start
    print(f"Training time: {train_time:.1f}s")

    # --- Evaluate on validation set ---
    val_features = combined_features[dataset.val_idx]
    val_labels = dataset.labels[dataset.val_idx]
    val_phases = [dataset.phases[i] for i in dataset.val_idx]

    predictions = model.predict(val_features)
    metrics = evaluate(predictions, val_labels, val_phases)
    metrics["train_seconds"] = train_time
    metrics["peak_vram_mb"] = 0.0
    metrics["n_features_total"] = int(combined_features.shape[1])
    metrics["n_features_cross_project"] = int(n_extra)

    print(json.dumps({k: float(v) for k, v in metrics.items()}))


if __name__ == "__main__":
    main()
