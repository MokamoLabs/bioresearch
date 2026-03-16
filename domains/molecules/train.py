"""
ADMET prediction + molecular generation model — MUTABLE.

This file is what the autoresearch agent modifies.

Two-layer optimization:
1. Prediction layer: Molecular fingerprints -> 22-endpoint ADMET profile
2. Generation layer: Target ADMET profile -> candidate molecules (SMILES)

Starting point: Ridge/LogReg per-endpoint prediction + random SMILES mutation generation.

Usage: python domains/molecules/train.py
Outputs metrics as JSON on the last line of stdout.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from domains.molecules.prepare import (
    load_data, evaluate, evaluate_generation, ADMET_ENDPOINTS, _compute_fingerprints,
)

SEED = int(os.environ.get("SEED", "42"))
TIME_BUDGET = int(os.environ.get("TIME_BUDGET", "600"))

np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Layer 1: ADMET Prediction Model
# ---------------------------------------------------------------------------

class FingerprintMLPModel:
    """
    Multi-task ADMET prediction using molecular fingerprints.

    For each endpoint, trains an independent Ridge regression/logistic regression.
    This is a simple but solid baseline.
    """

    def __init__(self, n_endpoints: int, endpoint_types: list[str], reg_strength: float = 1.0, fp_dim: int = 256):
        self.n_endpoints = n_endpoints
        self.endpoint_types = endpoint_types
        self.reg_strength = reg_strength
        self.fp_dim = fp_dim
        self.models: list = [None] * n_endpoints

    def fit(self, fingerprints: np.ndarray, labels: np.ndarray):
        """Fit per-endpoint models."""
        from sklearn.linear_model import Ridge, LogisticRegression

        for j in range(self.n_endpoints):
            mask = ~np.isnan(labels[:, j])
            if mask.sum() < 10:
                continue

            X = fingerprints[mask].astype(np.float64)
            y = labels[mask, j].astype(np.float64)

            if self.endpoint_types[j] == "classification":
                if len(np.unique(y)) < 2:
                    continue
                model = LogisticRegression(
                    C=1.0 / max(self.reg_strength, 0.01),
                    max_iter=500,
                    random_state=SEED,
                    solver="lbfgs",
                )
                model.fit(X, y)
            else:
                model = Ridge(alpha=max(self.reg_strength, 1.0), random_state=SEED)
                model.fit(X, y)

            self.models[j] = model

    def predict(self, fingerprints: np.ndarray) -> np.ndarray:
        """Predict all endpoints."""
        n = fingerprints.shape[0]
        preds = np.zeros((n, self.n_endpoints), dtype=np.float32)

        for j in range(self.n_endpoints):
            if self.models[j] is None:
                continue

            fp64 = fingerprints.astype(np.float64)
            if self.endpoint_types[j] == "classification":
                try:
                    preds[:, j] = self.models[j].predict_proba(fp64)[:, 1]
                except Exception:
                    preds[:, j] = self.models[j].predict(fp64)
            else:
                preds[:, j] = self.models[j].predict(fp64)

        return preds

    def predict_single(self, smiles: str) -> dict[str, float]:
        """Predict ADMET profile for a single SMILES string."""
        fp = _compute_fingerprints([smiles], fp_dim=self.fp_dim)
        preds = self.predict(fp)
        endpoint_names = list(ADMET_ENDPOINTS.keys())
        return {name: float(preds[0, j]) for j, name in enumerate(endpoint_names)}


# ---------------------------------------------------------------------------
# Layer 2: Molecular Generator (baseline: random SMILES mutation)
# ---------------------------------------------------------------------------

# Common SMILES building blocks for mutation
SMILES_FRAGMENTS = [
    "C", "CC", "CCC", "c1ccccc1", "C(=O)", "N", "O", "S", "F", "Cl", "Br",
    "C(=O)O", "C(=O)N", "c1ccncc1", "C1CCCCC1", "C(O)", "C(N)", "C#N",
    "c1ccc(O)cc1", "c1ccc(N)cc1", "C(F)(F)F", "OC", "NC",
]


class MoleculeGenerator:
    """
    Baseline molecular generator: random mutation of known drug-like SMILES.

    Strategy: take existing molecules from the training set, apply random
    mutations (insertions, deletions, substitutions of fragments), filter
    for validity, and return candidates.

    The agent should replace this with more sophisticated generation
    (e.g., VAE, diffusion, REINFORCE, genetic algorithm with proper chemistry).
    """

    def __init__(self, seed_smiles: list[str], rng_seed: int = 42):
        self.seed_smiles = [s for s in seed_smiles if len(s) > 3]
        self.rng = np.random.RandomState(rng_seed)

    def generate(self, n_molecules: int = 100, target_profile: dict[str, float] | None = None) -> list[str]:
        """Generate n_molecules candidate SMILES strings."""
        generated = set()
        attempts = 0
        max_attempts = n_molecules * 20

        while len(generated) < n_molecules and attempts < max_attempts:
            attempts += 1
            # Pick a random seed molecule
            base = self.rng.choice(self.seed_smiles)

            # Apply random mutation
            mutated = self._mutate(base)
            if mutated and mutated not in generated:
                generated.add(mutated)

        return list(generated)

    def _mutate(self, smiles: str) -> str | None:
        """Apply a random mutation to a SMILES string."""
        mutation_type = self.rng.choice(["insert", "delete", "substitute"])

        if mutation_type == "insert":
            frag = self.rng.choice(SMILES_FRAGMENTS)
            pos = self.rng.randint(0, max(1, len(smiles)))
            return smiles[:pos] + frag + smiles[pos:]

        elif mutation_type == "delete" and len(smiles) > 5:
            start = self.rng.randint(0, len(smiles) - 2)
            length = self.rng.randint(1, min(4, len(smiles) - start))
            return smiles[:start] + smiles[start + length:]

        elif mutation_type == "substitute":
            frag = self.rng.choice(SMILES_FRAGMENTS)
            if len(smiles) > 3:
                start = self.rng.randint(0, len(smiles) - 2)
                length = self.rng.randint(1, min(3, len(smiles) - start))
                return smiles[:start] + frag + smiles[start + length:]

        return smiles


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    dataset = load_data(use_tdc=False)  # Change to True when TDC is available
    print(f"Dataset: {len(dataset.smiles)} molecules, {len(dataset.endpoint_names)} endpoints")

    # --- Layer 1: ADMET Prediction ---
    train_fp = dataset.fingerprints[dataset.train_idx]
    train_labels = dataset.labels[dataset.train_idx]

    model = FingerprintMLPModel(
        n_endpoints=len(dataset.endpoint_names),
        endpoint_types=dataset.endpoint_types,
        fp_dim=dataset.fingerprints.shape[1],
    )
    model.fit(train_fp, train_labels)

    prediction_time = time.time() - t_start
    print(f"Prediction model training: {prediction_time:.1f}s")

    # Evaluate prediction
    val_fp = dataset.fingerprints[dataset.val_idx]
    val_labels = dataset.labels[dataset.val_idx]
    predictions = model.predict(val_fp)
    pred_metrics = evaluate(predictions, val_labels, dataset.endpoint_names, dataset.endpoint_types)

    # --- Layer 2: Molecular Generation ---
    t_gen_start = time.time()

    # Build a target ADMET profile from median of successful training drugs
    target_profile = {}
    for j, name in enumerate(dataset.endpoint_names):
        vals = dataset.labels[dataset.train_idx, j]
        valid = vals[~np.isnan(vals)]
        if len(valid) > 0:
            target_profile[name] = float(np.median(valid))

    # Generate molecules
    train_smiles = [dataset.smiles[i] for i in dataset.train_idx]
    generator = MoleculeGenerator(seed_smiles=train_smiles, rng_seed=SEED)
    generated = generator.generate(n_molecules=100, target_profile=target_profile)

    gen_time = time.time() - t_gen_start
    print(f"Generation: {len(generated)} molecules in {gen_time:.1f}s")

    # Evaluate generation
    gen_metrics = evaluate_generation(
        generated_smiles=generated,
        target_profile=target_profile,
        prediction_model=model,
    )

    # Combine all metrics
    metrics = {}
    metrics.update(pred_metrics)
    metrics.update(gen_metrics)
    metrics["train_seconds"] = time.time() - t_start
    metrics["peak_vram_mb"] = 0.0

    print(json.dumps({k: float(v) for k, v in metrics.items()}))


if __name__ == "__main__":
    main()
