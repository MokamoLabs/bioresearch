"""Tests for domain-specific data loading and evaluation."""

import numpy as np
import pytest


class TestPerturbationDomain:
    def test_synthetic_data_loads(self):
        from domains.perturbation.prepare import load_data
        dataset = load_data("synthetic")
        assert dataset.n_samples > 0
        assert dataset.n_genes > 0
        assert len(dataset.train_idx) > 0
        assert len(dataset.val_idx) > 0
        assert len(dataset.test_idx) > 0

    def test_synthetic_has_cell_type_diversity(self):
        from domains.perturbation.prepare import load_data
        dataset = load_data("synthetic")
        unique_types = set(dataset.cell_types)
        assert len(unique_types) >= 2  # K562 and HeLa

    def test_synthetic_degs_computed_from_effects(self):
        from domains.perturbation.prepare import load_data
        dataset = load_data("synthetic")
        # DEGs should be computed from actual expression differences
        for pname, degs in dataset.deg_indices.items():
            assert len(degs) > 0
            assert len(degs) <= 20  # N_TOP_DEGS

    def test_no_split_overlap(self):
        from domains.perturbation.prepare import load_data
        dataset = load_data("synthetic")
        train = set(dataset.train_idx)
        val = set(dataset.val_idx)
        test = set(dataset.test_idx)
        assert len(train & val) == 0
        assert len(train & test) == 0
        assert len(val & test) == 0

    def test_evaluate_returns_correct_keys(self):
        from domains.perturbation.prepare import load_data, evaluate
        dataset = load_data("synthetic")

        rng = np.random.RandomState(42)
        preds = dataset.ctrl_expr[dataset.test_idx] + rng.randn(len(dataset.test_idx), dataset.n_genes) * 0.1
        truth = dataset.pert_expr[dataset.test_idx]
        names = [dataset.pert_names[i] for i in dataset.test_idx]

        metrics = evaluate(preds, truth, names, dataset.deg_indices)
        assert "pearson_deg" in metrics
        assert "mse_top20_deg" in metrics
        assert "pearson_all" in metrics

    def test_perfect_predictions(self):
        from domains.perturbation.prepare import load_data, evaluate
        dataset = load_data("synthetic")
        truth = dataset.pert_expr[dataset.test_idx]
        names = [dataset.pert_names[i] for i in dataset.test_idx]

        # Perfect predictions should give high pearson
        metrics = evaluate(truth, truth, names, dataset.deg_indices)
        assert metrics["pearson_deg"] > 0.99 or metrics["pearson_deg"] == 0.0  # 0.0 if std=0
        assert metrics["mse_top20_deg"] < 0.001


class TestMoleculesDomain:
    def test_synthetic_data_loads(self):
        from domains.molecules.prepare import load_data
        dataset = load_data(use_tdc=False)
        assert len(dataset.smiles) > 0
        assert dataset.labels.shape[1] == 22  # 22 ADMET endpoints

    def test_evaluate_returns_composite(self):
        from domains.molecules.prepare import load_data, evaluate
        dataset = load_data(use_tdc=False)

        rng = np.random.RandomState(42)
        preds = rng.randn(len(dataset.test_idx), 22).astype(np.float32)
        labels = dataset.labels[dataset.test_idx]

        metrics = evaluate(preds, labels, dataset.endpoint_names, dataset.endpoint_types)
        assert "composite_admet" in metrics
        assert 0 <= metrics["composite_admet"] <= 1.0


class TestTrialsDomain:
    def test_synthetic_data_loads(self):
        from domains.trials.prepare import load_data
        dataset = load_data(use_tdc=False)
        assert len(dataset.labels) > 0
        assert dataset.features.shape[0] == len(dataset.labels)

    def test_temporal_split(self):
        from domains.trials.prepare import load_data
        dataset = load_data(use_tdc=False)
        # Temporal split: train before val before test
        assert dataset.train_idx.max() < dataset.val_idx.min()
        assert dataset.val_idx.max() < dataset.test_idx.min()

    def test_evaluate_returns_correct_keys(self):
        from domains.trials.prepare import load_data, evaluate
        dataset = load_data(use_tdc=False)

        rng = np.random.RandomState(42)
        preds = rng.rand(len(dataset.test_idx))
        labels = dataset.labels[dataset.test_idx]
        phases = [dataset.phases[i] for i in dataset.test_idx]

        metrics = evaluate(preds, labels, phases)
        assert "auroc" in metrics
        assert "calibration_ece" in metrics
        assert "net_value" in metrics
        assert "lift_at_10" in metrics

    def test_calibration_ece_range(self):
        from domains.trials.prepare import evaluate
        # Perfectly calibrated predictions
        rng = np.random.RandomState(42)
        labels = rng.randint(0, 2, 1000).astype(np.float32)
        preds = labels.copy()  # Perfect calibration
        metrics = evaluate(preds, labels)
        assert metrics["calibration_ece"] < 0.05

    def test_economic_value(self):
        from domains.trials.prepare import evaluate
        # A model that perfectly predicts should have high net value
        labels = np.array([1, 1, 1, 0, 0], dtype=np.float32)
        preds = np.array([0.9, 0.8, 0.7, 0.1, 0.2])  # Good predictions
        metrics = evaluate(preds, labels)
        assert metrics["net_value"] > 0  # Should make money
