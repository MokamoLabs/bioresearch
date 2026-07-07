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

    def test_no_cell_index_overlap(self):
        from domains.perturbation.prepare import load_data
        dataset = load_data("synthetic")
        train = set(dataset.train_idx)
        val = set(dataset.val_idx)
        test = set(dataset.test_idx)
        assert len(train & val) == 0
        assert len(train & test) == 0
        assert len(val & test) == 0

    def test_hybrid_split_has_seen_and_unseen(self):
        from domains.perturbation.prepare import load_data
        dataset = load_data("synthetic")
        train_perts = set(dataset.pert_names[i] for i in dataset.train_idx)
        val_perts = set(dataset.pert_names[i] for i in dataset.val_idx)
        test_perts = set(dataset.pert_names[i] for i in dataset.test_idx)

        # Some perturbations should appear in both train and val/test (seen-split)
        seen_in_val = train_perts & val_perts
        seen_in_test = train_perts & test_perts
        assert len(seen_in_val) > 0 or len(seen_in_test) > 0, "No seen-split perturbations found"

        # Some perturbations should be unseen (only in val/test)
        unseen_val = val_perts - train_perts
        unseen_test = test_perts - train_perts
        assert len(unseen_val) > 0 or len(unseen_test) > 0, "No unseen perturbations found"

    def test_perturbation_features_exist(self):
        from domains.perturbation.prepare import load_data
        dataset = load_data("synthetic")
        assert len(dataset.pert_features) > 0
        for pname, features in dataset.pert_features.items():
            assert "target_genes" in features
            assert "pathway" in features
            assert len(features["target_genes"]) > 0
            assert 0 <= features["pathway"] < dataset.n_pathways

    def test_gene_pathway_info(self):
        from domains.perturbation.prepare import load_data
        dataset = load_data("synthetic")
        assert dataset.gene_pathway is not None
        assert len(dataset.gene_pathway) == dataset.n_genes
        assert dataset.n_pathways > 0
        # All pathway IDs should be valid
        assert all(0 <= p < dataset.n_pathways for p in dataset.gene_pathway)

    def test_unseen_perts_have_features(self):
        """Unseen perturbations should have features available for generalization."""
        from domains.perturbation.prepare import load_data
        dataset = load_data("synthetic")
        train_perts = set(dataset.pert_names[i] for i in dataset.train_idx)
        val_perts = set(dataset.pert_names[i] for i in dataset.val_idx)
        test_perts = set(dataset.pert_names[i] for i in dataset.test_idx)

        unseen = (val_perts | test_perts) - train_perts
        for pname in unseen:
            assert pname in dataset.pert_features, f"Unseen pert {pname} missing features"

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

    def test_delta_based_pearson(self):
        """pearson_deg should measure delta correlation, not absolute expression."""
        from domains.perturbation.prepare import load_data, evaluate
        dataset = load_data("synthetic")
        test_ctrl = dataset.ctrl_expr[dataset.test_idx]
        test_truth = dataset.pert_expr[dataset.test_idx]
        test_names = [dataset.pert_names[i] for i in dataset.test_idx]

        # Zero-delta prediction (just ctrl) should give low/zero pearson_deg
        metrics_zero = evaluate(test_ctrl, test_truth, test_names, dataset.deg_indices,
                                ctrl_expr=test_ctrl)
        # Without ctrl_expr, falls back to absolute values (higher)
        metrics_abs = evaluate(test_ctrl, test_truth, test_names, dataset.deg_indices)

        # Delta-based (with ctrl_expr) should be lower than absolute (without ctrl_expr)
        # for zero-delta predictions, because ctrl expression inflates absolute pearson
        assert metrics_zero["pearson_deg"] < metrics_abs["pearson_deg"] or metrics_zero["pearson_deg"] == 0.0

    def test_perfect_predictions(self):
        from domains.perturbation.prepare import load_data, evaluate
        dataset = load_data("synthetic")
        truth = dataset.pert_expr[dataset.test_idx]
        names = [dataset.pert_names[i] for i in dataset.test_idx]
        ctrl = dataset.ctrl_expr[dataset.test_idx]

        # Perfect predictions should give high pearson
        metrics = evaluate(truth, truth, names, dataset.deg_indices, ctrl_expr=ctrl)
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


class TestMoleculesValidity:
    """Guards for the Phase-2 rebuild: the synthetic ADMET task must have real signal."""

    def test_random_predictor_scores_near_chance(self):
        """A random predictor must NOT score ~0.65 (the old null-task artifact)."""
        from domains.molecules.prepare import (
            SyntheticAdapter, evaluate, _ENDPOINT_NAMES, _ENDPOINT_TYPES)

        d = SyntheticAdapter().load(0)
        rng = np.random.RandomState(7)
        preds = rng.randn(len(d.test_idx), 22).astype(np.float32)
        comp = evaluate(preds, d.labels[d.test_idx], _ENDPOINT_NAMES, _ENDPOINT_TYPES)["composite_admet"]
        assert comp < 0.45, f"random predictor scored {comp}: composite is still gameable"

    def test_signal_is_learnable(self):
        """A simple model trained on fingerprints must beat chance by a real margin."""
        from domains.molecules.prepare import (
            SyntheticAdapter, evaluate, _ENDPOINT_NAMES, _ENDPOINT_TYPES)
        from sklearn.linear_model import Ridge

        d = SyntheticAdapter().load(0)
        preds = np.zeros((len(d.test_idx), 22), dtype=np.float32)
        Xtr, Xte = d.fingerprints[d.train_idx], d.fingerprints[d.test_idx]
        for j in range(22):
            ytr = d.labels[d.train_idx, j]
            m = ~np.isnan(ytr)
            model = Ridge(alpha=1.0).fit(Xtr[m], ytr[m])
            preds[:, j] = model.predict(Xte)
        comp = evaluate(preds, d.labels[d.test_idx], _ENDPOINT_NAMES, _ENDPOINT_TYPES)["composite_admet"]
        assert comp > 0.55, f"learned model only scored {comp}: no learnable signal"

    def test_oracle_ceiling_above_floor(self):
        from domains.molecules.prepare import SyntheticAdapter

        rep = SyntheticAdapter().oracle_ceiling(0)["composite_admet"]
        assert rep.oracle > rep.floor + 0.15, (rep.floor, rep.oracle)


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


class TestPerturbationValidity:
    """Guards for the Phase-2 honest-substrate rebuild of the perturbation task."""

    def test_generator_hidden_from_agent(self):
        """The data-generating mechanism must NOT appear in anything the agent sees."""
        from pathlib import Path
        from domains.perturbation.prepare import SyntheticAdapter

        agent_visible = (
            Path("domains/perturbation/prepare.py").read_text()
            + Path("domains/perturbation/program.md").read_text()
            + SyntheticAdapter().describe_schema()
        )
        # These are constants/phrases from the true generative mechanism.
        for banned in ("tanh", "0.3x", "push this to", "0.5+"):
            assert banned not in agent_visible, f"generator leak: '{banned}' is agent-visible"
        # ...but the mechanism really does exist, hidden away.
        assert "tanh" in Path("domains/perturbation/_generator.py").read_text()

    def test_oracle_ceiling_above_floor(self):
        """The oracle must beat the floor, giving the agent real, measurable headroom."""
        from domains.perturbation.prepare import SyntheticAdapter

        ceiling = SyntheticAdapter().oracle_ceiling(world_seed=0)
        assert "pearson_deg" in ceiling
        rep = ceiling["pearson_deg"]
        assert rep.oracle > rep.floor + 0.2, (rep.floor, rep.oracle)
        # Floor captures 0% of headroom; oracle captures 100%.
        assert abs(rep.fraction_captured(rep.floor)) < 1e-9
        assert abs(rep.fraction_captured(rep.oracle) - 1.0) < 1e-9
        # Lower-is-better metric has oracle below floor.
        assert ceiling["mse_top20_deg"].oracle < ceiling["mse_top20_deg"].floor

    def test_multi_world_generalization_variance(self):
        """Different world seeds are genuinely different data draws (not one fixed world)."""
        from domains.perturbation.prepare import SyntheticAdapter

        adapter = SyntheticAdapter()
        w0 = adapter.load(0)
        w1 = adapter.load(1)
        # Same schema/shape...
        assert w0.ctrl_expr.shape == w1.ctrl_expr.shape
        # ...but different underlying data (control expression differs across worlds).
        assert not np.array_equal(w0.ctrl_expr, w1.ctrl_expr)
        # And different target-gene assignments for the same perturbation name.
        t0 = w0.pert_features["PERT_000"]["target_genes"]
        t1 = w1.pert_features["PERT_000"]["target_genes"]
        assert not (len(t0) == len(t1) and np.array_equal(np.sort(t0), np.sort(t1)))

    def test_splits_are_leakage_free(self):
        """The explicit split contract must be disjoint across train/select/test."""
        from domains.perturbation.prepare import SyntheticAdapter

        d = SyntheticAdapter().load(0)
        assert d.splits is not None
        d.splits.validate(n_samples=d.n_samples)  # raises on any overlap
        # val_idx is the select split (backward-compatible alias).
        assert np.array_equal(np.sort(d.val_idx), np.sort(d.splits.select_idx))


class TestTrialsValidity:
    """Guards for the Phase-2 rebuild of the trials task."""

    def test_signal_is_learnable(self):
        from domains.trials.prepare import SyntheticAdapter, evaluate
        from sklearn.linear_model import LogisticRegression

        d = SyntheticAdapter().load(0)
        clf = LogisticRegression(max_iter=500).fit(d.features[d.train_idx], d.labels[d.train_idx])
        preds = clf.predict_proba(d.features[d.test_idx])[:, 1]
        phases = [d.phases[i] for i in d.test_idx]
        auroc = evaluate(preds, d.labels[d.test_idx], phases)["auroc"]
        assert auroc > 0.58, f"trials AUROC only {auroc}: signal too weak"

    def test_oracle_ceiling_above_floor(self):
        from domains.trials.prepare import SyntheticAdapter

        rep = SyntheticAdapter().oracle_ceiling(0)["auroc"]
        assert rep.oracle > rep.floor + 0.1, (rep.floor, rep.oracle)

    def test_cross_domain_extraction_does_not_crash(self):
        """Regression guard for the dead `pert_embeddings` chain (Tier-5 bug)."""
        from domains.trials.train import extract_perturbation_features

        result = extract_perturbation_features(["DRUG_0"], [["TARGET_1"]])
        # Must return cleanly (ndarray or None) — never raise AttributeError.
        assert result is None or hasattr(result, "shape")


class TestNegativeControl:
    """The calibration task must be genuinely signal-free (oracle == floor)."""

    def test_zero_headroom(self):
        from domains.negative_control.prepare import SyntheticAdapter

        rep = SyntheticAdapter().oracle_ceiling(0)["accuracy"]
        # By construction there is nothing to learn: the best predictor is the base rate.
        assert abs(rep.oracle - rep.floor) < 1e-9

    def test_labels_independent_of_features(self):
        from domains.negative_control.prepare import SyntheticAdapter, evaluate
        from sklearn.linear_model import LogisticRegression

        d = SyntheticAdapter().load(0)
        clf = LogisticRegression(max_iter=200).fit(d.features[d.train_idx], d.labels[d.train_idx])
        preds = clf.predict_proba(d.features[d.test_idx])[:, 1]
        acc = evaluate(preds, d.labels[d.test_idx])["accuracy"]
        # A trained model must not meaningfully beat chance on held-out data.
        assert acc < 0.6, f"negative control leaked signal (acc={acc})"


class TestRealMoleculesADMET:
    """Phase-3 real TDC ADMET adapter (skips cleanly when deps/data are absent)."""

    def test_scaffold_split_partitions_by_whole_scaffold(self):
        pytest.importorskip("rdkit")
        from domains.molecules.prepare import _scaffold_split

        smis = ["c1ccccc1CCO", "c1ccccc1CC", "c1ccccc1C(=O)O", "C1CCCCC1", "C1CCCCC1CC",
                "CCO", "CCCO", "c1ccncc1", "c1ccncc1C", "c1ccncc1CC"]
        tr, se, te = _scaffold_split(smis, frac_train=0.6, frac_select=0.2)
        idx = set(map(int, tr)) | set(map(int, se)) | set(map(int, te))
        assert idx == set(range(len(smis)))                 # every molecule placed once
        assert not (set(map(int, tr)) & set(map(int, te)))  # disjoint splits

    def test_real_admet_loads_with_signal_if_cached(self):
        pytest.importorskip("tdc")
        pytest.importorskip("rdkit")
        import os
        from pathlib import Path

        base = Path(os.path.expanduser("~/.cache/bioresearch/molecules"))
        if not ((base / "admet_processed.npz").exists() or (base / "admet_group").exists()):
            pytest.skip("TDC ADMET data not downloaded")

        from domains.molecules.prepare import RealAdapter

        d = RealAdapter().load()
        assert len(d.smiles) > 1000
        assert d.splits is not None
        d.splits.validate(n_samples=len(d.smiles))
        assert np.isfinite(d.labels).any()       # real labels present
        assert float(d.fingerprints.sum()) > 0   # real, non-zero fingerprints


class TestRealPerturbationNorman:
    """Phase-3 real Norman 2019 adapter (skips cleanly when data isn't downloaded)."""

    def test_real_norman_loads_if_cached(self):
        import os
        from pathlib import Path

        cache = (Path(os.path.expanduser("~/.cache/bioresearch/perturbation"))
                 / "norman_2019_processed.npz")
        if not cache.exists():
            pytest.skip("Norman 2019 not downloaded/processed")

        from domains.perturbation.prepare import RealAdapter

        d = RealAdapter(dataset_name="norman_2019").load()
        assert d.n_samples > 1000
        assert d.splits is not None
        d.splits.validate(n_samples=d.n_samples)
        # Real per-perturbation target genes exist for at least some perturbations.
        assert any(len(f["target_genes"]) > 0 for f in d.pert_features.values())
        # The hybrid split really holds out unseen perturbations.
        train_perts = set(d.pert_names[i] for i in d.train_idx)
        val_perts = set(d.pert_names[i] for i in d.val_idx)
        assert len(val_perts - train_perts) > 0
