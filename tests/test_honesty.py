"""
Consolidated honesty gate (Phase 7).

One place that asserts the cross-domain guarantees which make "the metric improved" mean
something: every real task has genuine, capturable headroom; the negative control has none;
the data generator is hidden from the agent; splits are leakage-free; and every train.py
honours the locked-test contract. If any of these regress, CI fails here.
"""

import importlib
from pathlib import Path

import pytest

# (domain, primary metric, real-headroom expected?) — negative_control is the null control.
DOMAINS = [
    ("perturbation", "pearson_deg", True),
    ("molecules", "composite_admet", True),
    ("trials", "auroc", True),
    ("negative_control", "accuracy", False),
]
ALL = [d[0] for d in DOMAINS]


def _adapter(domain):
    mod = importlib.import_module(f"domains.{domain}.prepare")
    return mod.SyntheticAdapter()


@pytest.mark.parametrize("domain,primary,has_headroom", DOMAINS)
def test_ceiling_matches_domain_type(domain, primary, has_headroom):
    """Real domains must have oracle > floor; the control must have oracle == floor."""
    rep = _adapter(domain).oracle_ceiling(0)
    assert primary in rep, f"{domain} has no ceiling for its primary metric {primary}"
    c = rep[primary]
    if has_headroom:
        assert abs(c.oracle - c.floor) > 0.1, f"{domain}: no real headroom ({c.floor}->{c.oracle})"
    else:
        assert abs(c.oracle - c.floor) < 1e-9, f"{domain}: control must be signal-free"


@pytest.mark.parametrize("domain", ALL)
def test_synthetic_splits_leakage_free(domain):
    d = _adapter(domain).load(0)
    assert d.splits is not None, f"{domain} synthetic dataset carries no Splits"
    d.splits.validate()  # raises on any train/select/test overlap


@pytest.mark.parametrize("domain", ALL)
def test_generator_is_hidden_and_separate(domain):
    """The mechanism lives in a separate _generator.py that prepare.py imports (not inlines)."""
    base = Path("domains") / domain
    assert (base / "_generator.py").exists(), f"{domain} has no hidden generator module"
    prepare = (base / "prepare.py").read_text()
    assert "_generator import" in prepare, f"{domain} prepare.py must import (not inline) its generator"


@pytest.mark.parametrize("domain", ALL)
def test_train_honours_locked_test_contract(domain):
    """Selection uses val by default; the loop switches to test only via EVAL_SPLIT."""
    train = (Path("domains") / domain / "train.py").read_text()
    assert "EVAL_SPLIT" in train, f"{domain} train.py must honour EVAL_SPLIT for the locked test"
    assert "val_idx" in train, f"{domain} train.py must default to the selection split"
