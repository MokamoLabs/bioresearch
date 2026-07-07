"""Honesty guards for the knowledge layer (Phase 4)."""

import os
import tempfile

import numpy as np
import pytest


def test_no_fabricated_sources_by_default():
    """An empty cache must yield zero sources and an honest packet (no fake advertising)."""
    from knowledge.retrieval import BioKnowledge

    with tempfile.TemporaryDirectory() as tmp:
        kb = BioKnowledge(cache_dir=tmp)
        assert kb.available_sources() == []
        packet = kb.get_knowledge_packet(gene_list=["TP53", "EGFR"])
        assert "No external biological knowledge" in packet
        # Must NOT advertise the old fabricated sources.
        for fake in ("ESM", "STRING", "Reactome pathway membership matrix", "GPT-4"):
            assert fake not in packet


def test_precompute_refuses_to_fabricate():
    """The heavy sources must raise rather than invent random data."""
    from knowledge import precompute

    for builder in (precompute.build_esm_structure, precompute.build_ppi_network,
                    precompute.build_drug_target, precompute.build_gene_text_embeddings):
        with pytest.raises(NotImplementedError):
            builder()


def test_real_pathway_source_reports_true_coverage(monkeypatch):
    """A real GMT builds a joinable source; coverage is measured, not faked."""
    from knowledge import precompute
    from knowledge.retrieval import BioKnowledge

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(precompute, "CACHE_DIR", __import__("pathlib").Path(tmp))
        gmt = os.path.join(tmp, "sets.gmt")
        with open(gmt, "w") as f:
            f.write("PATHWAY_A\tdesc\tTP53\tEGFR\tKRAS\n")
            f.write("PATHWAY_B\tdesc\tEGFR\tMYC\n")
        precompute.build_pathway_membership_from_gmt(gmt)

        kb = BioKnowledge(cache_dir=tmp)
        assert "pathway_membership" in kb.available_sources()

        # Real genes are covered...
        covered, total = kb.coverage("pathway_membership", ["TP53", "EGFR", "NOTAGENE"])
        assert (covered, total) == (2, 3)
        packet = kb.get_knowledge_packet(gene_list=["TP53", "EGFR"])
        assert "pathway_membership" in packet and "covers 2/2" in packet

        # ...synthetic gene ids are NOT covered, and are honestly reported as such.
        packet2 = kb.get_knowledge_packet(gene_list=["GENE_0000", "GENE_0001"])
        assert "do not cover" in packet2


def test_register_source_seam():
    """Real sources can be registered in-process (used by real-data campaigns)."""
    from knowledge.retrieval import BioKnowledge

    with tempfile.TemporaryDirectory() as tmp:
        kb = BioKnowledge(cache_dir=tmp)
        emb = np.arange(6, dtype=np.float32).reshape(3, 2)
        kb.register_source("go_emb", emb, {"TP53": 0, "EGFR": 1, "KRAS": 2},
                           description="real GO embeddings (2d)")
        assert "go_emb" in kb.available_sources()
        got = kb.get_embeddings("go_emb", entities=["EGFR", "TP53"])
        assert np.array_equal(got, np.array([[2, 3], [0, 1]], dtype=np.float32))
