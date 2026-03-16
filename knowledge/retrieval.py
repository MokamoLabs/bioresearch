"""
BioKnowledge: gene/drug/disease embeddings and biological priors.

Provides a unified API for retrieving biological knowledge that the agent
can use to inform its architecture decisions.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class KnowledgeSource:
    name: str
    description: str
    dims: int
    path: str  # path to pre-computed embeddings
    loaded: bool = False
    data: Optional[np.ndarray] = None
    index: Optional[dict[str, int]] = None  # entity name -> row index


class BioKnowledge:
    """
    Unified biological knowledge retrieval.

    Supports multiple knowledge sources:
    - Gene text embeddings (from GPT-4 descriptions)
    - Gene Ontology graph embeddings
    - PPI network (STRING database)
    - Pathway membership (Reactome)
    - Protein structure embeddings (ESM)
    - Drug-target affinity (ChEMBL)
    """

    DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/bioresearch/knowledge")

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = Path(cache_dir or self.DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sources: dict[str, KnowledgeSource] = {}
        self._register_default_sources()

    def _register_default_sources(self):
        """Register all available knowledge sources."""
        sources = [
            KnowledgeSource(
                name="gene_text_emb",
                description="Gene function text embeddings from GPT-4 descriptions (768d)",
                dims=768,
                path=str(self.cache_dir / "gene_text_embeddings.npz"),
            ),
            KnowledgeSource(
                name="gene_ontology",
                description="Gene Ontology graph embeddings (128d)",
                dims=128,
                path=str(self.cache_dir / "gene_ontology_embeddings.npz"),
            ),
            KnowledgeSource(
                name="ppi_network",
                description="Protein-protein interaction network from STRING (sparse adjacency)",
                dims=0,  # sparse matrix, not fixed dims
                path=str(self.cache_dir / "ppi_string.npz"),
            ),
            KnowledgeSource(
                name="pathway_membership",
                description="Reactome pathway membership matrix (N genes x P pathways)",
                dims=0,
                path=str(self.cache_dir / "reactome_pathways.npz"),
            ),
            KnowledgeSource(
                name="esm_structure",
                description="ESM protein structure embeddings (1280d)",
                dims=1280,
                path=str(self.cache_dir / "esm_structure_embeddings.npz"),
            ),
            KnowledgeSource(
                name="drug_target",
                description="Drug-target affinity matrix from ChEMBL (D drugs x N genes)",
                dims=0,
                path=str(self.cache_dir / "chembl_drug_target.npz"),
            ),
        ]
        for s in sources:
            self.sources[s.name] = s

    def available_sources(self) -> list[str]:
        """List available knowledge sources (those with pre-computed data)."""
        return [name for name, s in self.sources.items() if Path(s.path).exists()]

    def all_sources(self) -> list[str]:
        """List all registered knowledge sources."""
        return list(self.sources.keys())

    def load(self, source_name: str) -> KnowledgeSource:
        """Load a knowledge source into memory."""
        if source_name not in self.sources:
            raise KeyError(f"Unknown knowledge source: {source_name}. Available: {list(self.sources.keys())}")

        source = self.sources[source_name]
        if source.loaded:
            return source

        if not Path(source.path).exists():
            raise FileNotFoundError(
                f"Knowledge source '{source_name}' not pre-computed. "
                f"Run `python -m knowledge.precompute --source {source_name}` first."
            )

        data = np.load(source.path, allow_pickle=True)
        source.data = data.get("embeddings", data.get("matrix", data.get("data")))
        if "index" in data:
            source.index = dict(data["index"].item()) if data["index"].ndim == 0 else None
        source.loaded = True
        return source

    def get_embeddings(self, source_name: str, entities: list[str] | None = None) -> np.ndarray:
        """Get embeddings for a knowledge source, optionally filtered to specific entities."""
        source = self.load(source_name)
        if source.data is None:
            raise RuntimeError(f"No data loaded for source '{source_name}'")

        if entities is None or source.index is None:
            return source.data

        indices = []
        for entity in entities:
            if entity in source.index:
                indices.append(source.index[entity])
        return source.data[indices]

    def get_knowledge_packet(self, gene_list: list[str] | None = None) -> str:
        """
        Build a knowledge packet string for the agent.

        This is included in the agent's prompt to inform architecture decisions.
        """
        available = self.available_sources()
        if not available:
            return "No pre-computed biological knowledge available. Run precompute.py first."

        lines = [
            "Available biological knowledge sources for this domain:",
            "",
        ]
        for name in available:
            source = self.sources[name]
            lines.append(f"- **{name}**: {source.description}")
            if source.dims > 0:
                lines.append(f"  Dimensions: {source.dims}")

            # Show coverage for gene list
            if gene_list and Path(source.path).exists():
                try:
                    s = self.load(name)
                    if s.index:
                        covered = sum(1 for g in gene_list if g in s.index)
                        lines.append(f"  Coverage: {covered}/{len(gene_list)} genes")
                except Exception:
                    pass

        lines.extend([
            "",
            "To use a knowledge source in train.py, load it via:",
            "  from knowledge.retrieval import BioKnowledge",
            "  kb = BioKnowledge()",
            "  embeddings = kb.get_embeddings('source_name')",
            "",
            "You can incorporate these as:",
            "- Initial gene embeddings",
            "- Conditioning signals",
            "- Graph adjacency for GNN message passing",
            "- Regularization targets",
            "- Feature augmentation",
        ])

        return "\n".join(lines)
