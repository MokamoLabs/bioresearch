"""
BioKnowledge: honest access to EXTERNAL biological priors.

History: this module previously advertised six impressive sources (GPT-4 gene embeddings,
ESM-2 structure, STRING PPI, Reactome pathways, ChEMBL drug-target) whose data was, in
fact, `numpy.random.randn` produced by knowledge/precompute.py — and keyed to gene names
that did not even match the datasets. It injected noise into the agent prompt while lending
false biological credibility.

This rewrite is honest by construction:
  * Nothing is fabricated. A source exists only if a real, self-describing `.npz` file is
    present on disk (or is registered in-process from real data via `register_source`).
  * Coverage is measured against the actual query genes. A source that does not cover a
    task's gene identifiers is NOT advertised to the agent.
  * With no real sources configured (the default, and the only honest state for the
    synthetic tasks whose gene ids have no external biology), the packet says so plainly
    and points the model at the real structure already inside the dataset.

Real-source file format (`.npz`): keys `data` (or `embeddings`/`matrix`), `index`
(dict entity->row), and optionally `description` (str). Build them with
knowledge/precompute.py or register them at runtime for real-data campaigns.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class KnowledgeSource:
    name: str
    description: str
    path: str = ""
    data: Optional[np.ndarray] = None
    index: Optional[dict] = None       # entity name -> row index
    loaded: bool = False


class BioKnowledge:
    """Honest registry of real external biological knowledge sources."""

    DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/bioresearch/knowledge")

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = Path(cache_dir or self.DEFAULT_CACHE_DIR)
        self.sources: dict[str, KnowledgeSource] = {}
        self._discover_sources()

    def _discover_sources(self):
        """Register only sources that genuinely exist on disk with a usable index.

        No hardcoded/fabricated sources: what you see is what is really available.
        """
        if not self.cache_dir.exists():
            return
        for path in sorted(self.cache_dir.glob("*.npz")):
            try:
                data = np.load(path, allow_pickle=True)
                if "index" not in data:
                    continue  # a knowledge source without an entity index is unusable
                description = str(data["description"]) if "description" in data else path.stem
                self.sources[path.stem] = KnowledgeSource(
                    name=path.stem, description=description, path=str(path)
                )
            except Exception:
                continue  # unreadable file -> simply not available (never fabricated)

    def register_source(self, name: str, data: np.ndarray, index: dict,
                        description: str) -> KnowledgeSource:
        """Seam for real sources built in-process (e.g. during a real-data campaign)."""
        src = KnowledgeSource(name=name, description=description, data=np.asarray(data),
                              index=dict(index), loaded=True)
        self.sources[name] = src
        return src

    def available_sources(self) -> list[str]:
        return list(self.sources.keys())

    def load(self, source_name: str) -> KnowledgeSource:
        if source_name not in self.sources:
            raise KeyError(f"Unknown knowledge source: {source_name}. "
                           f"Available: {self.available_sources()}")
        source = self.sources[source_name]
        if source.loaded:
            return source
        data = np.load(source.path, allow_pickle=True)
        source.data = data.get("data", data.get("embeddings", data.get("matrix")))
        idx = data["index"]
        source.index = idx.item() if getattr(idx, "ndim", 1) == 0 else dict(idx)
        source.loaded = True
        return source

    def get_embeddings(self, source_name: str, entities: list[str] | None = None) -> np.ndarray:
        source = self.load(source_name)
        if source.data is None:
            raise RuntimeError(f"No data loaded for source '{source_name}'")
        if entities is None or source.index is None:
            return source.data
        rows = [source.index[e] for e in entities if e in source.index]
        return source.data[rows]

    def coverage(self, source_name: str, entities: list[str]) -> tuple[int, int]:
        """Return (n_covered, n_total) real coverage of `entities` by a source's index."""
        source = self.load(source_name)
        if not source.index:
            return (0, len(entities))
        covered = sum(1 for e in entities if e in source.index)
        return (covered, len(entities))

    def get_knowledge_packet(self, gene_list: list[str] | None = None) -> str:
        """Agent-facing description of genuinely-available, gene-covering knowledge."""
        if not self.sources:
            return (
                "No external biological knowledge sources are configured for this task. "
                "Rely on the biological structure already present in the dataset itself "
                "(e.g. gene_pathway groupings and per-perturbation target features)."
            )

        lines = ["External biological knowledge sources available for this task:", ""]
        advertised = 0
        for name, src in self.sources.items():
            cov: Optional[tuple[int, int]] = None
            if gene_list:
                try:
                    cov = self.coverage(name, gene_list)
                except Exception:
                    cov = None
                if not cov or cov[0] == 0:
                    continue  # don't advertise a source that covers none of these genes
            advertised += 1
            line = f"- **{name}**: {src.description}"
            if cov:
                line += f"  (covers {cov[0]}/{cov[1]} of this task's genes)"
            lines.append(line)

        if advertised == 0:
            n = len(gene_list) if gene_list else 0
            return (
                f"Configured knowledge sources do not cover this task's gene identifiers "
                f"(checked {n}). Rely on the dataset's own structure (gene_pathway, target "
                "features) rather than external priors."
            )

        lines += [
            "",
            "Load in train.py via:",
            "  from knowledge.retrieval import BioKnowledge",
            "  emb = BioKnowledge().get_embeddings('<source_name>', entities=gene_names)",
        ]
        return "\n".join(lines)
