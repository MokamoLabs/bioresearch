"""
Build REAL external biological-knowledge sources for BioKnowledge.

This module used to fabricate every source with `numpy.random.randn` and save it under
names like "ESM-2 structure embeddings" — pure noise dressed up as biology. That is gone.

What remains is honest:
  * `build_pathway_membership_from_gmt` builds a genuine gene->pathway binary matrix from a
    standard GMT gene-set file (e.g. Reactome/MSigDB), keyed by real gene symbols. It is
    dependency-free and verifiable.
  * The heavy sources (protein-language-model embeddings, STRING PPI, ChEMBL affinities)
    are NOT fabricated. Each builder documents the real pipeline and refuses to run rather
    than inventing data. Wire them to their real sources when needed.

Output `.npz` format consumed by knowledge/retrieval.py: `data` (2-D array), `index`
(dict entity->row), `description` (str).

Usage:
    python -m knowledge.precompute --pathways path/to/reactome.gmt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

CACHE_DIR = Path(os.path.expanduser("~/.cache/bioresearch/knowledge"))


def build_pathway_membership_from_gmt(gmt_path: str, out_name: str = "pathway_membership") -> Path:
    """Build a real gene x pathway membership matrix from a GMT gene-set file.

    GMT format: each line is `pathway_name<TAB>description<TAB>gene1<TAB>gene2<TAB>...`.
    The result is a binary matrix [n_genes, n_pathways] with a gene-symbol index, so it can
    be joined to any dataset that uses the same real gene symbols (e.g. Norman 2019).
    """
    gmt = Path(gmt_path)
    if not gmt.exists():
        raise FileNotFoundError(f"GMT file not found: {gmt}")

    pathways: list[str] = []
    pathway_genes: list[list[str]] = []
    for line in gmt.read_text().splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        pathways.append(parts[0])
        pathway_genes.append([g for g in parts[2:] if g])

    genes = sorted({g for gs in pathway_genes for g in gs})
    if not genes:
        raise ValueError(f"No genes parsed from {gmt}")
    gene_index = {g: i for i, g in enumerate(genes)}

    matrix = np.zeros((len(genes), len(pathways)), dtype=np.float32)
    for j, gs in enumerate(pathway_genes):
        for g in gs:
            matrix[gene_index[g], j] = 1.0

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / f"{out_name}.npz"
    description = (
        f"Real gene->pathway membership from {gmt.name}: "
        f"{len(genes)} genes x {len(pathways)} pathways"
    )
    np.savez(out, data=matrix, index=gene_index, description=description)
    print(f"Wrote {out}: {description}")
    return out


def _refuse(source: str, how: str):
    raise NotImplementedError(
        f"Refusing to fabricate '{source}'. Build it from the real source: {how}\n"
        "This module no longer generates random placeholders."
    )


def build_gene_text_embeddings(*_a, **_k):
    _refuse("gene_text_emb",
            "embed real gene descriptions (NCBI Gene / UniProt) with a real embedding model, "
            "then save {data, index (gene->row), description}.")


def build_esm_structure(*_a, **_k):
    _refuse("esm_structure",
            "run ESM-2 over the real protein sequences for your genes and cache the embeddings.")


def build_ppi_network(*_a, **_k):
    _refuse("ppi_network",
            "download the STRING PPI graph, filter to your genes, save the adjacency + index.")


def build_drug_target(*_a, **_k):
    _refuse("drug_target",
            "pull real ChEMBL drug-target affinities and save the matrix keyed by real ids.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build real biological knowledge sources")
    parser.add_argument("--pathways", type=str,
                        help="Path to a GMT gene-set file to build real pathway membership")
    parser.add_argument("--out-name", type=str, default="pathway_membership")
    args = parser.parse_args()

    if args.pathways:
        build_pathway_membership_from_gmt(args.pathways, out_name=args.out_name)
    else:
        print("Nothing to do. Provide --pathways <file.gmt> to build real pathway membership.")
        print("Other sources (gene_text_emb, esm_structure, ppi_network, drug_target) require")
        print("their real pipelines — this module will not fabricate them.")
