# AutoPerturb: Agent Program

## Task
Improve the perturbation prediction model in `train.py`. The model predicts
post-perturbation gene expression given control expression and a perturbation label.

## What You Can Modify
- `train.py` — everything is fair game: model architecture, optimizer, hyperparameters,
  training loop, loss function, data augmentation, feature engineering.

## What You Cannot Modify
- `prepare.py` — frozen evaluation harness. Do not touch.
- The metric definitions or evaluation logic.
- The data splits.

## Constraints
- Model must complete training within the TIME_BUDGET (default 600s).
- Must output metrics as JSON on the last line of stdout.
- Can use numpy, scipy, and torch (if GPU available).
- Can import from `knowledge.retrieval` to access biological priors.
- Each seed uses the same dataset but a different 90% subsample of training data.
  Keep the seed-controlled subsample logic intact so evaluation is meaningful.

## Data Structure
The dataset (`PerturbationDataset`) has these fields you can use:

| Field | Type | Description |
|---|---|---|
| `ctrl_expr` | ndarray (n_samples × n_genes) | Control expression per cell |
| `pert_expr` | ndarray (n_samples × n_genes) | Perturbed expression per cell |
| `pert_names` | list[str] | Perturbation name per cell |
| `cell_types` | list[str] | "K562" or "HeLa" per cell |
| `pert_features` | dict[str, dict] | **Per-perturbation features for generalization** |
| `gene_pathway` | ndarray (n_genes,) | Pathway ID for each gene |
| `n_pathways` | int | Total number of pathways |
| `deg_indices` | dict[str, ndarray] | Top-20 DEG indices per perturbation |

### Perturbation Features (`pert_features[pert_name]`)
Each perturbation has features available for ALL perturbations (including unseen ones):
- `"target_genes"`: ndarray of gene indices directly targeted by this perturbation
- `"pathway"`: int, the pathway ID of the primary target gene

These features are the key to generalizing to unseen perturbations.

## Data Split (Hybrid)
The split tests two abilities:
1. **Seen perturbations with held-out cells** — can the model predict better per-cell
   than the mean delta? (Tests expression-dependent modeling)
2. **Unseen perturbations with features** — can the model generalize to new perturbations
   using target gene and pathway features? (Tests feature-based transfer)

- 50% of perturbations: train-only (all cells in training set)
- 20% of perturbations: seen-split (cells divided 70/15/15 across train/val/test)
- 30% of perturbations: unseen (all cells in val or test only, with features available)

## Metric Specifications
- **pearson_deg** [PRIMARY]: Per-cell Pearson correlation on **predicted delta vs true delta**
  (pred - ctrl vs truth - ctrl) on top-20 DEGs. Higher is better. This directly measures
  whether the model captures the perturbation effect pattern per cell.
- **mse_top20_deg** [GUARD]: Per-cell MSE on top-20 DEGs. Must not degrade >10%.
- **direction_acc** [GUARD]: Up/down direction accuracy. Must stay >0.7.
- **cross_context** [BONUS]: Generalization gap across cell types. Lower is better.
- **pearson_all** [DIAGNOSTIC]: Pearson on all genes (absolute values). Reported only.

## Why the Baseline is Suboptimal
The linear baseline uses `ctrl + mean_delta[pert_name]` for seen perturbations and
`ctrl + global_mean_delta` for unseen. It scores ~0.19 on pearson_deg. It misses:

1. **Expression-dependent effects**: The perturbation effect on each gene scales with the
   cell's control expression level (via tanh modulation). Cells with high ctrl expression
   for a target gene respond more strongly. The baseline ignores this → suboptimal per-cell.

2. **Cell-type-specific responses**: K562 and HeLa cells respond with different magnitudes
   (1.0x vs 0.6x scaling). The baseline ignores cell type → noisy predictions.

3. **Perturbation features for unseen perts**: Each perturbation has known target genes and
   pathway. Models that use these can predict perturbation-specific effects for unseen perts
   instead of falling back to `global_mean_delta`. This is the largest source of headroom.

4. **Pathway propagation**: Perturbation effects propagate to secondary genes in the same
   pathway (at 0.3x dampening). Models that learn pathway structure can predict these
   secondary effects.

## Strategy Guidelines
1. The biggest improvement comes from **using perturbation features for unseen perturbations**.
   The baseline gets ~0.08 pearson_deg on unseen perts. Using target genes and pathway can
   push this to 0.5+.
2. For seen perturbations (baseline ~0.79), learn **expression-dependent delta mapping** —
   predict different deltas for cells with different ctrl expression levels.
3. Condition on **cell type** (available in `dataset.cell_types`) for per-cell-type deltas.
4. Use **gene pathway structure** (`dataset.gene_pathway`) to model secondary effects.
5. Pathway-based transfer: perturbations targeting the same pathway have shared effects.
   Use `dataset.pert_features[pname]["pathway"]` to group perturbations.
6. Don't get stuck on regularization variants — if 3 attempts fail, switch approach families.
7. The data has nonlinear structure. MLP/neural approaches for `f(ctrl, pert_features) → delta`
   are worth trying.
8. Make one focused change per iteration. Don't combine unrelated ideas.

## Knowledge Menu
You have access to these biological priors (via `knowledge.retrieval.BioKnowledge`):

| Source | Dims | What It Captures |
|---|---|---|
| gene_text_emb | 768 | Functional gene descriptions |
| gene_ontology | 128 | Functional categories (GO graph) |
| ppi_network | sparse | Protein-protein interactions (STRING) |
| pathway_membership | N×P | Gene-pathway links (Reactome) |
| esm_structure | 1280 | Protein 3D structure (ESM-2) |
| drug_target | D×N | Drug binding data (ChEMBL) |
