# AutoPerturb: Agent Program

## Task
Improve the perturbation prediction model in `train.py`. The model predicts
post-perturbation gene expression given control expression and a perturbation label.
You are scored on how well the **predicted delta** (pred − ctrl) matches the **true
delta** (truth − ctrl) on each perturbation's most-changed genes, per cell.

## What You Can Modify
- `train.py` — everything: model architecture, optimizer, hyperparameters, training
  loop, loss function, feature engineering.

## What You Cannot Modify
- `prepare.py` — frozen evaluation harness (data loading + metrics). Do not touch.
- The data splits or metric definitions.

## Constraints
- Model must complete training within the TIME_BUDGET (default 600s).
- Must output metrics as JSON on the last line of stdout.
- Can use numpy, scipy, scikit-learn, and torch (if available).
- **Each experiment SEED is an independent data world.** The dataset you receive is one
  realization; the loop evaluates you across several. A change is kept only if it helps
  *consistently across worlds*, so avoid tricks that fit one particular draw.

## Data Structure
The dataset (`PerturbationDataset`) exposes these fields:

| Field | Type | Description |
|---|---|---|
| `ctrl_expr` | ndarray (n_cells × n_genes) | Control expression per cell |
| `pert_expr` | ndarray (n_cells × n_genes) | Perturbed expression per cell (target) |
| `pert_names` | list[str] | Perturbation name per cell |
| `cell_types` | list[str] | "K562" or "HeLa" per cell |
| `pert_features` | dict[str, dict] | Per-perturbation features, for ALL perturbations |
| `gene_pathway` | ndarray (n_genes,) | Pathway id per gene |
| `n_pathways` | int | Number of pathways |
| `deg_indices` | dict[str, ndarray] | Top-20 DEG indices per perturbation |
| `train_idx` / `val_idx` / `test_idx` | ndarray | Split indices (`val_idx` = selection split) |

### Perturbation Features (`pert_features[pert_name]`)
Available for every perturbation, **including unseen ones**:
- `"target_genes"`: gene indices directly targeted by this perturbation
- `"pathway"`: pathway id of the primary target

## Data Split (Hybrid)
The split tests two distinct abilities:
1. **Seen perturbations, held-out cells** — can you predict better *per cell* than a
   single average delta? (Tests cell-level modeling.)
2. **Unseen perturbations** — cells for some perturbations appear only in val/test. You
   never saw their expression during training, but you DO have their `target_genes` and
   `pathway`. (Tests feature-based transfer.)

Composition: ~50% of perturbations are train-only, ~20% are seen-split (cells divided
across train/val/test), ~30% are unseen (val/test only, with features available).

## Metric Specifications
- **pearson_deg** [PRIMARY]: per-cell Pearson correlation between predicted delta and
  true delta on top-20 DEGs. Higher is better.
- **mse_top20_deg** [GUARD]: per-cell MSE on top-20 DEGs. Must not degrade materially.
- **direction_acc** [GUARD]: up/down direction accuracy on DEGs.
- **cross_context** [BONUS]: generalization gap across cell types. Lower is better.
- **pearson_all** [DIAGNOSTIC]: Pearson on all genes (absolute values). Reported only.

## Where the Headroom Is (levers, not answers)
The starting baseline predicts `ctrl + mean_delta[pert]` for seen perturbations and
`ctrl + global_mean_delta` for unseen ones. It leaves large, *measurable* headroom — the
harness can report the gap between this floor and a mechanism-aware oracle. The levers
that are known to matter, which the baseline ignores, are:

1. **Perturbation features for unseen perts.** The baseline falls back to one global
   delta for every unseen perturbation. `target_genes` and `pathway` let you predict a
   perturbation-specific effect instead. This is the largest single source of headroom.
2. **Per-cell variation.** The true delta is not identical across cells of the same
   perturbation — it depends on the cell's state. A model that predicts one delta per
   perturbation leaves per-cell signal on the table. What the dependence *is* is for you
   to learn from the training split.
3. **Cell type.** `cell_types` is available; responses are not identical across types.
4. **Pathway / gene structure.** `gene_pathway` groups genes; effects are not independent
   across genes in related pathways.

You are expected to *discover* the functional relationships from the training data — they
are deliberately not written down here. Form a hypothesis, implement it, and let the
cross-world evaluation tell you whether it generalizes.

## Strategy Guidelines
1. Prioritize generalizing to **unseen perturbations** via their features — that is where
   the baseline is weakest and the headroom is largest.
2. For seen perturbations, try to beat the single-mean-delta prediction with a per-cell
   model conditioned on control expression and/or cell type.
3. Make **one focused change per iteration**; don't combine unrelated ideas.
4. If a family of approaches (e.g. regularization variants) fails a few times, switch to a
   fundamentally different direction rather than tuning the same idea.
