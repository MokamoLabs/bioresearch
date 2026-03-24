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

## Metric Specifications
- **pearson_deg** [PRIMARY]: Pearson correlation on top-20 DEGs per perturbation.
  Higher is better. This is the hardest, most informative metric.
- **mse_top20_deg** [GUARD]: MSE on top-20 DEGs. Must not degrade >10%.
- **direction_acc** [GUARD]: Up/down direction accuracy. Must stay >0.7.
  (Note: baseline is near 0 for unseen perturbations — guard effectively inactive.)
- **cross_context** [BONUS]: Generalization gap across cell types. Lower is better.
- **pearson_all** [DIAGNOSTIC]: Pearson on all genes. Reported only.

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

## Strategy Guidelines
1. The data has realistic complexity: gene pathway correlations, expression-dependent
   perturbation effects, and cell-type-specific responses.
2. The linear baseline misses nonlinear effects and cell-type differences — exploit these.
3. Try MLP/neural approaches for nonlinear expression-to-delta mapping.
4. Condition on cell type (available in `dataset.cell_types`) for better predictions.
5. Gene pathway structure means perturbation effects propagate — model secondary effects.
6. For unseen perturbations, pathway-based transfer learning beats `global_mean_delta`.
7. Don't get stuck on regularization variants — if 3 attempts fail, switch approach families.
8. Attention over genes conditioned on perturbation identity is worth trying.
9. Don't overfit to training perturbations — generalization to unseen perturbations is key.
