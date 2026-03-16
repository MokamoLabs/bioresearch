# AutoMol: Agent Program

## Task
Improve the unified multi-task ADMET prediction model in `train.py`.
The model takes a SMILES string and predicts 22 ADMET endpoints simultaneously.

## What You Can Modify
- `train.py` — model architecture, molecular representations, training loop, loss weighting.

## What You Cannot Modify
- `prepare.py` — frozen evaluation harness and data loading.
- The 22 endpoints, their types, or their clinical importance weights.
- The train/val/test splits.

## Constraints
- Must complete within TIME_BUDGET.
- Must output metrics as JSON on the last line of stdout.
- Can use numpy, scipy, torch, rdkit, sklearn.

## Metric Specifications
- **composite_admet** [PRIMARY]: Clinically-weighted average across all 22 endpoints.
  Higher is better. hERG and DILI get 2x weight (cardiac and liver toxicity).
- **Per-endpoint scores** [GUARD]: No single endpoint may degrade >15% from baseline.
- **gen_profile_match** [PRIMARY2]: For generative models — fraction of generated molecules
  meeting all target ADMET criteria.
- **gen_diversity** [GUARD]: Tanimoto diversity of generated molecules must stay >0.7.

## Architecture Ideas to Explore
1. **Molecular representations**: Morgan fingerprints, ECFP, MACCS keys, learned GNN embeddings,
   SMILES transformers, 3D conformer features.
2. **Multi-task learning**: Shared trunk + task-specific heads, task grouping,
   gradient balancing (GradNorm, uncertainty weighting).
3. **Pre-training**: Self-supervised on ChEMBL 2.4M molecules before fine-tuning.
4. **Uncertainty**: Ensemble, MC dropout, evidential deep learning.
5. **Clinical weighting**: Loss weighting proportional to clinical importance.
6. **Missing data**: Smart imputation, masked loss, auxiliary objectives.

## Key Insight
Nobody has used autonomous search for a *unified* multi-task ADMET architecture.
The composite metric with clinical weights makes gaming impossible — improving
hERG at the expense of clearance lowers the composite score.
