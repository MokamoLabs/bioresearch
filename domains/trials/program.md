# AutoTrial: Agent Program

## Task
Improve the clinical trial success prediction model in `train.py`.
Given drug properties, targets, indication, and trial design, predict success probability.

## What You Can Modify
- `train.py` — model architecture, feature engineering, training loop.

## What You Cannot Modify
- `prepare.py` — frozen evaluation with economic metrics.
- The temporal data split (prevents data leakage).

## Constraints
- Must complete within TIME_BUDGET.
- Must output metrics as JSON on the last line of stdout.
- Can use numpy, scipy, torch, sklearn, xgboost.

## Metric Specifications
- **auroc** [PRIMARY]: Discrimination — can the model rank trials by success probability?
- **calibration_ece** [GUARD]: Must stay <0.15. Predictions must be trustworthy probability
  estimates, not just rankings.
- **net_value** [GUARD]: Must be >0. If you followed this model's "go/no-go" advice,
  would you make money? (success = +$1B, failure = -$200M, trial cost = $50M)
- **lift_at_10** [BONUS]: Are successes concentrated in the top 10% of predictions?
- **phase_stratified_auroc** [DIAGNOSTIC]: Per-phase performance breakdown.

## Architecture Ideas
1. **Feature engineering**: Drug fingerprints, target embeddings, indication embeddings,
   historical success rates per indication/target.
2. **Cross-project features**: ADMET profiles from AutoMol, perturbation signatures from AutoPerturb.
3. **Calibration**: Platt scaling, isotonic regression, temperature scaling.
4. **Ensembling**: XGBoost + neural net + logistic regression ensemble.
5. **Multi-phase modeling**: Separate models per phase vs. unified with phase as feature.
6. **Temporal modeling**: Account for time trends in trial success rates.

## The Chain (What Makes This Insane)
This model can consume outputs from AutoPerturb and AutoMol:
- AutoMol provides ADMET profiles → drug quality features
- AutoPerturb provides perturbation signatures → mechanism of action features
- Combined: molecule → cellular effect → patient outcome prediction
