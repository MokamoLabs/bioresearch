# Prediction Task: Agent Program

## Task
Improve the binary classifier in `train.py`. The model predicts a 0/1 label from a
feature vector. Maximize accuracy on the selection split.

## What You Can Modify
- `train.py` — model, features, training loop, everything.

## What You Cannot Modify
- `prepare.py` — frozen data loading and evaluation.
- The metric or the splits.

## Constraints
- Complete within TIME_BUDGET.
- Output metrics as JSON on the last line of stdout.
- Each experiment SEED is an independent data draw; a change is kept only if it helps
  consistently across seeds.

## Metric Specifications
- **accuracy** [PRIMARY]: fraction of correctly classified samples. Higher is better.

## Ideas to Explore
- Regularization strength, feature scaling, nonlinear models, ensembles, calibration.
