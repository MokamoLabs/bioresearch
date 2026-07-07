"""
Negative-control model — MUTABLE (the agent modifies this like any other domain).

The baseline fits logistic regression on the features. Because the labels are independent
of the features, no model can beat chance in expectation — so this file's purpose is not to
"win" but to let the loop measure how often it keeps a spurious improvement.

Usage: python domains/negative_control/train.py
Outputs metrics as JSON on the last line of stdout.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from domains.negative_control.prepare import load_data, evaluate

SEED = int(os.environ.get("SEED", "42"))
np.random.seed(SEED)


def main():
    t_start = time.time()
    dataset = load_data()

    # Seed-controlled 90% subsample of training data (a little cross-seed jitter).
    rng = np.random.RandomState(SEED)
    n_sub = int(len(dataset.train_idx) * 0.9)
    sub = rng.choice(dataset.train_idx, n_sub, replace=False)

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=200)
    model.fit(dataset.features[sub], dataset.labels[sub])

    # Selection split by default; loop sets EVAL_SPLIT=test for the locked-test eval.
    eval_idx = dataset.test_idx if os.environ.get("EVAL_SPLIT") == "test" else dataset.val_idx
    val_pred = model.predict_proba(dataset.features[eval_idx])[:, 1]
    metrics = evaluate(val_pred, dataset.labels[eval_idx])
    metrics["train_seconds"] = time.time() - t_start
    metrics["peak_vram_mb"] = 0.0
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
