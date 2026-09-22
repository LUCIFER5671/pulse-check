"""Metric helpers and baselines for the heart disease model comparison.

F2 is the decisive metric. This is a screening problem: a missed disease
case (false negative) costs far more than a false alarm, which only leads
to a follow-up test. F2 weights recall 4x as heavily as precision
(beta^2 = 4), so it ranks models the way the clinical costs do.
"""
from functools import partial

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, fbeta_score,
                             precision_score, recall_score)

METRICS = {
    "accuracy": accuracy_score,
    "precision": partial(precision_score, zero_division=0),
    "recall": partial(recall_score, zero_division=0),
    "f1": partial(f1_score, zero_division=0),
    "f2": partial(fbeta_score, beta=2, zero_division=0),
}


def _score(y_true, y_pred):
    return {name: round(float(fn(y_true, y_pred)), 4) for name, fn in METRICS.items()}


def score_model(pipeline, X, y):
    """Score a fitted pipeline on (X, y) with all METRICS."""
    return _score(y, pipeline.predict(X))


def majority_baseline(y_train, y_eval):
    """Always predict the training set's most common class.

    The floor any model has to beat. Positives are the majority here
    (~55%), so this baseline predicts "disease" for everyone and gets
    recall = 1. That gives it a high F2 even though it tells you nothing,
    so a model needs a clearly higher F2 before it counts as useful.
    """
    majority = pd.Series(y_train).mode()[0]
    return _score(y_eval, np.full(len(y_eval), majority))


def random_baseline(y_train, y_eval, n_sims=10000, seed=0):
    """Mean metrics of random guessing at the training-set positive rate.

    Shows what the class balance alone produces with no information about
    the patient. Averaging over many simulations removes the luck of any
    single draw. The metrics are computed with vectorized confusion counts,
    since 10k sklearn calls per metric would be slow. The formulas match
    sklearn's, including zero_division=0.
    """
    y = np.asarray(y_eval).astype(bool)
    rng = np.random.default_rng(seed)
    pred = rng.random((n_sims, len(y))) < np.mean(y_train)

    tp = (pred & y).sum(axis=1)
    fp = (pred & ~y).sum(axis=1)
    fn = (~pred & y).sum(axis=1)

    def ratio(num, den):
        return np.divide(num, den, out=np.zeros(len(num)), where=den > 0)

    sims = {
        "accuracy": (pred == y).mean(axis=1),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "f2": ratio(5 * tp, 5 * tp + 4 * fn + fp),
    }
    return {name: round(float(v.mean()), 4) for name, v in sims.items()}


def results_table(rows, sort_by="f2"):
    """Collect score dicts (each with a 'model' key) into a table ranked by sort_by."""
    df = pd.DataFrame(rows)
    if "model" in df.columns:
        df = df.set_index("model")
    return df.sort_values(sort_by, ascending=False, kind="stable")
