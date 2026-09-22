"""Settle the dev-set near-tie with repeated cross-validation on train.

The best config from each family landed within 0.016 F1 of each other on a
184-row dev set. One changed prediction moves F1 by about 0.005 there, so
that ranking is mostly noise. Here each finalist is re-scored with 5-fold CV
repeated 6 times (30 folds) on the training split only. All models get the
same folds, so their per-fold scores can be paired: comparing each fold's
difference cancels out easy/hard fold effects that are shared by every model.

Dev and test are discarded unused. Run from the repo root:
    python -m src.confirm
"""
from ast import literal_eval
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from src.data import build_pipeline, load_raw, make_xy, split
from src.evaluate import METRICS
from src.experiments import MODELS, RESULTS_PATH as DEV_RESULTS_PATH

CV_RESULTS_PATH = DEV_RESULTS_PATH.with_name("cv_results.csv")
SCORING = {name: make_scorer(fn) for name, fn in METRICS.items()}
BASELINE = "majority_baseline"


def finalists():
    """Best dev config per family, plus the majority baseline, as (name, params, estimator)."""
    dev = pd.read_csv(DEV_RESULTS_PATH, keep_default_na=False)
    base = {name: est for name, est, _ in MODELS}
    best = (dev[dev["model"].isin(base)]
            .sort_values("f1", ascending=False, kind="stable")
            .drop_duplicates("model"))
    out = []
    for name, params in zip(best["model"], best["params"]):
        est = base[name]
        out.append((name, params, est.__class__(**{**est.get_params(), **literal_eval(params)})))
    out.append((BASELINE, "most_frequent", DummyClassifier(strategy="most_frequent")))
    return out


def main():
    X, y, cat_cols = make_xy(load_raw())
    X_train, _, _, y_train, _, _ = split(X, y)  # dev and test deliberately discarded

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=6, random_state=42)
    folds = list(cv.split(X_train, y_train))  # materialized so every model sees identical folds

    rows, fold_f1 = [], {}
    for name, params, est in finalists():
        scores = cross_validate(build_pipeline(est, cat_cols), X_train, y_train,
                                cv=folds, scoring=SCORING, n_jobs=-1)
        fold_f1[name] = pd.Series(scores["test_f1"])
        row = {"model": name, "params": params}
        for m in METRICS:
            row[f"{m}_mean"] = round(scores[f"test_{m}"].mean(), 4)
            row[f"{m}_std"] = round(scores[f"test_{m}"].std(ddof=1), 4)
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("f1_mean", ascending=False, kind="stable")
    table.to_csv(CV_RESULTS_PATH, index=False)

    print(f"{len(folds)}-fold repeated CV on train (n={len(X_train)}) -> {CV_RESULTS_PATH}\n")
    print(f"{'model':<22} F1 mean +/- std")
    for _, r in table.iterrows():
        print(f"{r['model']:<22} {r['f1_mean']:.4f} +/- {r['f1_std']:.4f}")

    # Paired per-fold differences: top model minus each other model.
    top = table["model"].iloc[0]
    print(f"\nPaired per-fold F1 difference, {top} minus each model:")
    print(f"{'vs':<22} {'mean diff':>9} {'std':>7}  verdict")
    for other in table["model"].iloc[1:]:
        d = fold_f1[top] - fold_f1[other]
        verdict = "separable" if d.mean() >= d.std(ddof=1) else "NOT separable"
        print(f"{other:<22} {d.mean():>9.4f} {d.std(ddof=1):>7.4f}  {verdict}")


if __name__ == "__main__":
    main()
