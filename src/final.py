"""Final evaluation: the only file that uses the test set.

The model was chosen from dev results and repeated CV on train, without
looking at test. Everything here is reporting only. Nothing computed in
this file feeds back into model or config choice, so the test numbers stay
honest estimates. The ablations below also score on test, but only to
describe the chosen config, never to pick a different one.

Run from the repo root:  python -m src.final
"""
from ast import literal_eval

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from src.confirm import finalists
from src.data import build_pipeline, load_raw, make_xy, split
from src.evaluate import majority_baseline, random_baseline, score_model
from src.experiments import RESULTS_PATH as DEV_RESULTS_PATH

RESULTS = DEV_RESULTS_PATH.parent
CHOSEN = "SVC"
EXPECTED_PARAMS = {"C": 1, "class_weight": "balanced", "gamma": "scale", "kernel": "rbf"}
FRACTIONS = [0.25, 0.5, 0.75, 1.0]
SEEDS = range(5)


def chosen_model():
    """Chosen config, read from dev_results.csv via the same path as confirm.py.

    The assert catches the case where dev_results.csv was regenerated and a
    different SVC config now tops it. That would silently change what gets
    evaluated on test.
    """
    _, params, est = next(f for f in finalists() if f[0] == CHOSEN)
    assert literal_eval(params) == EXPECTED_PARAMS, f"dev_results.csv changed: {params}"
    return est, params


def test_evaluation(est, params, X_train, y_train, X_test, y_test, cat_cols):
    """Fit on train, score once on test, alongside both baselines."""
    pipe = build_pipeline(clone(est), cat_cols).fit(X_train, y_train)
    rows = [
        {"model": CHOSEN, "params": params, **score_model(pipe, X_test, y_test)},
        {"model": "majority_baseline", "params": "", **majority_baseline(y_train, y_test)},
        {"model": "random_baseline", "params": "n_sims=10000",
         **random_baseline(y_train, y_test)},
    ]
    return pipe, pd.DataFrame(rows)


def data_ablation(est, X_train, y_train, X_dev, y_dev, X_test, y_test, cat_cols):
    """F1 as a function of training-set size: is more data still helping?

    Each fraction of train is drawn as a stratified subsample under 5 seeds
    so the class balance doesn't change with size. Dev and test stay fixed,
    so differences come from training-set size alone. At 100% every seed
    uses the same data and SVC is deterministic, so the std is 0 by
    construction.
    """
    rows = []
    for frac in FRACTIONS:
        for seed in SEEDS:
            if frac < 1.0:
                Xs, _, ys, _ = train_test_split(X_train, y_train, train_size=frac,
                                                stratify=y_train, random_state=seed)
            else:
                Xs, ys = X_train, y_train
            pipe = build_pipeline(clone(est), cat_cols).fit(Xs, ys)
            for split_name, Xe, ye in [("dev", X_dev, y_dev), ("test", X_test, y_test)]:
                rows.append({"fraction": frac, "n_train": len(Xs), "seed": seed,
                             "split": split_name, **score_model(pipe, Xe, ye)})
    raw = pd.DataFrame(rows)
    return (raw.groupby(["fraction", "n_train", "split"])
            .agg(f1_mean=("f1", "mean"), f1_std=("f1", "std"),
                 f2_mean=("f2", "mean"), accuracy_mean=("accuracy", "mean"),
                 precision_mean=("precision", "mean"), recall_mean=("recall", "mean"))
            .round(4).reset_index())


def plot_ablation(summary, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for split_name, marker in [("dev", "o"), ("test", "s")]:
        s = summary[summary["split"] == split_name]
        ax.errorbar(s["n_train"], s["f1_mean"], yerr=s["f1_std"], marker=marker,
                    capsize=4, label=split_name)
    ax.set_xlabel("training rows")
    ax.set_ylabel("F1")
    ax.set_title(f"{CHOSEN} F1 vs training size (mean ± std, 5 seeds)")
    ax.set_xticks(summary["n_train"].unique())
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def site_ablation(est, raw):
    """Refit with and without the source-hospital column, score both on test.

    Site is a confound: prevalence ranges from 36% to 94% by hospital, so
    including it can let the model predict from where a patient was seen
    rather than their clinical signs. make_xy keeps row order and y is
    identical either way, so split() returns the same rows in both arms.
    Only the feature set changes.
    """
    rows, test_indices = [], []
    for drop_site in (True, False):
        X, y, cat_cols = make_xy(raw, drop_site=drop_site)
        X_train, _, X_test, y_train, _, y_test = split(X, y)
        test_indices.append(list(X_test.index))
        pipe = build_pipeline(clone(est), cat_cols).fit(X_train, y_train)
        rows.append({"drop_site": drop_site,
                     "n_features": pipe[0].transform(X_test).shape[1],
                     **score_model(pipe, X_test, y_test)})
    assert test_indices[0] == test_indices[1], "site arms saw different test rows"
    return pd.DataFrame(rows)


def error_analysis(pipe, raw, X_test, y_test):
    """Misclassified test cases with their original, readable feature values.

    Errors are grouped by sex, age decade, chest-pain type, and whether chol
    was originally unmeasured. The chol group checks whether the median
    imputation for the 172 sentinel zeros is where the model breaks down.
    FN and FP are kept separate because a missed disease case and a false
    alarm have very different costs in screening.
    """
    cases = raw.loc[X_test.index].copy()
    cases["y_true"] = y_test.values
    cases["y_pred"] = pipe.predict(X_test)
    cases["error"] = np.select(
        [(cases.y_true == 1) & (cases.y_pred == 0), (cases.y_true == 0) & (cases.y_pred == 1)],
        ["FN", "FP"], default="correct")
    cases["age_decade"] = (cases["age"] // 10 * 10).astype(str) + "s"
    cases["chol_status"] = np.select(
        [cases["chol"].isna(), cases["chol"] == 0],
        ["NaN", "zero (sentinel)"], default="measured")

    breakdowns = {}
    for col in ["sex", "age_decade", "cp", "chol_status"]:
        b = pd.crosstab(cases[col], cases["error"])
        b = b.reindex(columns=["correct", "FN", "FP"], fill_value=0)
        b.insert(0, "n", b.sum(axis=1))
        b["error_rate"] = ((b["FN"] + b["FP"]) / b["n"]).round(3)
        breakdowns[col] = b.drop(columns="correct")
    return cases[cases["error"] != "correct"], breakdowns


def main():
    raw = load_raw()
    X, y, cat_cols = make_xy(raw)
    X_train, X_dev, X_test, y_train, y_dev, y_test = split(X, y)
    est, params = chosen_model()
    pd.set_option("display.width", 200)

    # 1. Test evaluation
    pipe, test_results = test_evaluation(est, params, X_train, y_train,
                                         X_test, y_test, cat_cols)
    test_results.to_csv(RESULTS / "test_results.csv", index=False)
    print(f"== 1. Test set (n={len(X_test)}), model fit on train (n={len(X_train)})")
    print(test_results.drop(columns="params").to_string(index=False))

    # 2. Data-value ablation
    summary = data_ablation(est, X_train, y_train, X_dev, y_dev, X_test, y_test, cat_cols)
    summary.to_csv(RESULTS / "ablation.csv", index=False)
    plot_ablation(summary, RESULTS / "ablation.png")
    print("\n== 2. Training-size ablation (F1 mean / std over 5 seeds)")
    print(summary.pivot(index="n_train", columns="split",
                        values=["f1_mean", "f1_std"]).to_string())

    # 3. Site ablation
    site = site_ablation(est, raw)
    site.to_csv(RESULTS / "site_ablation.csv", index=False)
    print("\n== 3. Site ablation (test)")
    print(site.to_string(index=False))

    # 4. Error analysis
    errors, breakdowns = error_analysis(pipe, raw, X_test, y_test)
    errors.to_csv(RESULTS / "errors.csv", index=False)
    n_fn, n_fp = (errors["error"] == "FN").sum(), (errors["error"] == "FP").sum()
    print(f"\n== 4. Error analysis: {len(errors)} errors on test ({n_fn} FN, {n_fp} FP)")
    for col, b in breakdowns.items():
        print(f"\n-- by {col}")
        print(b.to_string())


if __name__ == "__main__":
    main()
