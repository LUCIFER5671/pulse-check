"""Dev-set model comparison.

Every config is fit on train and scored on dev. The test split is discarded unused
here: it is reserved for one final evaluation of the chosen model, so that
the hyperparameter search can't overfit to it.

F1 is the decisive metric for ranking. All five metrics are recorded so the
trade-offs (e.g. precision vs recall) stay visible.

Run from the repo root:  python -m src.experiments
"""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.data import build_pipeline, load_raw, make_xy, split
from src.evaluate import majority_baseline, random_baseline, results_table, score_model

RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "dev_results.csv"
SORT_METRIC = "f1"

# (family name, base estimator, parameter grid). List-of-dicts grids let SVC
# search gamma only for the rbf kernel, where it has an effect.
MODELS = [
    ("LogisticRegression", LogisticRegression(max_iter=5000), {
        "C": [0.01, 0.1, 1, 10, 100],
        "class_weight": [None, "balanced"],
    }),
    ("SVC", SVC(), [
        {"kernel": ["linear"], "C": [0.1, 1, 10],
         "class_weight": [None, "balanced"]},
        {"kernel": ["rbf"], "C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1],
         "class_weight": [None, "balanced"]},
    ]),
    ("KNN", KNeighborsClassifier(), {
        "n_neighbors": [1, 3, 5, 11, 25, 51],
        "weights": ["uniform", "distance"],
    }),
    ("RandomForest", RandomForestClassifier(random_state=42), {
        "n_estimators": [300],
        "max_depth": [None, 5, 10],
        "min_samples_leaf": [1, 5, 10],
        "class_weight": [None, "balanced"],
    }),
    ("HistGradientBoosting", HistGradientBoostingClassifier(random_state=42), {
        "learning_rate": [0.05, 0.1],
        "max_depth": [None, 3, 6],
        "max_iter": [200],
    }),
    ("SGD", SGDClassifier(max_iter=5000, tol=1e-3, random_state=42), {
        "loss": ["hinge", "log_loss"],
        "alpha": [0.0001, 0.001, 0.01],
        "class_weight": [None, "balanced"],
    }),
]


def run_grid(X_train, y_train, X_dev, y_dev, cat_cols):
    """Fit every config on train and score it on dev. Returns one row per config.

    Each config gets a fresh pipeline from build_pipeline, so imputation and
    scaling are refit on train for every run and never see dev.
    """
    rows = []
    for name, estimator, grid in MODELS:
        for params in ParameterGrid(grid):
            model = estimator.__class__(**{**estimator.get_params(), **params})
            pipe = build_pipeline(model, cat_cols).fit(X_train, y_train)
            rows.append({"model": name, "params": str(params),
                         **score_model(pipe, X_dev, y_dev)})
    return rows


def main():
    X, y, cat_cols = make_xy(load_raw())
    X_train, X_dev, _, y_train, y_dev, _ = split(X, y)  # test deliberately discarded

    rows = run_grid(X_train, y_train, X_dev, y_dev, cat_cols)
    rows.append({"model": "majority_baseline", "params": "",
                 **majority_baseline(y_train, y_dev)})
    rows.append({"model": "random_baseline", "params": "n_sims=10000",
                 **random_baseline(y_train, y_dev)})

    table = results_table(rows, sort_by=SORT_METRIC).reset_index()
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    table.to_csv(RESULTS_PATH, index=False)

    # Table is sorted by F1, so the first row per family is its best config.
    best = table.drop_duplicates("model", keep="first")
    print(f"{len(rows) - 2} configs + 2 baselines -> {RESULTS_PATH}\n")
    print(f"Best dev config per family (ranked by {SORT_METRIC}):")
    with pd.option_context("display.width", 200, "display.max_colwidth", 80):
        print(best.to_string(index=False))


if __name__ == "__main__":
    main()
