"""Data loading, cleaning, splitting, and preprocessing for UCI heart disease.

Design rule: anything that *learns* from data (imputation medians/modes,
scaling means, one-hot categories) lives inside the sklearn Pipeline so it is
fit on the training split only. make_xy does only row-independent fixes
(sentinel -> NaN, bool -> 0/1) that leak nothing across splits.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM_COLS = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
CAT_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
SITE_COL = "dataset"
ZERO_SENTINEL_COLS = ["chol", "trestbps"]
BOOL_COLS = ["fbs", "exang"]

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "heart_disease_uci.csv"


def load_raw(path=DEFAULT_PATH):
    """Read the CSV untouched.

    Kept separate from cleaning so the raw frame is always available for
    EDA (e.g. counting the chol==0 sentinels before they are converted).
    """
    return pd.read_csv(path)


def make_xy(df, drop_site=True):
    """Split a raw frame into features X, binary target y, and cat_cols.

    - Target: num is 0-4 severity; we predict presence (num > 0), which is
      the standard framing and keeps classes roughly balanced.
    - chol/trestbps == 0 means "not measured" (a cholesterol of 0 is
      impossible). Left as 0, the median imputer would never see them as
      missing and the scaler would treat them as extreme low values, so they
      become NaN here, before any imputation happens.
    - fbs/exang become 1/0 with NaN preserved, so "unknown" is not silently
      merged with False at this stage.
    - The site column is dropped by default: disease prevalence ranges from
      ~36% (Hungary) to ~94% (Switzerland), so a model can score well by
      learning which hospital a row came from rather than anything about
      the patient. drop_site=False keeps it (as a categorical) for
      comparison experiments.
    """
    y = (df["num"] > 0).astype(int)
    X = df.drop(columns=["id", "num"]).copy()

    for col in ZERO_SENTINEL_COLS:
        X[col] = X[col].replace(0, np.nan)

    # pandas may parse TRUE/FALSE as bools or leave them as strings; handle both.
    bool_map = {True: 1.0, False: 0.0, "TRUE": 1.0, "FALSE": 0.0}
    for col in BOOL_COLS:
        X[col] = X[col].map(bool_map)  # unmapped values (NaN) stay NaN

    cat_cols = list(CAT_COLS)
    if drop_site:
        X = X.drop(columns=[SITE_COL])
    else:
        cat_cols.append(SITE_COL)

    return X, y, cat_cols


def split(X, y, seed=42):
    """Stratified 60/20/20 train/dev/test split.

    Stratifying on y keeps the disease rate equal across splits, which
    matters with only 920 rows: an unlucky split could otherwise shift the
    base rate enough to distort dev/test metrics. Dev is for model selection;
    test is touched once at the end.
    """
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed
    )
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=seed
    )
    return X_train, X_dev, X_test, y_train, y_dev, y_test


def build_preprocessor(cat_cols, add_indicator=True):
    """ColumnTransformer for numeric + categorical columns.

    Numeric: median imputation (robust to the skew in chol/oldpeak), then
    standardization so linear models and distance-based models see comparable
    scales. add_indicator appends a 0/1 "was missing" column per feature with
    NaNs; missingness here is not random -- it tracks which hospital measured
    what (e.g. Switzerland has almost no chol), so the indicator carries
    signal that plain imputation would erase.

    Categorical: mode imputation, then one-hot. handle_unknown='ignore'
    avoids crashes if a rare category appears only in dev/test;
    drop='if_binary' removes the redundant column for two-level features.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=add_indicator)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUM_COLS),
        ("cat", categorical, list(cat_cols)),
    ])


def build_pipeline(model, cat_cols, add_indicator=True):
    """Preprocessor + model as one estimator.

    Bundling them means fit() learns imputation/scaling statistics from the
    training data only, and cross-validation refits them per fold, so no
    information from dev/test (or held-out folds) leaks into preprocessing.
    """
    return Pipeline([
        ("prep", build_preprocessor(cat_cols, add_indicator=add_indicator)),
        ("model", model),
    ])


if __name__ == "__main__":
    df = load_raw()
    X, y, cat_cols = make_xy(df)
    X_train, X_dev, X_test, y_train, y_dev, y_test = split(X, y)

    print(f"train: {len(X_train)}  dev: {len(X_dev)}  test: {len(X_test)}")
    print(f"chol NaN rate after sentinel fix: {X['chol'].isna().mean():.1%} "
          f"({X['chol'].isna().sum()}/{len(X)})")
