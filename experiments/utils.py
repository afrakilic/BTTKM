"""Data loading, train/test split and metrics shared by the experiments."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


def load_csv(name):
    """Load data/<name>.csv as a float array. The first row is skipped only if it is a header
    (not numeric); energy.csv, for example, has no header."""
    df = pd.read_csv(ROOT / "data" / f"{name}.csv", header=None, low_memory=False)
    first_is_header = pd.to_numeric(df.iloc[0], errors="coerce").isna().any()
    return (df.iloc[1:] if first_is_header else df).values.astype(float)


def split(X, y, seed, train_frac=0.90):
    """Random train/test split (seeded) and input standardisation with training statistics."""
    np.random.seed(seed)
    idx = np.random.permutation(len(X))
    n_train = int(train_frac * len(X))
    X_train, X_test = X[idx[:n_train]], X[idx[n_train:]]
    y_train, y_test = y[idx[:n_train]], y[idx[n_train:]]

    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
    X_std[X_std == 0] = 1
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    return X_train, X_test, y_train, y_test


def student_t_nll_and_coverage(y, mean, std, nu, level=0.95):
    """Per-point Student's-t NLL and the coverage of the central `level` interval.

    `std` is the predictive standard deviation, so the t scale is std / sqrt(nu / (nu - 2)).
    """
    y, mean, std = np.asarray(y).flatten(), np.asarray(mean).flatten(), np.asarray(std).flatten()
    if nu <= 2:
        raise ValueError("nu must be > 2 for the variance (and hence std) to be defined.")
    scale = std / np.sqrt(nu / (nu - 2))
    nll = -stats.t.logpdf(y, df=nu, loc=mean, scale=scale)
    t_crit = stats.t.ppf(1 - (1 - level) / 2, df=nu)
    inside = (y >= mean - t_crit * scale) & (y <= mean + t_crit * scale)
    return nll, float(np.mean(inside))


def classification_nll(y, mean, std, eps=1e-15):
    """Binary NLL for labels y in {-1, +1}, using P(y = +1) = P(f > 0) under N(mean, std^2)."""
    p = np.clip(stats.norm.sf(0, loc=np.ravel(mean), scale=np.ravel(std)), eps, 1 - eps)
    y01 = (y + 1) // 2
    return -np.mean(y01 * np.log(p) + (1 - y01) * np.log(1 - p))