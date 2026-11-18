"""UCI regression: BTT-KM on Airfoil, Energy, Yacht and Concrete.

For each data set and each of 10 random 90/10 splits:
  - train BTT-KM (M = 20, initial TT ranks 5, rank pruning on) and compute RMSE,
  - analytic NLL and 95% coverage from the Student's-t predictive,
  - the same metrics from 3000 Monte Carlo samples of the variational posterior,
  - run time and the average inferred inner TT rank.

Per-run results are saved to results/per_run_results.csv (after every data set).
Print the tables with regression_summary.py.

Usage:  python experiments/regression.py                    (all data sets)
        python experiments/regression.py yacht concrete     (some data sets)
        python experiments/regression.py --no-mc            (skip the Monte Carlo check)
"""

import sys
import time

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from utils import ROOT, load_csv, split, student_t_nll_and_coverage

sys.path.insert(0, str(ROOT))
from bttkm import BTTKM
from bttkm.tt import pure_power_features, contract_left

# n_features: number of input columns (the next column is the target).
# Energy: columns 0-7 are the building features, 8 is heating load (Y1, our target),
# 9 is cooling load (Y2, not used), as in the standard benchmark.
# a0: shape of the Gamma prior on the noise precision (b0 = 1e-3 for all)
DATASETS = {
    "airfoil":  dict(n_features=5, a0=1e-1),
    "energy":   dict(n_features=8, a0=1e0),
    "yacht":    dict(n_features=6, a0=1e0),
    "concrete": dict(n_features=8, a0=1e-1),
}
M, R_MAX, MAX_ITER = 20, 5, 50
N_RUNS, MC_SAMPLES = 10, 3000


def mc_samples(model, X, n_samples, seed):
    """Monte Carlo samples of f and tau from the variational posterior (standardised scale).

    For each sample: draw every TT core from its Gaussian posterior (independently),
    contract with the features, and draw tau ~ Gamma(a_N, rate b_N).
    """
    rng = np.random.default_rng(seed)
    N, D = X.shape
    Phi = pure_power_features(X, [M] * D)
    means = [np.asarray(W) for W in model.W_D]
    factors = []
    for S in model.V:  # L with L L^T = Sigma
        S = 0.5 * (S + S.T)
        w, V = np.linalg.eigh(S)
        factors.append(V * np.sqrt(np.maximum(w, 0.0)))

    f = np.empty((n_samples, N))
    for s in range(n_samples):
        W_s = []
        for W, L in zip(means, factors):
            w = W.ravel(order="F") + L @ rng.standard_normal(L.shape[1])
            W_s.append(w.reshape(W.shape, order="F"))
        f[s] = np.asarray(contract_left(Phi, W_s, N)).ravel()

    a_N, b_N = float(np.squeeze(model.a)), float(np.squeeze(model.b))
    tau = rng.gamma(shape=a_N, scale=1.0 / b_N, size=n_samples)
    return f, tau


def mc_metrics(f, tau, y, y_mean, y_std, seed):
    """Monte Carlo metrics on the original scale, computed as in the BTN-KM code:
    RMSE of the MC mean of f, NLL of the MC predictive (each sample scored with its own
    Gaussian likelihood, averaged with logsumexp) and coverage of the empirical 2.5-97.5% interval.
    """
    y = np.asarray(y).flatten()
    f = f * y_std + y_mean
    sd = y_std / np.sqrt(tau)[:, None]  # noise std of each sample
    rmse = float(np.sqrt(np.mean((f.mean(axis=0) - y) ** 2)))
    log_dens = -0.5 * np.log(2 * np.pi * sd**2) - 0.5 * (y[None, :] - f) ** 2 / sd**2
    nll = float(-np.mean(logsumexp(log_dens, axis=0) - np.log(len(tau))))
    y_s = f + sd * np.random.default_rng(seed).standard_normal(f.shape)
    lower, upper = np.percentile(y_s, 2.5, axis=0), np.percentile(y_s, 97.5, axis=0)
    cov = float(np.mean((y >= lower) & (y <= upper)))
    return rmse, nll, cov


def run_dataset(name, mc=True):
    cfg = DATASETS[name]
    data = load_csv(name)
    X, y = data[:, : cfg["n_features"]], data[:, cfg["n_features"]].reshape(-1, 1)
    rows = []
    for run in range(N_RUNS):
        t0 = time.time()
        X_train, X_test, y_train, y_test = split(X, y, seed=run)
        y_mean, y_std = y_train.mean(), y_train.std()
        y_train = (y_train - y_mean) / y_std

        model = BTTKM().train(
            X_train, y_train, M=M, R=R_MAX,
            a0=cfg["a0"], b0=1e-3, c0=1e-5, d0=1e-6, g0=1e-6, h0=1e-6,
            max_iter=MAX_ITER, prune_rank=True,
        )
        mean, std = model.predict(X_test)
        mean, std = mean * y_std + y_mean, std * y_std

        # Student's-t predictive with nu = 2 * a_N
        nu = float(np.squeeze(2 * model.a))
        nll, coverage = student_t_nll_and_coverage(y_test, mean, std, nu)
        rmse = float(np.sqrt(np.mean((mean - y_test) ** 2)))
        runtime = time.time() - t0  # train + predict + metrics

        row = dict(dataset=name, run=run, rmse=rmse, nll=float(np.mean(nll)), coverage=coverage,
                   nu=nu, runtime_s=runtime, avg_eff_rank=float(model.avg_effective_rank))
        msg = (f"[{name}] run {run}: RMSE={rmse:.4f}  NLL={np.mean(nll):.4f}  cov={coverage * 100:.1f}%  "
               f"R_eff={model.avg_effective_rank:.2f}  ({runtime:.1f}s)")

        if mc:  # Monte Carlo check on the same model and test set
            t1 = time.time()
            f_s, tau_s = mc_samples(model, X_test, MC_SAMPLES, seed=1000 + run)
            mc_rmse, mc_nll, mc_cov = mc_metrics(f_s, tau_s, y_test, y_mean, y_std, seed=2000 + run)
            row.update(mc_rmse=mc_rmse, mc_nll=mc_nll, mc_coverage=mc_cov, mc_runtime_s=time.time() - t1)
            msg += f"\n        MC: RMSE={mc_rmse:.4f}  NLL={mc_nll:.4f}  cov={mc_cov * 100:.1f}%  ({row['mc_runtime_s']:.1f}s)"
        rows.append(row)
        print(msg, flush=True)
    return rows


if __name__ == "__main__":
    args = sys.argv[1:]
    mc = "--no-mc" not in args
    names = [a for a in args if a != "--no-mc"] or list(DATASETS)

    (ROOT / "results").mkdir(exist_ok=True)
    out = ROOT / "results/per_run_results.csv"
    all_rows = []
    for name in names:
        all_rows += run_dataset(name, mc=mc)
        pd.DataFrame(all_rows).to_csv(out, index=False)  # save after every data set
    print(f"\nSaved {out.relative_to(ROOT)} -- now run: python experiments/regression_summary.py")