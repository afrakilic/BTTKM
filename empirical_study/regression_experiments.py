"""
Runs all BTTKM experiments (airfoil, energy, yacht, concrete) and saves the results.

Usage (from the repo root, where config.py, data/ and functions/ live):
    python experiments.py                         # all datasets
    python experiments.py --datasets yacht concrete
    python experiments.py --plot                  # also show the diagnostic plots
    python experiments.py --mc-samples 2000       # MC samples per test set (default 1000)
    python experiments.py --no-mc                 # skip Monte Carlo evaluation

Outputs (in results/):
    per_run_results.csv   one row per (dataset, seed): analytic + MC RMSE, NLL, coverage, runtime,
                          and the average effective TT rank (avg_eff_rank)
    summary.json          mean/std per dataset (+ hyperparameters)
    raw/<dataset>.npz     last-run predictions, W_D, lambdas, deltas (for plotting)

Tables are built from these files by results.py.

Requires bttkm.train() to return 5 values: (W_D, lambdas, deltas, tau, avg_effective_rank), the
mean of the D-1 inner bond ranks after pruning (boundary ranks R_1 = R_{D+1} = 1 excluded).
"""

import os, sys, json, argparse

sys.path.append(os.getcwd())
from config import *  # noqa: F401,F403  (provides np, pd, time, ...)
from functions.BTTKM import bttkm
from functions.TT_helpers import compute_G_left, pure_power_features_TT
from functions.utils import plot_predictive_bounds, plot_precision_vs_norms
from scipy import stats

# --------------------------------------------------------------------------
# Dataset configurations (hyperparameters taken from the original scripts)
# a,b: tau (shape, scale) | c,d: lambda (shape, scale) | g,h: delta (shape, scale)
# --------------------------------------------------------------------------
COMMON = dict(input_dimension=20, max_rank=5, max_iter=50, g=1e-1, h=1e-1)

DATASETS = {
    "airfoil":  dict(path="data/airfoil.csv",  n_features=5, a=1e1, b=1e-1, c=1e0,  d=1e-1),
    "energy":   dict(path="data/energy.csv",   n_features=9, a=1e2, b=1e-1, c=1e0,  d=1e-1),
    "yacht":    dict(path="data/yacht.csv",    n_features=6, a=1e2, b=1e-1, c=1e0,  d=1e-1),
    "concrete": dict(path="data/concrete.csv", n_features=8, a=1e1, b=1e-1, c=1e0,  d=1e-1),
}

N_RUNS = 10
TRAIN_FRAC = 0.90
CONF_LEVEL = 0.95
N_STD = 1.959963985   # Gaussian 95% half-width, used for MC coverage
MC_SAMPLES = 3000
RESULTS_DIR = "results"


def student_t_nll_and_coverage(y_true, mean, std, nu, conf_level=0.95):
    """
    Exact Student's-t NLL and empirical coverage of the nominal `conf_level`
    predictive interval, for y_i | y ~ T(mean_i, scale_i, nu), where
        std = sqrt(nu/(nu-2)) * scale  (std as returned by model.predict).
    """
    y_true = np.asarray(y_true).flatten()
    mean = np.asarray(mean).flatten()
    std = np.asarray(std).flatten()

    if nu <= 2:
        raise ValueError("nu must be > 2 for the variance (and hence std) to be defined.")

    scale = std / np.sqrt(nu / (nu - 2))
    nll_per_point = -stats.t.logpdf(y_true, df=nu, loc=mean, scale=scale)

    alpha = 1 - conf_level
    t_crit = stats.t.ppf(1 - alpha / 2, df=nu)
    inside = (y_true >= mean - t_crit * scale) & (y_true <= mean + t_crit * scale)
    return nll_per_point, float(np.mean(inside))


def _psd_factor(S):
    """Return L with L @ L.T = S for a (numerically) PSD covariance S."""
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    return V * np.sqrt(np.maximum(w, 0.0))


def mc_predictive_samples(model, X, input_dimension, n_samples=3000, seed=0):
    """
    Monte Carlo predictive samples from the BTTKM variational posterior
    (standardised target scale).

    For each sample s:
        1. draw every TT core  vec(W_d) ~ N(vec(mean_d), Sigma_d)   (independent across cores)
        2. f_s = Phi(x) contracted with the sampled cores
        3. draw tau_s ~ Gamma(a_N, rate=b_N) and y_s = f_s + eps,  eps ~ N(0, 1/tau_s)

    Cores are vectorised in Fortran order, matching how Sigma is defined in
    bttkm.train(). Returns y_samples, f_samples of shape (n_samples, N_test).
    """
    rng = np.random.default_rng(seed)
    N, D = X.shape
    Phi = pure_power_features_TT(X, Ms=[input_dimension] * D)

    means = [np.asarray(W) for W in model.W_D]
    factors = [_psd_factor(np.asarray(S)) for S in model.V]
    a_N = float(np.squeeze(model.a))
    b_N = float(np.squeeze(model.b))

    f = np.empty((n_samples, N))
    for s in range(n_samples):
        W_s = []
        for W, L in zip(means, factors):
            w = W.ravel(order="F") + L @ rng.standard_normal(L.shape[1])
            W_s.append(w.reshape(W.shape, order="F"))
        f[s] = np.asarray(compute_G_left(Phi, W_s, N)).ravel()

    tau_s = rng.gamma(shape=a_N, scale=1.0 / b_N, size=n_samples)
    y = f + rng.standard_normal((n_samples, N)) / np.sqrt(tau_s)[:, None]
    return y, f


def mc_metrics(y_samples, y_true):
    """RMSE of the MC mean, Gaussian NLL from sample moments, and mean +/- 1.96 std coverage."""
    y_true = np.asarray(y_true).flatten()
    mu = y_samples.mean(axis=0)
    sd = y_samples.std(axis=0, ddof=1)
    rmse = float(np.sqrt(np.mean((mu - y_true) ** 2)))
    nll = float(np.mean(0.5 * np.log(2 * np.pi * sd**2) + 0.5 * (y_true - mu) ** 2 / sd**2))
    cov = float(np.mean((y_true >= mu - N_STD * sd) & (y_true <= mu + N_STD * sd)))
    return rmse, nll, cov, mu, sd


def load_dataset(cfg):
    df = pd.read_csv(cfg["path"], header=None)
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True).values.astype(float)
    nf = cfg["n_features"]
    return df[:, :nf], df[:, nf].reshape(-1, 1)


def run_dataset(name, cfg, plot=False, mc_samples=MC_SAMPLES):
    hp = {**COMMON, **cfg}
    X, y = load_dataset(cfg)
    rows = []
    last = {}

    for i in range(N_RUNS):
        t0 = time.time()

        # split (seeded by run index, 90/10)
        np.random.seed(i)
        idx = np.random.permutation(len(X))
        split = int(TRAIN_FRAC * len(X))
        X_train, X_test = X[idx[:split]], X[idx[split:]]
        y_train, y_test = y[idx[:split]], y[idx[split:]]

        # standardise with training statistics only
        X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
        X_std[X_std == 0] = 1
        X_train = (X_train - X_mean) / X_std
        X_test = (X_test - X_mean) / X_std
        y_mean, y_std = y_train.mean(), y_train.std()
        y_train = (y_train - y_mean) / y_std

        # train
        model = bttkm(X_train.shape[1])
        W_D, lambdas, deltas, tau, avg_eff_rank = model.train(
            features=X_train,
            target=y_train,
            input_dimension=hp["input_dimension"],
            max_rank=hp["max_rank"],
            shape_parameter_tau=hp["a"],
            scale_parameter_tau=hp["b"],
            shape_parameter_lambda=hp["c"],
            scale_parameter_lambda=hp["d"],
            shape_parameter_delta=hp["g"],
            scale_parameter_delta=hp["h"],
            max_iter=hp["max_iter"],
            precision_update=True,
            lambda_update=True,
            delta_update=True,
            plot_results=False,
            prune_rank=True,
        )

        # predict
        pred_mean, pred_std, mse = model.predict(
            features=X_test, input_dimension=hp["input_dimension"], rank=hp["max_rank"]
        )
        pred_mean_u = pred_mean * y_std + y_mean
        pred_std_u = pred_std * y_std

        rmse = float(np.sqrt(np.mean((pred_mean_u - y_test) ** 2)))

        # Student-t predictive distribution, nu_y = 2 * a_N (model.a is set by train())
        nu_y = float(np.squeeze(2 * model.a))
        nll_pp, coverage = student_t_nll_and_coverage(
            y_test, pred_mean_u, pred_std_u, nu_y, CONF_LEVEL
        )

        runtime = time.time() - t0   # analytic pipeline only (train + predict + metrics)

        row = dict(dataset=name, run=i, rmse=rmse, nll=float(np.mean(nll_pp)),
                   coverage=coverage, nu=nu_y, runtime_s=runtime,
                   avg_eff_rank=float(avg_eff_rank))
        msg = (f"[{name}] run {i}: RMSE={rmse:.4f}  NLL={np.mean(nll_pp):.4f}  "
               f"cov={coverage*100:.1f}%  R_eff={avg_eff_rank:.2f}  ({runtime:.1f}s)")

        # Monte Carlo predictive evaluation (same trained model, same test split)
        if mc_samples > 0:
            t1 = time.time()
            y_s, _ = mc_predictive_samples(model, X_test, hp["input_dimension"],
                                           n_samples=mc_samples, seed=1000 + i)
            y_s = y_s * y_std + y_mean                      # back to original scale
            mc_rmse, mc_nll, mc_cov, mc_mu, mc_sd = mc_metrics(y_s, y_test)
            row.update(mc_rmse=mc_rmse, mc_nll=mc_nll, mc_coverage=mc_cov,
                       mc_runtime_s=time.time() - t1)
            msg += (f"\n        MC({mc_samples}): RMSE={mc_rmse:.4f}  NLL={mc_nll:.4f}  "
                    f"cov={mc_cov*100:.1f}%  ({row['mc_runtime_s']:.1f}s)")
            last_mc = dict(mc_mean=mc_mu, mc_std=mc_sd)
        else:
            last_mc = {}

        rows.append(row)
        print(msg)

        last = dict(y_test=y_test, pred_mean=pred_mean_u, pred_std=pred_std_u,
                    W_D=np.asarray(W_D, dtype=object), lambdas=np.asarray(lambdas, dtype=object),
                    deltas=np.asarray(deltas, dtype=object), **last_mc)

    if plot:
        plot_predictive_bounds(last["y_test"], last["pred_mean"], last["pred_std"])
        plot_precision_vs_norms(W_D, lambdas, deltas)

    return rows, last, hp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--plot", action="store_true", help="show diagnostic plots")
    parser.add_argument("--mc-samples", type=int, default=MC_SAMPLES,
                        help="Monte Carlo samples per test set (default %(default)s)")
    parser.add_argument("--no-mc", action="store_true", help="skip Monte Carlo evaluation")
    args = parser.parse_args()

    os.makedirs(os.path.join(RESULTS_DIR, "raw"), exist_ok=True)
    all_rows, summary = [], {}

    for name in args.datasets:
        rows, last, hp = run_dataset(name, DATASETS[name], plot=args.plot,
                                   mc_samples=0 if args.no_mc else args.mc_samples)
        all_rows += rows
        d = pd.DataFrame(rows)
        summary[name] = {
            "hyperparameters": {k: v for k, v in hp.items() if k != "path"},
            "n_runs": N_RUNS,
            "total_runtime_s": float(d.runtime_s.sum()),
            **{f"{m}_{s}": float(fn(d[m].values))
               for m in [c for c in ("rmse", "nll", "coverage", "mc_rmse", "mc_nll", "mc_coverage",
                                     "avg_eff_rank")
                         if c in d.columns]
               for s, fn in (("mean", np.mean), ("std", np.std))},  # np.std: ddof=0
        }
        np.savez(os.path.join(RESULTS_DIR, "raw", f"{name}.npz"), **last)

        # save after every dataset so partial progress isn't lost
        pd.DataFrame(all_rows).to_csv(os.path.join(RESULTS_DIR, "per_run_results.csv"), index=False)
        with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    print(f"\nSaved results to {RESULTS_DIR}/ -- now run: python results.py")


if __name__ == "__main__":
    main()