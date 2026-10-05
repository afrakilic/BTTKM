"""
Synthetic TT experiment, run in two configurations on the same data:

  A) "updates_on"  : lambda and delta updates ON, (c,d) = (1, 1e-1), (g,h) = (1, 1e-1)
  B) "updates_off" : lambda and delta updates OFF, c = d = g = h = 1
                     (lambdas and deltas stay at their prior mean c/d = g/h = 1)

Usage (from the repo root):
    python synthetic_experiment.py              # run both
    python synthetic_experiment.py --only A     # or B
"""

import os, sys, argparse

sys.path.append(os.getcwd())
from config import *  # Import everything from config.py
from functions.TT_helpers import (generate_tt_cores,
                                  compute_G_left,
                                  pure_power_features_TT)
from functions.BTTKM import bttkm
from functions.utils import (plot_precision_vs_norms,
                             plot_predictive_bounds)

# ----------------------------------------------------------------------
# Data (identical for both configurations)
# ----------------------------------------------------------------------
D = 4
I_true = 3
R_true = 3

Ms = [I_true] * D
Rs = [1] + [R_true] * (D - 1) + [1]
N = 100

rng = np.random.default_rng(42)
X = rng.random((N, D))
W_D_true = 10 * generate_tt_cores(D=D, Ms=Ms, Rs=Rs, seed=1)
Phi = pure_power_features_TT(X, Ms=Ms)
y = compute_G_left(Phi, W_D_true, N) + 1e-3 * rng.standard_normal((N, 1))

# ----------------------------------------------------------------------
# Initialisation / configurations
# ----------------------------------------------------------------------
I_init = 10
R_init = 10

a, b = 1e+2, 1e-1   # tau prior (same in both)

CONFIGS = {
    "A": dict(
        name="updates_on",
        c=1e0, d=1e-1,
        g=1e0, h=1e-1,
        lambda_update=True,
        delta_update=True,
    ),
    "B": dict(
        name="updates_off",
        c=1e0, d=1e0,
        g=1e0, h=1e0,
        lambda_update=False,
        delta_update=False,
    ),
}


def run(cfg):
    print(f"\n{'=' * 70}\nConfiguration: {cfg['name']}  "
          f"(lambda_update={cfg['lambda_update']}, delta_update={cfg['delta_update']}, "
          f"c={cfg['c']}, d={cfg['d']}, g={cfg['g']}, h={cfg['h']})\n{'=' * 70}")

    start_time = time.time()

    # train the model
    model = bttkm(X.shape[1])
    W_D, lambdas, deltas, tau = model.train(
        features=X,
        target=y,
        input_dimension=I_init,
        max_rank=R_init,
        shape_parameter_tau=a,
        scale_parameter_tau=b,
        shape_parameter_lambda=cfg["c"],
        scale_parameter_lambda=cfg["d"],
        shape_parameter_delta=cfg["g"],
        scale_parameter_delta=cfg["h"],
        max_iter=100,
        precision_update=True,
        lambda_update=cfg["lambda_update"],
        delta_update=cfg["delta_update"],
        plot_results=True,
        prune_rank=False,
    )

    # Predict (mse is returned by the predict function)
    prediction_mean, prediction_std, mse = model.predict(
        features=X, input_dimension=I_init, rank=R_init
    )

    prediction_mean_unscaled = prediction_mean
    rmse = np.sqrt(np.mean((prediction_mean_unscaled - y) ** 2))
    print(f"[{cfg['name']}] RMSE: {rmse:.4f}   runtime: {time.time() - start_time:.1f}s")

    plot_predictive_bounds(y, prediction_mean_unscaled, prediction_std)
    plot_precision_vs_norms(W_D, lambdas, deltas)

    return rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(CONFIGS), default=None,
                        help="run a single configuration (A or B)")
    args = parser.parse_args()

    keys = [args.only] if args.only else list(CONFIGS)
    results = {CONFIGS[k]["name"]: run(CONFIGS[k]) for k in keys}

    print("\nSummary (RMSE):")
    for name, r in results.items():
        print(f"  {name:12s} {r:.4f}")