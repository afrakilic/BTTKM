"""UCI classification: BTT-KM on Adult and Spambase.

Labels are mapped to {-1, +1}, the model is fitted as a regression and the
predicted class is the sign of the predictive mean. Reports the
misclassification rate, NLL and average inferred inner TT rank over 10
random 90/10 splits. A split whose core update cannot be inverted is skipped.

Usage:  python experiments/classification.py              (Adult, then Spambase)
        python experiments/classification.py spambase     (~15 min)
        python experiments/classification.py adult        (~1.5 h)
"""

import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score

from utils import ROOT, load_csv, split, classification_nll

sys.path.insert(0, str(ROOT))
from bttkm import BTTKM

# n_features: number of input columns (the next column is the label)
# Run order when no argument is given: Adult first, then Spambase.
DATASETS = {
    "adult":    dict(n_features=96, M=40, R=3, a0=1e0, c0=1e-6),
    "spambase": dict(n_features=57, M=30, R=5, a0=1e-1, c0=1e-5),
}


def run_dataset(name):
    cfg = DATASETS[name]
    data = load_csv(name)
    X = data[:, : cfg["n_features"]]
    y = np.where(data[:, cfg["n_features"]] > 0, 1, -1)

    print(f"\n=== {name} ===", flush=True)
    errors, nlls, ranks = [], [], []
    run_times = []  # time of each completed run (s)
    start = time.time()
    for run in range(10):
        t0 = time.time()
        X_train, X_test, y_train, y_test = split(X, y, seed=run)
        try:
            model = BTTKM().train(
                X_train, y_train, M=cfg["M"], R=cfg["R"],
                a0=cfg["a0"], b0=1e-3, c0=cfg["c0"], d0=1e-6, g0=1e-6, h0=1e-6,
                max_iter=50, prune_rank=True,
            )
        except RuntimeError as e:
            if "INVERSION_FAILED" in str(e):
                print(f"  run {run}: skipped ({e})", flush=True)
                continue
            raise
        mean, std = model.predict(X_test)
        errors.append(1 - accuracy_score(y_test, np.sign(mean)))
        nlls.append(classification_nll(y_test, mean, std))
        ranks.append(model.avg_effective_rank)
        run_times.append(time.time() - t0)
        print(f"  run {run}: error={errors[-1]:.4f}  NLL={nlls[-1]:.4f}  R_eff={ranks[-1]:.2f}"
              f"  time={run_times[-1]:.2f} s", flush=True)

    return dict(
        runs=len(errors),
        runtime=time.time() - start,
        error=(np.mean(errors), np.std(errors)),
        nll=(np.mean(nlls), np.std(nlls)),
        rank=(np.mean(ranks), np.std(ranks)),
        run_time=(np.mean(run_times), np.std(run_times)),
    )


names = sys.argv[1:] or list(DATASETS)  # no argument: run all data sets
results = {name: run_dataset(name) for name in names}

print("\nCLASSIFICATION RESULTS (BTT-KM)")
for name, r in results.items():
    print(f"\n{name}: {r['runs']} runs, total runtime {r['runtime']:.2f} s")
    print(f"Misclassification: {r['error'][0]:.4f} ± {r['error'][1]:.4f}")
    print(f"NLL:               {r['nll'][0]:.4f} ± {r['nll'][1]:.4f}")
    print(f"Effective R:       {r['rank'][0]:.2f} ± {r['rank'][1]:.2f}")
    print(f"Time per run:      {r['run_time'][0]:.2f} ± {r['run_time'][1]:.2f} s")


# spambase: 10 runs, total runtime 2665.48 s
# Misclassification: 0.0941 ± 0.0357
# NLL:               0.2571 ± 0.0447
# Effective R:       2.83 ± 0.20
# Time per run:      266.55 ± 12.58 s


# adult: 10 runs, total runtime 24462.38 s
# Misclassification: 0.1446 ± 0.0051
# NLL:               0.3293 ± 0.0079
# Effective R:       3.00 ± 0.00
# Time per run:      2446.24 ± 69.06 s