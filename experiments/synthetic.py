"""Section 4: effect of the rank and feature precisions on synthetic data (Figures 3 and 4).

Ground truth: D = 4 inputs, TT ranks 3 and 3 active features per core (out of 5),
cores drawn from N(0, 1) and scaled by 10, N = 100 samples, noise std 1e-3.
The model starts from R = M = 5 and runs 100 iterations without rank pruning.

Prints the RMSE, the variance explained (%) per rank / feature component and
the posterior mean precisions, and shows:
  - the variational lower bound,
  - the predictions with +-2 std bounds,
  - Figure 3: variance explained per rank component with log lambda_d,
  - Figure 4: variance explained per feature with log delta_d.
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bttkm import BTTKM
from bttkm.tt import random_tt_cores, contract_left, pure_power_features

# ---- Ground truth ----
D, M_TRUE, R_TRUE, N = 4, 3, 3, 100
M_INIT, R_INIT = 5, 5

rng = np.random.default_rng(42)
X = rng.standard_normal((N, D))
Rs_true = [1] + [R_TRUE] * (D - 1) + [1]
W_small = random_tt_cores([M_TRUE] * D, Rs_true, seed=1)
W_true = [np.zeros((Rs_true[d], M_INIT, Rs_true[d + 1])) for d in range(D)]
for d in range(D):
    W_true[d][:, :M_TRUE, :] = 10 * W_small[d]  # only the first 3 features are active
Phi = pure_power_features(X, [M_INIT] * D)
y = contract_left(Phi, W_true, N) + 1e-3 * rng.standard_normal((N, 1))

# ---- Train ----
start = time.time()
model = BTTKM().train(
    X, y, M=M_INIT, R=R_INIT,
    a0=1e-3, b0=1e-3, c0=1e-6, d0=1e-6, g0=1e-6, h0=1e-6,
    max_iter=100, lower_bound_tol=1e-10, prune_rank=False,
)
mean, std = model.predict(X)
rmse = np.sqrt(np.mean((mean - y) ** 2))
print(f"\nRMSE: {rmse:.4f}   runtime: {time.time() - start:.1f}s")

W, lambdas, deltas = model.W_D, model.lambdas, model.deltas


def rank_variance_explained(d):
    """% variance explained per component of TT rank R_{d+1} (cores d-1 and d), as used for pruning."""
    p = np.sum(W[d - 1] ** 2, axis=(0, 1)) + np.sum(W[d] ** 2, axis=(1, 2))
    return p / np.sum(p) * 100


def feature_variance_explained(d):
    """% variance explained per feature (lateral slice) of core d."""
    q = np.sum(W[d] ** 2, axis=(0, 2))
    return q / np.sum(q) * 100


# ---- Tables ----
def table(columns, index_name, fmt):
    df = pd.DataFrame({k: pd.Series(v) for k, v in columns.items()})
    df.index = df.index + 1
    df.index.name = index_name
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(df.to_string(float_format=fmt))


print("\nVariance explained per rank component (%)")
table({f"lambda_{d + 1}": rank_variance_explained(d) for d in range(1, D)}, "rank index", lambda x: f"{x:.17g}")
print("\nVariance explained per feature (%)")
table({f"delta_{d + 1}": feature_variance_explained(d) for d in range(D)}, "feature index", lambda x: f"{x:.17g}")
print("\nPosterior mean rank precisions E[lambda_d]")
table({f"lambda_{d + 1}": np.ravel(lambdas[d]) for d in range(1, D)}, "rank index", lambda x: f"{x:.3e}")
print("\nPosterior mean feature precisions E[delta_d]")
table({f"delta_{d + 1}": np.ravel(deltas[d]) for d in range(D)}, "feature index", lambda x: f"{x:.3e}")

# ---- Figures ----
GREY, BLACK = "#8C8C8C", "#000000"
STYLE = {
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 13,
    "axes.titlesize": 15, "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 12, "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
}

with plt.rc_context(STYLE):
    # Lower bound
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(np.arange(1, len(model.lower_bound) + 1), model.lower_bound, color="#222222", linewidth=1.0, marker="o", markersize=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Lower bound")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    # Predictions with +-2 std, sorted by y
    y_true = y.ravel()
    order = np.argsort(y_true)
    n = np.arange(N)
    pred, sd = mean.ravel()[order], std.ravel()[order]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.fill_between(n, pred - 2 * sd, pred + 2 * sd, color=GREY, alpha=0.2, linewidth=0, label=r"$\pm 2\sigma$")
    ax.scatter(n, y_true[order], color="black", s=6, zorder=3, label=r"$y$")
    ax.scatter(n, pred, facecolor="white", edgecolor=BLACK, s=10, linewidth=0.7, zorder=4, label=r"$\hat{y}$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$y$")
    ax.legend(frameon=False, loc="upper left")
    plt.tight_layout()


def bar_figure(ve, logp, titles, xlabel, prec_label):
    """One panel per TT rank / core: variance explained (bars, left axis) and log precisions (line, right axis)."""
    P = len(titles)
    ve_max = min(100, np.nanmax([np.nanmax(v) for v in ve]) * 1.1)
    all_p = np.concatenate([np.ravel(p) for p in logp])
    p_lim = (np.nanmin(all_p) - 1, np.nanmax(all_p) + 1)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, P, figsize=(2.6 * P, 2.4))
        for i, ax in enumerate(np.atleast_1d(axes)):
            x = np.arange(len(ve[i]))
            ax2 = ax.twinx()
            ax2.set_zorder(ax.get_zorder() + 1)
            ax2.patch.set_visible(False)
            ax.bar(x, ve[i], width=0.5, color=GREY, alpha=0.4, linewidth=0)
            ax2.plot(x, logp[i], color=BLACK, linestyle="-", marker="o", markersize=5, linewidth=1.6,
                     markeredgecolor="white", markeredgewidth=0.5)
            ax.set_ylim(0, ve_max)
            ax2.set_ylim(*p_lim)
            ax.set_xticks(x)
            ax.set_xticklabels([str(k + 1) for k in x])
            ax.set_title(titles[i], pad=4)
            first, last = i == 0, i == P - 1
            ax.spines[["top", "right"]].set_visible(False)
            ax2.spines["top"].set_visible(False)
            ax2.spines["left"].set_visible(False)
            ax2.spines["right"].set_visible(last)
            if first:
                ax.set_ylabel(r"$\%$")
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", length=0)
            if last:
                ax2.set_ylabel(prec_label)
            else:
                ax2.set_yticklabels([])
                ax2.tick_params(axis="y", length=0)
        plt.tight_layout(rect=(0, 0.07, 1, 1))
        fig.supxlabel(xlabel, y=0.02, fontsize=14)


# Figure 3: rank components
bar_figure(
    ve=[rank_variance_explained(d) for d in range(1, D)],
    logp=[np.log(lambdas[d]) for d in range(1, D)],
    titles=[rf"$\tilde{{\boldsymbol{{\lambda}}}}_{{{d + 1}}}$" for d in range(1, D)],
    xlabel=r"$r_d$",
    prec_label=r"$\log\,\tilde{\boldsymbol{\lambda}}_d$",
)
# Figure 4: feature components
bar_figure(
    ve=[feature_variance_explained(d) for d in range(D)],
    logp=[np.log(deltas[d]) for d in range(D)],
    titles=[rf"$\tilde{{\boldsymbol{{\delta}}}}_{{{d + 1}}}$" for d in range(D)],
    xlabel=r"$m_d$",
    prec_label=r"$\log\,\tilde{\boldsymbol{\delta}}_d$",
)
plt.show()
