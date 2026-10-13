"""
Synthetic TT experiment: lambda/delta updates on.

Usage:
    python synthetic_experiment.py
"""

import os, sys

sys.path.append(os.getcwd())
from config import *  # Import everything from config.py
from functions.TT_helpers import (generate_tt_cores,
                                  compute_G_left,
                                  pure_power_features_TT)
from functions.BTTKM import bttkm

# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
D = 4
I_true = 3
R_true = 3

Ms = [I_true] * D
Rs = [1] + [R_true] * (D - 1) + [1]
N = 100

# ----------------------------------------------------------------------
# Initialisation / configuration
# ----------------------------------------------------------------------
I_init = 5
R_init = 5

rng = np.random.default_rng(42)
X = rng.standard_normal((N, D))

# the true cores are placed in cores with I_init features,
# so that only the first I_true features are nonzero
W_small = generate_tt_cores(D=D, Ms=Ms, Rs=Rs, seed=1)
W_D_true = [np.zeros((Rs[d], I_init, Rs[d + 1])) for d in range(D)]
for d in range(D):
    W_D_true[d][:, :I_true, :] =  5*W_small[d]
Phi = pure_power_features_TT(X, Ms=[I_init] * D)          # model's feature map

y = compute_G_left(Phi, W_D_true, N) + 1e-3 * rng.standard_normal((N, 1))
a, b = 1e-3, 1e-3

CONFIG = dict(
    name="updates_on",
    c=1e-6, d=1e-6,
    g=1e-6, h=1e-6,
    lambda_update=True,
    delta_update=True,
)


def run(cfg):
    print(f"\n{'=' * 70}\nConfiguration: {cfg['name']}  "
          f"(lambda_update={cfg['lambda_update']}, delta_update={cfg['delta_update']}, "
          f"c={cfg['c']}, d={cfg['d']}, g={cfg['g']}, h={cfg['h']})\n{'=' * 70}")

    start_time = time.time()

    # train the model
    model = bttkm(X.shape[1])
    W_D, lambdas, deltas, tau,_ = model.train(
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
        lower_bound_tol= 1e-10,
        precision_update=True,
        lambda_update=cfg["lambda_update"],
        delta_update=cfg["delta_update"],
        plot_results=True,
        prune_rank=False
    )

    # Predict (mse is returned by the predict function)
    prediction_mean, prediction_std, mse = model.predict(
        features=X, input_dimension=I_init, rank=R_init
    )

    prediction_mean_unscaled = prediction_mean
    rmse = np.sqrt(np.mean((prediction_mean_unscaled - y) ** 2))
    print(f"[{cfg['name']}] RMSE: {rmse:.4f}   runtime: {time.time() - start_time:.1f}s")

    return dict(rmse=rmse, W_D=W_D, lambdas=lambdas, deltas=deltas,
                pred=np.asarray(prediction_mean_unscaled).ravel(),
                std=np.asarray(prediction_std).ravel())


# ----------------------------------------------------------------------
# Variance explained (%) and precisions
# ----------------------------------------------------------------------
COLOR_BAR  = "#8C8C8C"   # medium gray, bars (variance explained)
COLOR_LINE = "#000000"   # black, line (log precision)
STYLE = {
    "font.family": "STIXGeneral",   # Times-like text, matches LaTeX papers
    "mathtext.fontset": "stix",     # renders bold Greek (boldsymbol lambda / delta)
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
}


def rank_variance_explained(W_D, d):
    """% variance explained per rank component of TT rank R_{d+1} (cores d-1 and d), pruning formula."""
    p = np.sum(W_D[d - 1] ** 2, axis=(0, 1)) + np.sum(W_D[d] ** 2, axis=(1, 2))
    return p / np.sum(p) * 100


def feature_variance_explained(W_D, d):
    """% variance explained per feature (lateral slice) of core d."""
    q = np.sum(W_D[d] ** 2, axis=(0, 2))
    return q / np.sum(q) * 100


def _bar_figure(ve, logp, titles, xlabel, prec_label, fname):
    """
    One panel per TT rank (lambda) or core (delta): variance explained as bars on a
    0-100 % axis and the log precisions as a line on the right axis.
    """
    P = len(titles)
    ve_max = min(100, np.nanmax([np.nanmax(v) for v in ve]) * 1.1)
    all_p = np.concatenate([np.ravel(p) for p in logp])
    p_lim = (np.nanmin(all_p) - 1, np.nanmax(all_p) + 1)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, P, figsize=(2.6 * P, 2.4))
        axes = np.atleast_1d(axes)

        for i, ax in enumerate(axes):
            x = np.arange(len(ve[i]))
            ax2 = ax.twinx()
            ax2.set_zorder(ax.get_zorder() + 1)
            ax2.patch.set_visible(False)

            ax.bar(x, ve[i], width=0.5, color=COLOR_BAR, alpha=0.4, linewidth=0)
            ax2.plot(x, logp[i], color=COLOR_LINE, linestyle="-",
                     marker="o", markersize=5, linewidth=1.6,
                     markeredgecolor="white", markeredgewidth=0.5)

            ax.set_ylim(0, ve_max)
            ax2.set_ylim(*p_lim)
            ax.set_xticks(x)
            ax.set_xticklabels([str(k) for k in x])
            ax.set_title(titles[i], pad=4)

            first, last = i == 0, i == P - 1
            ax.spines["top"].set_visible(False)
            ax2.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
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
        plt.savefig(fname + ".pdf", bbox_inches="tight")
        plt.show()


def plot_predictive(res, fname="predictive"):
    """Predictions with +/- 2 std bounds."""
    y_true = np.asarray(y).ravel()
    order = np.argsort(y_true)
    x = np.arange(len(y_true))

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        pred, std = res["pred"][order], res["std"][order]
        ax.fill_between(x, pred - 2 * std, pred + 2 * std, color=COLOR_BAR, alpha=0.2,
                        linewidth=0, label=r"$\pm 2\sigma$")
        ax.scatter(x, y_true[order], color="black", s=6, zorder=3, label=r"$y$")
        ax.scatter(x, pred, facecolor="white", edgecolor=COLOR_LINE, s=10, linewidth=0.7,
                   zorder=4, label=r"$\hat{y}$")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel(r"$y$")
        ax.set_xlabel(r"$n$")
        ax.legend(frameon=False, loc="upper left")
        plt.tight_layout()
        plt.savefig(fname + ".pdf", bbox_inches="tight")
        plt.show()


def variance_explained_tables(res, fname="variance_explained"):
    """
    Tables of % variance explained per rank component (lambda_2, ..., lambda_D) and per feature
    (delta_1, ..., delta_D), one column per TT rank / core. Printed and saved as CSV with 3 decimals.
    """
    W = res["W_D"]
    D = len(W)
    rank_df = pd.DataFrame({f"lambda_{d + 1}": pd.Series(rank_variance_explained(W, d))
                            for d in range(1, D)})
    rank_df.index.name = "rank index"
    feat_df = pd.DataFrame({f"delta_{d + 1}": pd.Series(feature_variance_explained(W, d))
                            for d in range(D)})
    feat_df.index.name = "feature index"

    fmt = lambda x: f"{x:.3f}"
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print("\nVariance explained per rank component (%)")
        print(rank_df.to_string(float_format=fmt))
        print("\nVariance explained per feature (%)")
        print(feat_df.to_string(float_format=fmt))

    rank_df.to_csv(fname + "_rank.csv", float_format="%.3f")
    feat_df.to_csv(fname + "_feature.csv", float_format="%.3f")
    return rank_df, feat_df


def plot_variance(res):
    W = res["W_D"]
    D = len(W)

    # lambda: rank components of each TT rank, labelled lambda_2, ..., lambda_D
    _bar_figure(
        ve=[rank_variance_explained(W, d) for d in range(1, D)],
        logp=[np.log(res["lambdas"][d]) for d in range(1, D)],
        titles=[rf"$\boldsymbol{{\lambda}}_{{{d+1}}}$" for d in range(1, D)],
        xlabel=r"$r_d$",
        prec_label=r"$\log\,\boldsymbol{\lambda}_d$",
        fname="lambda_variance",
    )

    # delta: features of core d
    _bar_figure(
        ve=[feature_variance_explained(W, d) for d in range(D)],
        logp=[np.log(res["deltas"][d]) for d in range(D)],
        titles=[rf"$\boldsymbol{{\delta}}_{{{d+1}}}$" for d in range(D)],
        xlabel=r"$m_d$",
        prec_label=r"$\log\,\boldsymbol{\delta}_d$",
        fname="delta_variance",
    )


if __name__ == "__main__":
    res = run(CONFIG)

    print(f"\nRMSE: {res['rmse']:.4f}")

    variance_explained_tables(res)
    plot_predictive(res)
    plot_variance(res)