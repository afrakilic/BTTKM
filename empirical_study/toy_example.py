"""
Synthetic TT experiment: lambda/delta updates on (A) vs off (B).

Usage:
    python synthetic_experiment.py            # both
    python synthetic_experiment.py --only A   # or B
"""

import os, sys, argparse

sys.path.append(os.getcwd())
from config import *  # Import everything from config.py
from functions.TT_helpers import (generate_tt_cores,
                                  compute_G_left,
                                  pure_power_features_TT)
from functions.BTTKM import bttkm
from matplotlib.legend_handler import HandlerTuple

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

a, b = 1e-1, 1e-3

CONFIGS = {
    "A": dict(
        name="updates_on",
        c=1e-6, d=1e-6,
        g=1e-6, h=1e-6,
        lambda_update=True,
        delta_update=True,
    ),
    "B": dict(
        name="updates_off",
        c=1e-6, d=1e-6,
        g=1e-6, h=1e-6,
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
        max_iter=200,
        # lower_bound_tol= 1e-10,
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
# ON / OFF overlay of variance explained (%) and precisions
# ----------------------------------------------------------------------
COLOR_ON  = "#3B6EA8"   # penalization on  (muted blue)
COLOR_OFF = "#C0605A"   # penalization off (muted red)
STYLE = {
    "font.family": "STIXGeneral",   # Times-like text, matches LaTeX papers
    "mathtext.fontset": "stix",     # renders bold Greek (boldsymbol lambda / delta)
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
}


def rank_variance_explained(W_D, d):
    """% variance explained per rank component of bond d (cores d-1 and d), pruning formula."""
    p = np.sum(W_D[d - 1] ** 2, axis=(0, 1)) + np.sum(W_D[d] ** 2, axis=(1, 2))
    return p / np.sum(p) * 100


def feature_variance_explained(W_D, d):
    """% variance explained per feature (lateral slice) of core d."""
    q = np.sum(W_D[d] ** 2, axis=(0, 2))
    return q / np.sum(q) * 100


def _pad(v, n):
    """Pad a 1-D array with NaN to length n (in case the two models have different sizes)."""
    v = np.asarray(v, dtype=float)
    return np.concatenate([v, np.full(n - len(v), np.nan)])


def _overlay_figure(ve_on, ve_off, logp_on, logp_off, titles, xlabel, prec_label, sym, fname):
    """
    One panel per bond (lambda) or core (delta). The variance explained of the ON and OFF
    models is drawn as overlapping semi-transparent bars on a shared 0-100 % axis, and the
    log precisions of both models as lines on the right axis.
    """
    P = len(titles)
    ve_max = min(100, np.nanmax([np.nanmax(v) for v in ve_on + ve_off]) * 1.1)
    all_p = np.concatenate([np.ravel(p) for p in logp_on + logp_off])
    p_lim = (np.nanmin(all_p) - 1, np.nanmax(all_p) + 1)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, P, figsize=(2.6 * P, 2.4))
        axes = np.atleast_1d(axes)

        for i, ax in enumerate(axes):
            n = max(len(ve_on[i]), len(ve_off[i]))
            x = np.arange(n)
            ax2 = ax.twinx()

            b_off = ax.bar(x, _pad(ve_off[i], n), width=0.7, color=COLOR_OFF, alpha=0.45, linewidth=0)
            b_on = ax.bar(x, _pad(ve_on[i], n), width=0.7, color=COLOR_ON, alpha=0.55, linewidth=0)
            l_off, = ax2.plot(x, _pad(logp_off[i], n), color=COLOR_OFF, linestyle="--",
                              marker="s", markersize=2.5, linewidth=0.9)
            l_on, = ax2.plot(x, _pad(logp_on[i], n), color=COLOR_ON, linestyle="-",
                             marker="o", markersize=2.5, linewidth=0.9)

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

        fig.supxlabel(xlabel, y=0.1, fontsize=9)
        fig.legend(
            handles=[(b_on, l_on), (b_off, l_off)],
            labels=[rf"{sym}: on", rf"{sym}: off"],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04),
        )
        plt.tight_layout()
        plt.savefig(fname + ".pdf", bbox_inches="tight")
        plt.savefig(fname + ".png", dpi=300, bbox_inches="tight")
        plt.show()


def plot_predictive_on_off(res_on, res_off, fname="predictive_on_off"):
    """Predictions with +/- 2 std bounds, penalization on (left) and off (right), shared y axis."""
    y_true = np.asarray(y).ravel()
    order = np.argsort(y_true)
    x = np.arange(len(y_true))

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=True)
        sym = r"$\boldsymbol{\lambda}_d, \boldsymbol{\delta}_d$"
        for ax, res, title, color in [(axes[0], res_on, f"{sym}: on", COLOR_ON),
                                      (axes[1], res_off, f"{sym}: off", COLOR_OFF)]:
            pred, std = res["pred"][order], res["std"][order]
            ax.fill_between(x, pred - 2 * std, pred + 2 * std, color=color, alpha=0.2,
                            linewidth=0, label=r"$\pm 2\sigma$")
            ax.scatter(x, y_true[order], color="black", s=6, zorder=3, label=r"$y$")
            ax.scatter(x, pred, facecolor="white", edgecolor=color, s=10, linewidth=0.7,
                       zorder=4, label=r"$\hat{y}$")
            ax.set_title(title, pad=4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes[0].set_ylabel(r"$y$")
        fig.supxlabel(r"$n$", y=0.06, fontsize=9)
        axes[0].legend(frameon=False, loc="upper left")
        plt.tight_layout()
        plt.savefig(fname + ".pdf", bbox_inches="tight")
        plt.savefig(fname + ".png", dpi=300, bbox_inches="tight")
        plt.show()


def variance_explained_tables(results, fname="variance_explained"):
    """
    Tables of % variance explained per rank component (lambda_2, ..., lambda_D) and per feature
    (delta_1, ..., delta_D), one column per bond / core and configuration. Printed and saved as
    CSV with 3 decimals.
    """
    rank_cols, feat_cols = {}, {}
    for name, res in results.items():
        W = res["W_D"]
        D = len(W)
        tag = "on" if name == "updates_on" else "off"
        for d in range(1, D):
            rank_cols[f"lambda_{d + 1} ({tag})"] = pd.Series(rank_variance_explained(W, d))
        for d in range(D):
            feat_cols[f"delta_{d + 1} ({tag})"] = pd.Series(feature_variance_explained(W, d))

    rank_df = pd.DataFrame(rank_cols)
    rank_df.index.name = "rank index"
    feat_df = pd.DataFrame(feat_cols)
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


def plot_on_off_overlay(res_on, res_off):
    W_on, W_off = res_on["W_D"], res_off["W_D"]
    D = len(W_on)

    # lambda: rank components of bond d, labelled lambda_2, ..., lambda_D
    _overlay_figure(
        ve_on=[rank_variance_explained(W_on, d) for d in range(1, D)],
        ve_off=[rank_variance_explained(W_off, d) for d in range(1, D)],
        logp_on=[np.log(res_on["lambdas"][d]) for d in range(1, D)],
        logp_off=[np.log(res_off["lambdas"][d]) for d in range(1, D)],
        titles=[rf"$\boldsymbol{{\lambda}}_{{{d+1}}}$" for d in range(1, D)],
        xlabel=r"$r_d$",
        prec_label=r"$\log\,\boldsymbol{\lambda}_d$",
        sym=r"$\boldsymbol{\lambda}_d$",
        fname="lambda_variance_on_off",
    )

    # delta: features of core d
    _overlay_figure(
        ve_on=[feature_variance_explained(W_on, d) for d in range(D)],
        ve_off=[feature_variance_explained(W_off, d) for d in range(D)],
        logp_on=[np.log(res_on["deltas"][d]) for d in range(D)],
        logp_off=[np.log(res_off["deltas"][d]) for d in range(D)],
        titles=[rf"$\boldsymbol{{\delta}}_{{{d+1}}}$" for d in range(D)],
        xlabel=r"$m_d$",
        prec_label=r"$\log\,\boldsymbol{\delta}_d$",
        sym=r"$\boldsymbol{\delta}_d$",
        fname="delta_variance_on_off",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(CONFIGS), default=None,
                        help="run a single configuration (A or B)")
    args = parser.parse_args()

    keys = [args.only] if args.only else list(CONFIGS)
    results = {CONFIGS[k]["name"]: run(CONFIGS[k]) for k in keys}

    print("\nSummary (RMSE):")
    for name, r in results.items():
        print(f"  {name:12s} {r['rmse']:.4f}")

    variance_explained_tables(results)

    if "updates_on" in results and "updates_off" in results:
        plot_predictive_on_off(results["updates_on"], results["updates_off"])
        plot_on_off_overlay(results["updates_on"], results["updates_off"])