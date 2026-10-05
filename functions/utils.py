"""
HELPER FUNCTIONS
Dependencies and configurations are centralized in `config.py`.
"""

import os, sys
sys.path.append(os.getcwd())
from config import *  # Import everything from config.py
import numpy as np

def safelog(x):
    return np.log(np.clip(x, 1e-10, 1e10))  # x to range [1e-10, 1e10]


def plot_predictive_bounds(y_test, prediction_mean, prediction_std, rmse=None):
    pred = prediction_mean.ravel()
    std  = prediction_std.ravel()

    sort_idx    = np.argsort(y_test.ravel())
    y_sorted    = y_test.ravel()[sort_idx]
    pred_sorted = pred[sort_idx]
    std_sorted  = std[sort_idx]
    x_sorted    = np.arange(len(y_sorted))

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.fill_between(x_sorted, pred_sorted - 2*std_sorted, pred_sorted + 2*std_sorted,
                    color='gray', alpha=0.2, label=r"$\pm 2\sigma$")
    ax.scatter(x_sorted, y_sorted,    color='black', s=15, alpha=0.9,
               label="True", zorder=3)
    ax.scatter(x_sorted, pred_sorted, color='white', s=15, alpha=0.9,
               edgecolors='black', linewidths=0.8, label="Predicted", zorder=4)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Observation index (sorted by true value)")
    ax.set_ylabel("Value")
    title = "Predictions with uncertainty"
    if rmse is not None:
        title += rf"  —  RMSE $=$ {rmse:.4f}"
    ax.set_title(title, pad=8)
    ax.legend(frameon=False, fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_precision_vs_norms(W_D, lambdas, deltas):
    BLUE   = "#4878CF"
    RED    = "#C44E52"
    ORANGE = "#DD8452"

    D = len(W_D)

    # ── Figure 1 : Lambda ────────────────────────────────────────────────────
    fig1, axes = plt.subplots(1, D-1, figsize=(4*(D-1), 3.5))
    if D-1 == 1:
        axes = [axes]

    all_norms1 = [np.linalg.norm(W_D[d-1], axis=(0,1)) + np.linalg.norm(W_D[d], axis=(1,2)) for d in range(1, D)]
    all_lams   = [np.log(lambdas[d]) for d in range(1, D)]
    norm1_max  = max(n.max() for n in all_norms1) + 1
    lam_max    = max(l.max() for l in all_lams)   + 1
    lam_min    = min(l.min() for l in all_lams)   - 1

    for d in range(1, D):
        ax  = axes[d-1]
        ax2 = ax.twinx()
        x   = np.arange(len(lambdas[d]))

        bar  = ax.bar(x, all_norms1[d-1], color=BLUE, alpha=0.75, width=0.5)
        line, = ax2.plot(x, np.log(lambdas[d]), color=RED, marker='o',
                         markersize=4, linewidth=1.2, linestyle='--')

        ax.set_ylim(0, norm1_max)
        ax2.set_ylim(lam_min, lam_max)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x]) 
        ax.set_title(rf"$\lambda_{d}$", pad=4)
        ax.set_xlabel("Rank index")

        ax.tick_params(axis='y', colors=BLUE)
        if d == 1:
            ax.set_ylabel("Component norm", color=BLUE)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis='y', length=0)

        ax2.tick_params(axis='y', colors=RED)
        ax2.spines['right'].set_color(RED)
        ax2.spines['left'].set_color(BLUE)
        ax.spines['right'].set_visible(False)
        if d == D-1:
            ax2.set_ylabel(r"$\log\,\lambda$ precision", color=RED)
        else:
            ax2.set_yticklabels([])
            ax2.tick_params(axis='y', length=0)

    fig1.legend(
        handles=[bar, line],
        labels=[r'$\|\mathcal{W}\|_3 + \|\mathcal{W}\|_1$', r'$\log\,\lambda$'],
        loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.08),
    )
    fig1.suptitle(r"$\lambda$ precision vs.\ TT-core rank norms", fontsize=12)
    plt.tight_layout()
    plt.show()

    # ── Figure 2 : Delta ─────────────────────────────────────────────────────
    fig2, axes = plt.subplots(1, D, figsize=(4*D, 3.5))
    if D == 1:
        axes = [axes]

    all_norms2 = [np.linalg.norm(W_D[d], axis=(0, 2)) for d in range(D)]
    all_deltas = [np.log(deltas[d]) for d in range(D)]
    norm2_max  = max(n.max() for n in all_norms2) + 1
    delta_max  = max(l.max() for l in all_deltas) + 1
    delta_min  = min(l.min() for l in all_deltas) - 1

    for d in range(D):
        ax  = axes[d]
        ax2 = ax.twinx()
        x   = np.arange(len(deltas[d]))

        bar  = ax.bar(x, all_norms2[d], color=ORANGE, alpha=0.75, width=0.5)
        line, = ax2.plot(x, np.log(deltas[d]), color=RED, marker='o',
                         markersize=4, linewidth=1.2, linestyle='--')

        ax.set_ylim(0, norm2_max)
        ax2.set_ylim(delta_min, delta_max)
        # ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x]) 
        ax.set_title(rf"$\delta_{d}$", pad=4)
        ax.set_xlabel("Feature index")

        ax.tick_params(axis='y', colors=ORANGE)
        if d == 0:
            ax.set_ylabel("Component norm", color=ORANGE)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis='y', length=0)

        ax2.tick_params(axis='y', colors=RED)
        ax2.spines['right'].set_color(RED)
        ax2.spines['left'].set_color(ORANGE)
        ax.spines['right'].set_visible(False)
        if d == D-1:
            ax2.set_ylabel(r"$\log\,\delta$ precision", color=RED)
        else:
            ax2.set_yticklabels([])
            ax2.tick_params(axis='y', length=0)

    fig2.legend(
        handles=[bar, line],
        labels=[r'$\|\mathcal{W}\|_2$', r'$\log\,\delta$'],
        loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.08),
    )
    fig2.suptitle(r"$\delta$ precision vs.\ TT-core feature norms", fontsize=12)
    plt.tight_layout()
    plt.show()

