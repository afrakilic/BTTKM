"""
HELPER FUNCTIONS
Dependencies and configurations are centralized in `config.py`.
"""

import os, sys
sys.path.append(os.getcwd())
from config import *  # Import everything from config.py
import numpy as np


def pure_power_features_full(X, input_dimension):
    """
    Pure-Power Polynomial Features

    Parameters:
    X : numpy.ndarray
        Input array of shape (n_samples, n_features).
    M : int
        Maximum power (degree) of the polynomial features.

    Returns:
    Mati : numpy.ndarray
        Array of shape (n_features, n_samples, M) containing unit-norm pure-power features.
    """
    # Compute the pure-power features
    Mati = np.power(X.T[:, :, np.newaxis], np.arange(input_dimension))

    # Normalize each sample's features along the last axis to have a unit norm
    norms = np.linalg.norm(
        Mati, axis=2, keepdims=True
    )  # Compute norms along the power axis
    Mati = Mati / norms  # Normalize features to unit norm

    return Mati


def columnwise_kronecker(A, B):
    """Compute the columnwise Kronecker product of two matrices A and B."""
    # Check dimensions
    if A.shape[1] != B.shape[1]:
        raise ValueError("Number of columns in A and B must be the same.")

    # Dimensions of input matrices
    m, n = A.shape
    p, _ = B.shape

    # Initialize the result matrix
    K = np.zeros((m * p, n))

    # Calculate columnwise Kronecker product
    for i in range(n):
        K[:, i] = np.kron(A[:, i], B[:, i])

    return K


def dotkron(*matrices):
    """
    Computes the row-wise right-Kronecker product of two or three matrices.

    Parameters:
        matrices: Two or three matrices with the same number of rows.

    Returns:
        y: The resulting row-wise right-Kronecker product.

    Raises:
        ValueError: If the matrices do not have the same number of rows
                    or if the number of matrices is not 2 or 3.
    """
    if len(matrices) == 2:
        L, R = matrices
        r1, c1 = L.shape
        r2, c2 = R.shape

        if r1 != r2:
            raise ValueError("Matrices should have equal rows!")

        # Row-wise right-Kronecker product for two matrices
        y = np.tile(L, (1, c2)) * np.kron(R, np.ones((1, c1)))

    elif len(matrices) == 3:
        L, M, R = matrices
        r1, _ = L.shape
        r2, _ = M.shape
        r3, _ = R.shape

        if r1 != r2 or r2 != r3:
            raise ValueError("Matrices should have equal rows!")

        # Recursive call for three matrices
        y = dotkron(L, dotkron(M, R))

    else:
        raise ValueError("Please input 2 or 3 matrices!")

    return y


def temp(
    Phi, V, R
):  # her bir RXR block'u vectorize edip rowlarina koyuyor yeni matrix'in
    I = Phi.shape[1]
    V = np.reshape(V, (I, R, I, R), order="F")
    V_permuted = np.transpose(V, axes=(0, 2, 1, 3))
    result = np.reshape(V_permuted, (I**2, R**2))
    return dotkron(Phi, Phi) @ result


def dotkronX(A, B, y):
    """
    Computes the Kronecker dot product for large matrices using batch processing.

    Parameters:
    A : np.ndarray (N, DA)
        First input matrix
    B : np.ndarray (N, DB)
        Second input matrix
    y : np.ndarray (N, 1)
        Target vector
    batch_size : int, optional
        Number of samples to process in each batch (default is 10000)

    Returns:
    CC : np.ndarray (DA*DB, DA*DB)
        Computed Kronecker product matrix
    Cy : np.ndarray (DA*DB, 1)
        Computed target vector
    """
    N, DA = A.shape
    _, DB = B.shape

    CC = np.zeros((DA * DB, DA * DB))
    Cy = np.zeros((DA * DB, 1))
    y = y.reshape(-1, 1)  # Ensure y has shape (N, 1)

    batch_size = 10000
    for n in range(0, N, batch_size):
        idx = min(n + batch_size - 1, N)  # Ensure we don't exceed N

        temp = (np.tile(A[n:idx, :], (1, DB))) * (
            np.kron(B[n:idx, :], np.ones((1, DA)))
        )

        CC += temp.T @ temp  # Accumulate Kronecker product
        Cy += temp.T @ y[n:idx, :]

    return CC, Cy


def safe_division(X, A, epsilon=1):
    A_safe = np.where(A == 0, epsilon, A)
    Y = X / A_safe
    return Y


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
        ax.set_xlim(-0.5, len(x) - 0.5)
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
        ax.set_xlim(-0.5, len(x) - 0.5)
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

