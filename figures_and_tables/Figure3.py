"""
This script generates Figure 3, demonstrating the ability of the BTN-Kernel machines to recover
ground truth factor matrices from synthetic data.

A synthetic tensor regression task is constructed by:
  - Sampling low-rank weight matrices and sparse feature interactions across three modes.
  - Generating output labels from a known multiplicative structure with added Gaussian noise.

BTN-Kernel machines is then trained to infer the underlying weight structure using variational updates
on the posterior of factor matrices and associated precision parameters.

The recovered weights are visualized as log-normalized heatmaps, with axes annotated by the
corresponding learned prior precisions (λ_M and λ_R). These results illustrate  BTN-Kernel machines'
capacity to uncover interpretable and sparse low-rank structure from noisy observations.
"""

import os, sys

sys.path.append(os.getcwd())
from config import *  # Import everything from config.pyg
from functions.utils import (
    pure_power_features_full,
    dotkron,
    temp,
    safe_division,
    dotkronX,
)

np.random.seed(1)

# TRUE MODEL
D, Rtrue, Rmax, Imax, N = 3, 3, 5, 5, 500
sigma_e = 1e-3
e = sigma_e * np.random.randn(N)

# Generate random indices and weight matrices in a loop
indices = [
    np.sort(np.random.choice(Imax, np.random.randint(1, Imax), replace=False))
    for _ in range(D)
]
Ws = [1e1 * np.random.randn(len(idx), Rtrue) for idx in indices]
#  features
X = np.random.rand(N, D)
Phi = pure_power_features_full(X, Imax)

y = np.sum(
    (Phi[0][:, indices[0]] @ Ws[0])
    * (Phi[1][:, indices[1]] @ Ws[1])
    * (Phi[2][:, indices[2]] @ Ws[2]),
    axis=1,
)
y = y + e


# initializations
a0, b0 = 1e-3, 1e-3
c0, d0 = 1e-6 * np.ones(Rmax), 1e-6 * np.ones(Rmax)
g0, h0 = 1e-6 * np.ones(Imax), 1e-6 * np.ones(Imax)

precision_update = True
lambda_update = True
delta_update = True

# factor matrices
W_D = [np.random.randn(Imax, Rmax) for _ in range(D)]  #  IXR
# Initialize the covariance matrices
WSigma_D = [0.1 * np.kron(np.eye(Rmax), np.eye(Imax)) for d in range(D)]

# Compute the Hadamard product of matrices in W_K
hadamard_product_V = np.ones((N, Rmax**2))  # Start with the first matrix
hadamard_product_mean = np.ones((N, Rmax))

c_N = c0
d_N = d0
lambda_R = c0 / d0
g_N = [g0 for _ in range(D)]
h_N = [h0 for _ in range(D)]
delta = [[g0 / h0] for _ in range(D)]
tau = a0 / b0

for d in range(len(W_D)):
    hadamard_product_V = hadamard_product_V * temp(
        Phi=Phi[d], V=WSigma_D[d], R=Rmax
    )  # Element-wise multiplication
    hadamard_product_mean = hadamard_product_mean * (Phi[d] @ W_D[d])

for it in range(100):

    for d in range(D):  # update the posterior q(vec(W^d)):

        hadamard_product_V = safe_division(
            hadamard_product_V, (temp(Phi=Phi[d], V=WSigma_D[d], R=Rmax))
        )
        hadamard_product_mean = safe_division(
            hadamard_product_mean, ((Phi[d] @ W_D[d]))
        )

        W_K_PROD_V = dotkron(Phi[d], Phi[d]).T @ hadamard_product_V
        cc, cy = dotkronX(Phi[d], hadamard_product_mean, y)
        V_temp = np.reshape(
            np.transpose(
                np.reshape(W_K_PROD_V, (Imax, Imax, Rmax, Rmax), order="F"),
                axes=(0, 2, 1, 3),
            ),
            (Imax * Rmax, Imax * Rmax),
            order="F",
        )

        WSigma_D[d] = np.linalg.pinv(
            tau * (cc + V_temp)
            + np.kron(lambda_R * np.eye(Rmax), delta[d] * np.eye(Imax))
        )
        W_D[d] = np.reshape((tau * WSigma_D[d] @ cy), (Imax, Rmax), order="F")

        hadamard_product_V = hadamard_product_V * temp(
            Phi=Phi[d], V=WSigma_D[d], R=Rmax
        )
        hadamard_product_mean = hadamard_product_mean * (Phi[d] @ W_D[d])

    # delta Update
    if delta_update:
        for d in range(D):
            mtemp = np.diag(W_D[d] @ (lambda_R * np.eye(Rmax)) @ W_D[d].T)
            vtemp = np.diag(
                np.reshape(
                    WSigma_D[d]
                    .reshape(Imax, Rmax, Imax, Rmax)
                    .transpose(0, 2, 1, 3)
                    .reshape(Imax**2, Rmax**2)
                    @ (lambda_R * np.eye(Rmax)).ravel(order="F"),
                    (Imax, Imax),
                )
            )
            g_N[d] = g0 + Rmax / 2
            h_N[d] = h0 * np.ones(Imax) + (mtemp + vtemp) / 2
            delta[d] = g_N[d] / h_N[d]

    if lambda_update:
        c_N = (0.5 * D * Imax) + c0
        d_N = 0

        for d in range(D):
            np.transpose(
                np.reshape(WSigma_D[d], (Imax, Rmax, Imax, Rmax), order="F"),
                axes=(0, 2, 1, 3),
            )
            mtemp = np.diag(W_D[d].T @ (delta[d] * np.eye(Imax)) @ W_D[d])
            vtemp = np.diag(
                np.reshape(
                    (delta[d] * np.eye(Imax)).ravel(order="F").T
                    @ WSigma_D[d]
                    .reshape(Imax, Rmax, Imax, Rmax)
                    .transpose(0, 2, 1, 3)
                    .reshape(Imax**2, Rmax**2),
                    (Rmax, Rmax),
                )
            )
            d_N += mtemp + vtemp

        d_N = d0 + (0.5 * d_N)
        lambda_R = c_N / d_N

    # Error Precision Update
    ss_error = np.dot(
        (y - np.sum(hadamard_product_mean, axis=1)),
        (y - np.sum(hadamard_product_mean, axis=1)),
    )
    covariance = np.sum(np.sum(hadamard_product_V, axis=1))
    err = ss_error  # + covariance

    if precision_update:
        a_N = a0 + (N / 2)
        b_N = b0 + (0.5 * (ss_error + covariance))
    else:
        a_N = a0
        b_N = b0

    tau = a_N / b_N


def print_rounded_scientific(label, array):
    print(f"{label}:")
    with np.printoptions(
        precision=2, suppress=False, formatter={"float_kind": lambda x: f"{x:.2e}"}
    ):
        print(array)
    print()


print_rounded_scientific("W_D[1]", W_D[0])
print_rounded_scientific("W_D[2]", W_D[1])
print_rounded_scientific("W_D[3]", W_D[2])
print_rounded_scientific("delta[1]", delta[0])
print_rounded_scientific("delta[2]", delta[1])
print_rounded_scientific("delta[3]", delta[2])
print_rounded_scientific("lambda_R", lambda_R)
print_rounded_scientific("tau", tau)

for i in range(3):
    # compute singular values only (no U, V^T)
    s = np.linalg.svd(W_D[i], compute_uv=False)
    print(f"Singular values of W_D[{i}]: {s}")

# Convert to absolute values
W_D_abs = [np.abs(W) for W in W_D]
raw_min = min(W.min() for W in W_D_abs)
vmin = max(raw_min, 1e-5)  # Ensure vmin > 0
vmax = max(W.max() for W in W_D_abs)

# Heatmap
fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), constrained_layout=True)
fig.subplots_adjust(wspace=0.2, hspace=5)
heatmap = None


def format_lambda(val):
    if abs(val) > 1e2:
        base, exp = f"{val:.2e}".split("e")
        base = float(base)  # Convert base to float
        return f"{round(base)} \\times 10^{{{int(exp)}}}"
    else:
        return f"{val:.2f}"


for i, ax in enumerate(axes):
    
    col_labels = [f"${format_lambda(val)}$" for j, val in enumerate(lambda_R)]

    row_labels = [f"${format_lambda(val)}$" for k, val in enumerate(delta[i])]

    heatmap = sns.heatmap(
        W_D_abs[i],
        cmap="gray_r",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        ax=ax,
        cbar=False,
        square=True,
        xticklabels=col_labels,
        yticklabels=row_labels,
        annot=False,
        linewidths=0.01,
        linecolor="lightgray",
    )
    ax.set_aspect("equal")
    ax.set_title(f"$\\tilde{{W}}^{{({i+1})}}$", fontsize=18)
    ax.tick_params(axis="x", rotation=45, labelsize=12.1)
    ax.tick_params(axis="y", labelsize=12.1)

cbar = fig.colorbar(
    heatmap.get_children()[0],
    ax=axes.ravel().tolist(),
    orientation="vertical",
    fraction=0.02,
    pad=0.05,
)
cbar.ax.tick_params(labelsize=10)
fig.subplots_adjust(left=0.16, bottom=0.16)

t_left = fig.text(
    -0.00, 0.5,
    r"$\boldsymbol{\delta}_d$",
    va="center",
    ha="center",
    rotation="vertical",
    fontsize=15,
)

t_bottom = fig.text(
    0.5, -0.05,
    r"$\boldsymbol{\lambda}$",
    va="center",
    ha="center",
    fontsize=15,
)

# plt.savefig(
#     "plot1.pdf",
#     format="pdf",
#     bbox_inches="tight",
#     bbox_extra_artists=[t_left, t_bottom],
# )

plt.show()