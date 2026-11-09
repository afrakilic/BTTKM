"""BTT-KM: Bayesian tensor train kernel machine.

Model: y = <Phi(x), W> + e, with the weight tensor W in tensor-train (TT) format,
cores W^(d) of shape (R_d, M_d, R_{d+1}), R_1 = R_{D+1} = 1.

Priors (all with Gamma hyperpriors):
  - lambda_d: one precision per component of TT rank R_d (shared by cores d-1 and d) -> prunes the ranks,
  - delta_d:  one precision per feature of core d                                     -> prunes feature dims,
  - tau:      noise precision.

`train` runs mean-field variational inference: a Bayesian ALS sweep over the
cores, followed (from iteration 6 on) by closed-form updates of lambda, delta
and tau. `predict` returns the mean and standard deviation of the
Student's-t predictive distribution.
"""

import numpy as np
from tqdm import trange

from .tt import (
    pure_power_features,
    random_tt_cores,
    khatri_rao_rows,
    unfold,
    contract_left,
    contract_right,
    second_moment_left,
    second_moment_right,
    ggT_to_core_matrix,
    safelog,
)


class BTTKM:
    def train(
        self,
        X,
        y,
        M,
        R,
        a0,
        b0,
        c0,
        d0,
        g0,
        h0,
        max_iter=100,
        init_seed=1,
        lambda_update=True,
        delta_update=True,
        precision_update=True,
        prune_rank=False,
        rank_tol=1e-3,
        lower_bound_tol=1e-4,
        verbose=True,
    ):
        """Fit the model.

        X, y      : training inputs (N, D) and targets (N, 1)
        M, R      : feature dimension per input and initial TT rank
        a0, b0    : Gamma prior of the noise precision tau
        c0, d0    : Gamma prior of the rank precisions lambda
        g0, h0    : Gamma prior of the feature precisions delta
        init_seed : seed of the random initial cores (entries ~ N(0, 1))
        rank_tol  : rank components explaining less than this % of the variance are pruned
        The precisions are only updated from iteration 6 on (it > 4).

        After training, the model stores:
          W_D, V        posterior means and covariances of the cores
          lambdas, deltas, tau
          lower_bound   variational lower bound per iteration
          effective_ranks, avg_effective_rank   inner TT ranks R_2 ... R_D
        """
        N, D = X.shape
        Ms = [M] * D
        Rs = [1] + [R] * (D - 1) + [1]

        tau = a0 / b0
        g0s = [g0 * np.ones(m) for m in Ms]; gNs = [g.copy() for g in g0s]
        h0s = [h0 * np.ones(m) for m in Ms]; hNs = [h.copy() for h in h0s]
        c0s = [c0 * np.ones(r) for r in Rs]; cNs = [c.copy() for c in c0s]
        d0s = [d0 * np.ones(r) for r in Rs]; dNs = [d.copy() for d in d0s]
        deltas = [g / h for g, h in zip(g0s, h0s)]
        lambdas = [c / d for c, d in zip(c0s, d0s)]

        # Initial cores and covariances
        Sigma = [1e-5 * np.eye(Rs[d] * Ms[d] * Rs[d + 1]) for d in range(D)]
        W_D = random_tt_cores(Ms, Rs, seed=init_seed)
        Phi = pure_power_features(X, Ms)

        LB = np.zeros(max_iter)
        rel_change = 0

        # Left environments: contraction of cores 1..d (means G, second moments H)
        G_left = [None] * (D + 1)
        H_left = [None] * (D + 1)
        G_left[0] = np.ones((N, 1))
        H_left[0] = np.ones((N, 1))

        pbar = trange(max_iter, desc="Training", leave=True, disable=not verbose)
        for it in pbar:

            # Right environments: contraction of cores d+1..D
            G_right = [None] * (D + 1)
            H_right = [None] * (D + 1)
            G_right[D] = np.ones((N, 1))
            H_right[D] = np.ones((N, 1))
            for k in range(D - 1, -1, -1):
                G_right[k] = contract_right([Phi[k]], [W_D[k]], N, G0=G_right[k + 1])
                H_right[k] = second_moment_right([Phi[k]], [W_D[k]], [Sigma[k]], N, H0=H_right[k + 1])

            for d in range(D):
                # ---- Core update q(W^(d)) ----
                Gd = khatri_rao_rows(khatri_rao_rows(G_right[d + 1], Phi[d]), G_left[d])
                ggT = ggT_to_core_matrix(
                    khatri_rao_rows(H_left[d], Phi[d]).T @ khatri_rao_rows(H_right[d + 1], Phi[d]),
                    R_d=W_D[d].shape[0], M_d=W_D[d].shape[1], R_next=W_D[d].shape[2],
                )
                prior = np.kron(
                    np.kron(lambdas[d + 1] * np.eye(Rs[d + 1]), deltas[d] * np.eye(Ms[d])),
                    lambdas[d] * np.eye(Rs[d]),
                )
                A = tau * ggT + prior
                try:  # inverse via eigendecomposition, with tiny eigenvalues clipped
                    eigvals, eigvecs = np.linalg.eigh(A)
                    eigvals = np.maximum(eigvals, 1e-10 * np.maximum(eigvals.max(), 1e-10))
                    Sigma[d] = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
                except np.linalg.LinAlgError:
                    try:
                        Sigma[d] = np.linalg.pinv(A)
                    except np.linalg.LinAlgError:
                        raise RuntimeError(f"INVERSION_FAILED at core d={d}, iter it={it}")
                W_D[d] = (tau * Sigma[d] @ Gd.T @ y).reshape(W_D[d].shape, order="F")

                G_left[d + 1] = contract_left([Phi[d]], [W_D[d]], N, G0=G_left[d])
                H_left[d + 1] = second_moment_left([Phi[d]], [W_D[d]], [Sigma[d]], N, H0=H_left[d])

                # ---- Rank precisions q(lambda_d) of R_d (shared by cores d-1 and d) ----
                if it > 4 and lambda_update and d > 0:
                    K1 = np.kron(lambdas[d + 1] * np.eye(lambdas[d + 1].size), deltas[d] * np.eye(deltas[d].size))
                    K2 = np.kron(deltas[d - 1] * np.eye(deltas[d - 1].size), lambdas[d - 1] * np.eye(lambdas[d - 1].size))

                    var_d = unfold(np.diag(Sigma[d]).reshape(W_D[d].shape, order="F"), 1)
                    W1 = unfold(W_D[d], 1)
                    e1 = np.diag(W1 @ K1 @ W1.T) + var_d @ np.kron(lambdas[d + 1], deltas[d])

                    var_prev = unfold(np.diag(Sigma[d - 1]).reshape(W_D[d - 1].shape, order="F"), 3)
                    W3 = unfold(W_D[d - 1], 3)
                    e3 = np.diag(W3 @ K2 @ W3.T) + var_prev @ np.kron(deltas[d - 1], lambdas[d - 1])

                    cNs[d] = c0s[d] + 0.5 * (deltas[d - 1].size * lambdas[d - 1].size + lambdas[d + 1].size * deltas[d].size)
                    dNs[d] = d0s[d] + 0.5 * (e1 + e3)
                    lambdas[d] = cNs[d] / dNs[d]
                    lambdas[d][lambdas[d] < 1e-5] = 1e-5

                # ---- Feature precisions q(delta_d) ----
                if it > 4 and delta_update:
                    K = np.kron(lambdas[d + 1] * np.eye(lambdas[d + 1].size), lambdas[d] * np.eye(lambdas[d].size))
                    var_d = unfold(np.diag(Sigma[d]).reshape(W_D[d].shape, order="F"), 2)
                    W2 = unfold(W_D[d], 2)
                    e2 = np.diag(W2 @ K @ W2.T) + var_d @ np.kron(lambdas[d + 1], lambdas[d])
                    gNs[d] = g0s[d] + 0.5 * (lambdas[d].size * lambdas[d + 1].size)
                    hNs[d] = h0s[d] + 0.5 * e2
                    deltas[d] = gNs[d] / hNs[d]
                    deltas[d][deltas[d] < 1e-5] = 1e-5

            # ---- Noise precision q(tau): E||y - Phi w||^2 ----
            y_hat = contract_left(Phi, W_D, N)
            err = float(np.sum(y.T @ y + (-2 * np.dot(y.T, y_hat)) + np.sum(second_moment_left(Phi, W_D, Sigma, N), axis=0)))
            if it > 4 and precision_update:
                a_N, b_N = a0 + (N / 2), b0 + (0.5 * err)
            else:
                a_N, b_N = a0, b0
            tau = a_N / b_N

            # ---- Rank pruning (optional) ----
            if it > 4 and prune_rank:
                for d in range(1, D):
                    stack = np.hstack((unfold(W_D[d - 1], 3), unfold(W_D[d], 1)))
                    power = np.diag(stack @ stack.T)
                    keep = power / np.sum(power) * 100 >= rank_tol
                    if not np.any(keep):
                        keep[np.argmax(power)] = True
                    if np.all(keep):
                        continue
                    idx_prev = np.arange(W_D[d - 1].size).reshape(W_D[d - 1].shape, order="F")[:, :, keep].ravel(order="F")
                    idx_d = np.arange(W_D[d].size).reshape(W_D[d].shape, order="F")[keep, :, :].ravel(order="F")
                    W_D[d - 1] = W_D[d - 1][:, :, keep]
                    W_D[d] = W_D[d][keep, :, :]
                    Rs[d] = int(np.sum(keep))
                    lambdas[d], cNs[d], dNs[d] = lambdas[d][keep], cNs[d][keep], dNs[d][keep]
                    c0s[d], d0s[d] = c0s[d][keep], d0s[d][keep]
                    Sigma[d - 1] = Sigma[d - 1][np.ix_(idx_prev, idx_prev)]
                    Sigma[d] = Sigma[d][np.ix_(idx_d, idx_d)]

            # ---- Variational lower bound ----
            LB[it] = self._lower_bound(W_D, Sigma, lambdas, deltas, Rs, Ms, tau, err,
                                       a_N, b_N, b0, cNs, dNs, d0s, gNs, hNs, h0s)

            # ---- Convergence check ----
            rel_change = (LB[it] - LB[it - 1]) / LB[2] if it > 2 else np.nan
            pbar.set_postfix({"RelChan": f"{rel_change:.2e}", "err": f"{err:.1e}"})
            if it > 5 and abs(rel_change) < lower_bound_tol:
                if verbose:
                    print("======= Converged =======")
                break

        # Noise precision posterior from the final residual (used for prediction)
        a_N = a0 + (N / 2)
        b_N = b0 + (0.5 * err)

        self.W_D, self.V, self.a, self.b = W_D, Sigma, a_N, b_N
        self.M = M
        self.lambdas, self.deltas, self.tau = lambdas, deltas, tau
        self.lower_bound = LB[: it + 1]
        self.effective_ranks = [W_D[d].shape[0] for d in range(1, D)]
        self.avg_effective_rank = float(np.mean(self.effective_ranks)) if self.effective_ranks else float("nan")
        return self

    @staticmethod
    def _lower_bound(W_D, Sigma, lambdas, deltas, Rs, Ms, tau, err, a_N, b_N, b0, cNs, dNs, d0s, gNs, hNs, h0s):
        """Variational lower bound (up to constants); only used to check convergence."""
        D = len(W_D)
        t1 = -0.5 * tau * err
        t2 = 0
        for d in range(D):
            prior = np.kron(np.kron(lambdas[d + 1] * np.eye(Rs[d + 1]), deltas[d] * np.eye(Ms[d])), lambdas[d] * np.eye(Rs[d]))
            w = W_D[d].ravel(order="F")
            t2 += -0.5 * np.trace(prior @ (np.outer(w, w) + Sigma[d]))
        t3 = 0.0
        for S in Sigma:
            sign, logdet = np.linalg.slogdet(S)
            if sign <= 0:  # Sigma should be positive definite
                logdet = -1e10
            t3 += 0.5 * logdet
        t4 = sum(np.sum(cNs[d] * (1 - safelog(dNs[d]) - (d0s[d] / dNs[d]))) for d in range(1, D))
        t5 = sum(
            np.sum(np.array(gNs[d]) * (1 - safelog(np.array(hNs[d])) - (h0s[d] / np.array(hNs[d]))))
            for d in range(len(gNs))
        )
        t6 = a_N * (1 - safelog(b_N) - (b0 / b_N))
        return t1 + t2 + t3 + t4 + t5 + t6

    def predict(self, X):
        """Mean (N, 1) and standard deviation (N,) of the Student's-t predictive (nu = 2 * a_N).

        The summed quadratic terms g^T Sigma g are divided by the number of core
        entries sum_d R_d M_d R_{d+1} of the trained model.
        """
        N, D = X.shape
        Phi = pure_power_features(X, [self.M] * D)
        mean = contract_left(Phi, self.W_D, N)

        G_right = [None] * (D + 1)
        G_right[D] = np.ones((N, 1))
        for k in range(D - 1, -1, -1):
            G_right[k] = contract_right([Phi[k]], [self.W_D[k]], N, G0=G_right[k + 1])

        G_left = np.ones((N, 1))
        quad = np.zeros(N)
        for d in range(D):
            g = khatri_rao_rows(khatri_rao_rows(G_right[d + 1], Phi[d]), G_left)
            quad += np.sum((g @ self.V[d]) * g, axis=1)
            G_left = contract_left([Phi[d]], [self.W_D[d]], N, G0=G_left)

        S = (2 * self.a / (2 * self.a - 2)) * ((self.b / self.a) + (quad))
        return mean, np.sqrt(S)
