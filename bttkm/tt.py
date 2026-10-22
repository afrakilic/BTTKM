"""Tensor-train helpers: feature map, core unfoldings and the left/right contractions.

TT cores are 3-way arrays W[d] of shape (R_d, M_d, R_{d+1}) with R_1 = R_{D+1} = 1.
Phi is a list of D feature matrices, Phi[d] of shape (N, M_d).
"""

import numpy as np


def pure_power_features(X, Ms, eps=1e-12):
    """Unit-norm pure-power features plus a 0.2 offset.

    Returns a list with, for every input d, the N x M_d matrix of powers
    x_d^0 ... x_d^(M_d - 1), each row scaled to unit norm, plus 0.2.
    """
    X = np.asarray(X)
    Phis = []
    for d in range(X.shape[1]):
        Phi = X[:, d : d + 1] ** np.arange(Ms[d])[None, :]
        Phi /= np.linalg.norm(Phi, axis=1, keepdims=True) + eps
        Phis.append(Phi + 0.2)
    return Phis


def random_tt_cores(Ms, Rs, seed=None, scale=1):
    """TT cores with i.i.d. N(0, scale^2) entries."""
    rng = np.random.default_rng(seed)
    return [scale * rng.standard_normal((Rs[d], Ms[d], Rs[d + 1])) for d in range(len(Ms))]


def khatri_rao_rows(A, B):
    """Row-wise Kronecker product: row n is kron(A[n], B[n])."""
    return (A[:, :, None] * B[:, None, :]).reshape(A.shape[0], -1)


def unfold(T, mode):
    """Mode-1, -2 or -3 unfolding (column-major) of a 3-way core T of shape (R, M, Rn).

    mode 1: R x (M*Rn),  mode 2: M x (R*Rn),  mode 3: Rn x (R*M).
    """
    r, m, n = T.shape
    if mode == 1:
        return T.reshape(r, m * n, order="F")
    if mode == 2:
        return np.transpose(T, (1, 0, 2)).reshape(m, r * n, order="F")
    if mode == 3:
        return np.transpose(T, (2, 0, 1)).reshape(n, r * m, order="F")
    raise ValueError("mode must be 1, 2, or 3")


def _second_moment_left(W, Sigma, R, M, Rn):
    """E[W kron W] of a core (mean outer product + covariance), arranged for the left sweep."""
    Sigma_t = Sigma.reshape(Rn, M, R, Rn, M, R).transpose(1, 2, 4, 5, 0, 3).reshape(R * M * R * M, Rn * Rn)
    K = np.kron(W.T, W.T) + Sigma_t
    return np.transpose(K.reshape(M, R, M, R, Rn, Rn), (0, 2, 1, 3, 4, 5)).reshape(M * M * R * R, Rn * Rn)


def _second_moment_right(W, Sigma, R, M, Rn):
    """E[W kron W] of a core (mean outer product + covariance), arranged for the right sweep."""
    MRn = M * Rn
    Sigma_t = Sigma.reshape(Rn, M, R, Rn, M, R).transpose(2, 5, 0, 1, 3, 4).reshape(R * R, MRn * MRn)
    K = np.kron(W.T, W.T) + Sigma_t.T
    return np.transpose(K.reshape(Rn, M, Rn, M, R, R), (1, 3, 0, 2, 4, 5)).reshape(M * M * Rn * Rn, R * R)


def contract_left(Phis, Ws, N, G0=None):
    """Mean contraction of the cores from the left: G0 -> G0 x core_1 x ... (N x R_next)."""
    G = G0 if G0 is not None else np.ones((N, 1))
    for Phi, W in zip(Phis, Ws):
        G = khatri_rao_rows(Phi, G) @ unfold(W, 3).T
    return G


def contract_right(Phis, Ws, N, G0=None):
    """Mean contraction of the cores from the right: ... x core_D x G0 (N x R_first)."""
    G = G0 if G0 is not None else np.ones((N, 1))
    for Phi, W in zip(Phis[::-1], Ws[::-1]):
        G = khatri_rao_rows(G, Phi) @ unfold(W, 1).T
    return G


def _second_moment_step(H, Phi, Ek, N, target_elems):
    """One step of the second-moment contraction, in row chunks to limit memory."""
    Phi2 = khatri_rao_rows(Phi, Phi)
    A, B, C = Phi2.shape[1], H.shape[1], Ek.shape[1]
    Ek2 = Ek.reshape(A, B * C)
    chunk = min(N, max(1, target_elems // (A * B)))
    if chunk >= N:
        tmp = (Phi2 @ Ek2).reshape(N, B, C)
        return np.matmul(H[:, None, :], tmp)[:, 0, :]
    H_new = np.empty((N, C))
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        tmp = (Phi2[start:end] @ Ek2).reshape(end - start, B, C)
        H_new[start:end] = np.matmul(H[start:end][:, None, :], tmp)[:, 0, :]
    return H_new


def second_moment_left(Phis, Ws, Sigmas, N, H0=None, target_elems=5_000_000):
    """Second-moment (E[g g^T]) contraction of the cores from the left."""
    H = H0 if H0 is not None else np.ones((N, 1))
    for Phi, W, S in zip(Phis, Ws, Sigmas):
        R, M, Rn = W.shape
        H = _second_moment_step(H, Phi, _second_moment_left(unfold(W, 3), S, R, M, Rn), N, target_elems)
    return H


def second_moment_right(Phis, Ws, Sigmas, N, H0=None, target_elems=5_000_000):
    """Second-moment (E[g g^T]) contraction of the cores from the right."""
    H = H0 if H0 is not None else np.ones((N, 1))
    for Phi, W, S in zip(Phis[::-1], Ws[::-1], Sigmas[::-1]):
        R, M, Rn = W.shape
        H = _second_moment_step(H, Phi, _second_moment_right(unfold(W, 1), S, R, M, Rn), N, target_elems)
    return H


def ggT_to_core_matrix(ggT, R_d, M_d, R_next):
    """Rearrange the (left x right) environment product into the (R_d M_d R_next)^2 matrix of one core."""
    n = R_d * M_d * R_next
    return ggT.reshape(R_d, R_d, M_d, R_next, R_next, M_d).transpose(3, 2, 0, 4, 5, 1).reshape(n, n)


def safelog(x):
    """log(x) with x clipped to [1e-10, 1e10]."""
    return np.log(np.clip(x, 1e-10, 1e10))
