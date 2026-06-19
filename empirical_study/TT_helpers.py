
import os, sys

sys.path.append(os.getcwd())
from config import *  # Import everything from config.py



def generate_tt_cores(D, R=4, M=8, Rs=None, Ms=None, seed=None, st = 1):
    rng = np.random.default_rng(seed)

    # mode sizes
    if Ms is None:
        Ms = [M] * D

    # ranks: R0=1, R_D=1 by default
    if Rs is None:
        Rs = [1] + [R] * (D - 1) + [1]

    W_D = []
    for d in range(D):
        W = st*rng.standard_normal((Rs[d], Ms[d], Rs[d+1]))

        # Frobenius normalization (same as MATLAB: norm(W(:)))
        #W /= np.linalg.norm(W)

        W_D.append(W)

    return W_D

def khatri_rao_rows(A, B):
    return (A[:, :, None] * B[:, None, :]).reshape(A.shape[0], -1)


def tt_core_unfolding(T_list, mode):
    """
    Apply mode-n unfolding to a list of 3-way tensors T[r,m,n].

    Parameters
    ----------
    T_list : list of np.ndarray
        Each element has shape (R, M, Rn).
    mode : int
        1, 2, or 3.

    Returns
    -------
    unfolded_list : list of np.ndarray
        Same length as T_list, each unfolded according to mode.
    """

    single_tensor = False
    if isinstance(T_list, np.ndarray):
        T_list = [T_list]
        single_tensor = True

    if mode not in (1, 2, 3):
        raise ValueError("mode must be 1, 2, or 3")

    unfolded = []

    for T in T_list:
        if T.ndim != 3:
            raise ValueError("Each tensor must be 3D")

        r, m, n = T.shape  # (R, M, Rn)

        if mode == 1:
            # R x (M*Rn), columns ~ kron(Rn, M)
            U = T.reshape(r, m * n, order="F")

        elif mode == 2:
            # M x (R*Rn), columns ~ kron(Rn, R)
            T_perm = np.transpose(T, (1, 0, 2))  # (M, R, Rn)
            U = T_perm.reshape(m, r * n, order="F")

        else:  # mode == 3
            # Rn x (R*M), columns ~ kron(M, R)
            T_perm = np.transpose(T, (2, 0, 1))  # (Rn, R, M)
            U = T_perm.reshape(n, r * m, order="F")

        unfolded.append(U)

    return unfolded


def Ew_kronW_left(W, Sigma, R, M, Rn):
    """
    W     : (R*M, Rn)
    Sigma : (R*M*Rn, R*M*Rn)

    Returns:
        (R*M)*(R*M) x (Rn*Rn) matrix:
        kron(W, W) + Sigma reshaped/permuted to (RM*RM, Rn*Rn)
    """
    RM = R * M

    # reshape Sigma: (R,M,Rn, R,M,Rn) with indices (r,m,rn, r',m',rn')
    # permute to:    (r,m, r',m', rn,rn')
    Sigma_tilde = (
        Sigma.reshape(Rn, M, R, Rn, M, R)
             .transpose(1, 2, 4, 5, 0, 3)
             .reshape(RM * RM, Rn * Rn)
    )

    K = np.kron(W.T, W.T)  + Sigma_tilde
    T = K.reshape(M, R, M, R, Rn, Rn)
    T_perm = np.transpose(T, (0, 2, 1, 3, 4, 5))
    result= T_perm.reshape(M*M*R*R, Rn*Rn)

    return result



def Ew_kronW_right(W, Sigma, R, M, Rn):
    """
    W     : (R*M, Rn)
    Sigma : (R*M*Rn, R*M*Rn)

    Returns:
        (R*M)*(R*M) x (Rn*Rn) matrix:
        kron(W, W) + Sigma reshaped/permuted to (RM*RM, Rn*Rn)
    """
    MRn = M * Rn

    Sigma_tilde = (
        Sigma.reshape(Rn, M, R, Rn, M, R)
             .transpose(2, 5, 0, 1, 3, 4)
             .reshape(R * R, MRn * MRn)
    )
    K = np.kron(W.T, W.T)  + Sigma_tilde.T
    T = K.reshape(Rn, M, Rn, M, R, R)
    T_perm = np.transpose(T, (1, 3, 0, 2, 4, 5))
    result = T_perm.reshape(M*M*Rn*Rn, R*R)
    return result




def init_sigmas(D, R=4, M=8, Rs=None, Ms=None):
    """
    Generate Sigma_d covariance matrices for vec(TT cores),
    matching the interface of generate_tt_cores.

    Each Sigma_d is initialized as an identity matrix of size
    (R_d * M_d * R_{d+1}) x (R_d * M_d * R_{d+1}).

    Args:
        D: number of TT cores
        R: default TT rank (used if Rs is None)
        M: default mode size (used if Ms is None)
        Rs: optional list of TT ranks, length D+1
        Ms: optional list of mode sizes, length D
        seed: optional (kept for API symmetry; not used for identity init)

    Returns:
        Sigmas: list of D covariance matrices
    """

    # mode sizes
    if Ms is None:
        Ms = [M] * D

    # ranks: R0=1, R_{D}=1 by default
    if Rs is None:
        Rs = [1] + [R] * (D - 1) + [1]

    Sigmas = []
    for d in range(D):
        n_d = Rs[d] * Ms[d] * Rs[d + 1]
        Sigmas.append(1e-4* np.eye(n_d))

    return Sigmas



def reshape_permute_ggT_to_sigma(ggT, R_d, M_d, R_next):

    """
    ggT: shape (M_d*R_d^2, M_d*R_next^2)
    returns Sigma: shape (R_d*M_d*R_next, R_d*M_d*R_next)
    """

    n = R_d * M_d * R_next
    
    # (m, r_d, r_d', m', r_next, r_next')
    T = ggT.reshape(R_d, R_d, M_d, R_next, R_next, M_d)

    # -> (r_d, m, r_next, r_d', m', r_next')
    T = T.transpose(3, 2, 0, 4, 5, 1)

    # flatten to square
    Sigma = T.reshape(n, n)
    return Sigma




def pure_power_features_TT(X, M=8, Ms=None, eps=1e-12):
    """
    Pure-power polynomial features.

    X  : (N, D)
    M  : fixed mode size (used if Ms is None)
    Ms : optional list of per-dimension mode sizes

    Returns:
        Phis : list of length D
               Phi[d] has shape (N, M_d), unit-norm rows
    """
    X = np.asarray(X)
    N, D = X.shape

    if Ms is None:
        Ms = [M] * D

    Phis = []
    for d in range(D):
        powers = np.arange(Ms[d])
        Phi = X[:, d:d+1] ** powers[None, :]
        Phi /= np.linalg.norm(Phi, axis=1, keepdims=True) + eps
        Phi = Phi  +0.2
        Phis.append(Phi)

    return Phis

def compute_G_left(Phis, Ws, N, G0=None):
    if (Phis is None or len(Phis) == 0) or (Ws is None or len(Ws) == 0):
        return G0 if G0 is not None else np.ones((N, 1))

    Ws_mat = tt_core_unfolding(Ws, mode=3)
    N_ = N if N is not None else Phis[0].shape[0]
    G = G0 if G0 is not None else np.ones((N_, 1))

    for k in range(len(Phis)):
        G = khatri_rao_rows(Phis[k], G) @ Ws_mat[k].T

    return G


def compute_G_right(Phis, Ws, N, G0=None):
    if (Phis is None or len(Phis) == 0) or (Ws is None or len(Ws) == 0):
        return G0 if G0 is not None else np.ones((N, 1))

    D = len(Phis)
    N_ = N if N is not None else Phis[0].shape[0]
    Ws_mat = tt_core_unfolding(Ws, mode=1)
    Phis_r = Phis[::-1]
    Ws_r   = Ws_mat[::-1]
    G = G0 if G0 is not None else np.ones((N_, 1))

    for k in range(D):
        G = khatri_rao_rows(G, Phis_r[k]) @ Ws_r[k].T

    return G


def compute_H_left(Phis, Ws, Sigmas, N, H0=None):
    if (Phis is None or len(Phis) == 0) or (Ws is None or len(Ws) == 0) or (Sigmas is None or len(Sigmas) == 0):
        return H0 if H0 is not None else np.ones((N, 1))

    N_ = N if N is not None else Phis[0].shape[0]
    H = H0 if H0 is not None else np.ones((N_, 1))
    Ws_mat = tt_core_unfolding(Ws, mode=3)

    for k in range(len(Phis)):
        Phi2 = khatri_rao_rows(Phis[k], Phis[k])
        R, M, Rn = Ws[k].shape
        H = khatri_rao_rows(Phi2, H) @ Ew_kronW_left(Ws_mat[k], Sigmas[k], R=R, M=M, Rn=Rn)

    return H


def compute_H_right(Phis, Ws, Sigmas, N, H0=None):
    if (Phis is None or len(Phis) == 0) or (Ws is None or len(Ws) == 0) or (Sigmas is None or len(Sigmas) == 0):
        return H0 if H0 is not None else np.ones((N, 1))

    N_ = N if N is not None else Phis[0].shape[0]
    H = H0 if H0 is not None else np.ones((N_, 1))
    Ws_mat = tt_core_unfolding(Ws, mode=1)
    Ws_r = Ws[::-1]
    Ph_r = Phis[::-1]
    Wm_r = Ws_mat[::-1]
    Sg_r = Sigmas[::-1]

    for k in range(len(Ph_r)):
        Phi2 = khatri_rao_rows(Ph_r[k], Ph_r[k])
        R, M, Rn = Ws_r[k].shape
        H = khatri_rao_rows(Phi2, H) @ Ew_kronW_right(Wm_r[k], Sg_r[k], R=R, M=M, Rn=Rn)

    return H
