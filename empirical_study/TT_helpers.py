import numpy as np


def khatri_rao_rows(A, B):
    return (A[:, :, None] * B[:, None, :]).reshape(A.shape[0], -1)


def compute_G_left(Phis, Ws):
    Ws = matricize_tt_cores(Ws, mode=3, transpose=True)
    G = np.ones((Phis[0].shape[0], 1))

    for k in range(len(Phis)):
        G = khatri_rao_rows(Phis[k], G) @ Ws[k]

    return G

def compute_G_right(Phis, Ws):
    D = len(Phis)
    N = Phis[0].shape[0]

    Ws_mat = matricize_tt_cores(Ws, mode=1, transpose=True)
    Phis_r = Phis[::-1]
    Ws_r   = Ws_mat[::-1]

    G = np.ones((N,1))   # corresponds to G_>D

    for k in range(D):
        KR = khatri_rao_rows(Phis_r[k], G)
        G = KR @ Ws_r[k]

    return G


def generate_tt_cores(D, R=4, M=8, Rs=None, Ms=None, seed=None):
    rng = np.random.default_rng(seed)

    # mode sizes
    if Ms is None:
        Ms = [M] * D

    # ranks: R1=1, RD+1=1 by default
    if Rs is None:
        Rs = [1] + [R] * (D - 1) + [1]

    W_D = []
    for d in range(D):
        W_D.append(rng.standard_normal((Rs[d], Ms[d], Rs[d+1])))

    return W_D

def matricize_tt_cores(W_D, mode=3, transpose=False):
    """
    n-mode unfolding of TT cores (R_d, M_d, R_{d+1}).

    mode = 1,2,3  -> returns W_(mode)
    transpose=True -> returns (W_(mode))^T
    """
    if mode not in (1, 2, 3):
        raise ValueError("mode must be 1, 2, or 3")

    n = mode - 1

    mats = []
    for W in W_D:
        A = np.moveaxis(W, n, 0).reshape(W.shape[n], -1)
        mats.append(A.T if transpose else A)

    return mats


def reshape_sigma(Sigma, Rd, Md, Rnext):
    """
    Sigma: ((Rd*Md*Rnext) x (Rd*Md*Rnext))

    Returns:
      ((Rd*Md)^2 x Rnext^2)
    """
    RM = Rd * Md

    S = Sigma.reshape(RM, Rnext, RM, Rnext)      # (RM, R+, RM, R+)
    S = S.transpose(0, 2, 1, 3)                  # (RM, RM, R+, R+)

    return S.reshape(RM*RM, Rnext*Rnext)


#BURAYA KADAR DEBUG ETTIM

# def Ew_kronW(W, Sigma, Rd, Md, Rn):
#     return np.kron(W, W) + reshape_sigma(Sigma, Rd, Md, Rn)



def Ew_kronW(W, Sigma):
    RM, Rn = W.shape
    return np.kron(W, W) + Sigma.reshape(RM, Rn, RM, Rn).transpose(0,2,1,3).reshape(RM*RM, Rn*Rn)

def compute_H_left(Phis, Ws, Sigmas):
    N = Phis[0].shape[0]
    H = np.ones((N,1))  # H_<1 (since R1=1)

    for k in range(len(Phis)):
        Phi2 = khatri_rao_rows(Phis[k], Phis[k])
        H = khatri_rao_rows(Phi2, H) @ Ew_kronW(Ws[k], Sigmas[k])

    return H

def compute_H_right(Phis, Ws, Sigmas):
    N = Phis[0].shape[0]
    H = np.ones((N,1))  # H_>D (since R_{D+1}=1)

    Ph_r = Phis[::-1]
    Wm_r = Ws[::-1]
    Sg_r = Sigmas[::-1]

    for k in range(len(Ph_r)):
        Phi2 = khatri_rao_rows(Ph_r[k], Ph_r[k])
        H = khatri_rao_rows(Phi2, H) @ Ew_kronW(Wm_r[k], Sg_r[k])

    return H


