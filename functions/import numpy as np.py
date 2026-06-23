import numpy as np
import scipy.io as sio
from functions.TT_helpers import (legendre_basis,
                                  initialize_tt_cores,
                                  compute_G_left,
                                  compute_G_right,
                                  tt_core_unfolding)

data = sio.loadmat('/Users/hakilic/Downloads/tt_data.mat')

x_mat = data['x']       # (N, D)
P_mat = data['P']       # (N, I, D)
W_arr = data['W_arr']   # (I, R^2, D)
B_arr = data['B_arr']   # (N, R^2, D)

N, D = x_mat.shape
I    = P_mat.shape[1]
R    = W_arr.shape[1]  # infer R from saved data

Phis = legendre_basis(x_mat, I=I)

W_D = initialize_tt_cores(N=N, D=D, I=I, R=3)


# --- Compare W_D vs W_arr ---
print("=== W shape check ===")
for d in range(D):
    w_py  = W_D[d].reshape(I, -1)           # (I, cols)
    w_mat = W_arr[:, :w_py.shape[1], d]     # (I, cols)
    diff  = np.abs(w_py - w_mat).max()
    print(f"W[{d}]: python={w_py.shape}, matlab={w_mat.shape}, max_diff={diff:.4e}")

# --- Compare B vs B_arr ---
B = compute_G_right(Phis, W_D, N)
print("\n=== B shape check ===")
for d in range(D):
    b_py  = B[d].reshape(N, -1)             # (N, cols)
    b_mat = B_arr[:, :b_py.shape[1], d]     # (N, cols)
    diff  = np.abs(b_py - b_mat).max()
    print(f"B[{d}]: python={b_py.shape}, matlab={b_mat.shape}, max_diff={diff:.4e}")