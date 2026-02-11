import os, sys

sys.path.append(os.getcwd())
from config import *  # Import everything from config.py
from TT_helpers import (generate_tt_cores, 
                        init_sigmas,
                        compute_G_left, 
                        compute_G_right,
                        compute_H_left,
                        compute_H_right,
                        khatri_rao_rows, 
                        reshape_permute_ggT_to_sigma,
                        pure_power_features,
                        generate_delta_lambda)

D = 4
Ms = [2,2, 5, 2]       # mode sizes per core (length D)
Rs = [1, 3, 2, 2, 1]  
N = 100

rng = np.random.default_rng(42)
X = rng.random((N, D))
W_D_true = generate_tt_cores(D=D, Ms=Ms, Rs=Rs, seed=1, st = 10)
Phi = pure_power_features(X,Ms=Ms) 
y = compute_G_left(Phi, W_D_true, N) + 1e-3 * rng.standard_normal((N, 1))
Sigma = init_sigmas(D=D, Ms=Ms, Rs=Rs)
tau = 1
deltas, lambdas = generate_delta_lambda(Ms=Ms, Rs=Rs)

W_D = generate_tt_cores(D=D, Ms=Ms, Rs=Rs)
for it in range (10):
    for d in range(D):
        G_left = compute_G_left(Phi[:d], W_D[:d], N)
        G_right = compute_G_right(Phi[d+1:], W_D[d+1:], N) 
        Gd = khatri_rao_rows(khatri_rao_rows(G_right, Phi[d]), G_left)
        H_left = compute_H_left(Phi[:d], W_D[:d], Sigma[:d], N)
        H_right = compute_H_right(Phi[d+1:], W_D[d+1:], Sigma[d+1:],N)
        ggT = reshape_permute_ggT_to_sigma(khatri_rao_rows(H_left, Phi[d]).T @ khatri_rao_rows( H_right, Phi[d]), 
                                        R_d=W_D[d].shape[0],
                                        M_d=W_D[d].shape[1],
                                        R_next=W_D[d].shape[2])
        
        Sigma[d] = np.linalg.inv(tau * ggT + np.kron(np.kron(lambdas[d+1]* np.eye(Rs[d+1]), deltas[d]*np.eye(Ms[d])), lambdas[d]*np.eye(Rs[d])))
        W_D[d] = (tau * Sigma[d] @ Gd.T @ y).reshape(W_D[d].shape[0],W_D[d].shape[1], W_D[d].shape[2])  
        del G_left, G_right, Gd, H_left, H_right, ggT

print(G_left)
