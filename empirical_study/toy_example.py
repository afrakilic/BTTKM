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
                        tt_core_unfolding)

D = 3
Ms = [10, 10, 10]       # mode sizes per core (length D)
Rs = [1, 3, 3, 1]  
N = 100

rng = np.random.default_rng(42)
X = rng.random((N, D))
W_D_true = generate_tt_cores(D=D, Ms=Ms, Rs=Rs, seed=1)
Phi = pure_power_features(X,Ms=Ms) 
y = compute_G_left(Phi, W_D_true, N) + 1e-1 * rng.standard_normal((N, 1))

#initialization
Sigma = init_sigmas(D=D, Ms=Ms, Rs=Rs)
tau = 1

g0 = 1e-6; h0 = 1e-6; c0 = 1e-6; d0= 1e-6

g0s = [g0 * np.ones(m) for m in Ms]; gNs = [g.copy() for g in g0s]
h0s = [h0 * np.ones(m) for m in Ms]; hNs = [h.copy() for h in h0s]
c0s = [c0 * np.ones(r) for r in Rs]; cNs = [c.copy() for c in c0s]
d0s = [d0 * np.ones(r) for r in Rs]; dNs = [d.copy() for d in d0s]

deltas0  = [g / h for g, h in zip(g0s, h0s)]; deltas  = [d.copy() for d in deltas0]
lambdas0 = [c / d for c, d in zip(c0s, d0s)]; lambdas = [l.copy() for l in lambdas0]

W_D = generate_tt_cores(D=D, Ms=Ms, Rs=Rs, seed=1)
#model learning
for it in range (10):
    for d in range(D):

        #tensor core update 
        G_left = compute_G_left(Phi[:d], W_D[:d], N)
        G_right = compute_G_right(Phi[d+1:], W_D[d+1:], N) 
        Gd = khatri_rao_rows(khatri_rao_rows(G_right, Phi[d]), G_left)
        H_left = compute_H_left(Phi[:d], W_D[:d], Sigma[:d], N)
        H_right = compute_H_right(Phi[d+1:], W_D[d+1:], Sigma[d+1:],N)
        ggT = reshape_permute_ggT_to_sigma(khatri_rao_rows(H_left, Phi[d]).T @ khatri_rao_rows( H_right, Phi[d]), 
                                        R_d=W_D[d].shape[0],
                                        M_d=W_D[d].shape[1],
                                        R_next=W_D[d].shape[2])
        # ggT = Gd.T@Gd
        Sigma[d] = np.linalg.pinv(tau * ggT + np.kron(np.kron(lambdas[d+1]* np.eye(Rs[d+1]), deltas[d]*np.eye(Ms[d])), lambdas[d]*np.eye(Rs[d])))
        W_D[d] = (tau * Sigma[d] @ Gd.T @ y).reshape(W_D[d].shape, order="F")


        #delta updates 

        Kdel = np.kron(lambdas[d+1] * np.eye(lambdas[d+1].size), lambdas[d] * np.eye(lambdas[d].size))

        Sigma_cored = np.diag(Sigma[d]).reshape(W_D[d].shape, order="F")
        Sigma2 = tt_core_unfolding(Sigma_cored, mode=2)[0]
        W2 = tt_core_unfolding(W_D[d], mode=2)[0]
        ewdel2 = np.diag(W2 @ Kdel @ W2.T) + Sigma2 @ np.kron(lambdas[d+1], lambdas[d])
        gNs[d] = g0s[d] + 0.5 * (lambdas[d].size * lambdas[d+1].size)
        hNs[d] = h0s[d] + 0.5 * ewdel2
        deltas[d] = gNs[d] / hNs[d]


        # #lambda updates 
        if d>0:
            Klam1= np.kron(lambdas[d+1]*np.eye(lambdas[d+1].size),deltas[d]*np.eye(deltas[d].size))
            Klam2= np.kron(deltas[d-1]*np.eye(deltas[d-1].size), lambdas[d-1]*np.eye(lambdas[d-1].size))
            Sigma1= tt_core_unfolding(Sigma_cored, mode=1)[0]
            W1 =  tt_core_unfolding(W_D[d], mode=1)[0]
            ewdel1 = np.diag(W1 @ Klam1 @ W1.T) + Sigma1 @ np.kron(lambdas[d+1],deltas[d])
            Sigma_coreb = np.diag(Sigma[d-1]).reshape(W_D[d-1].shape, order="F")
            Sigma3 = tt_core_unfolding(Sigma_coreb, mode=3)[0]
            W3 =  tt_core_unfolding(W_D[d-1], mode=3)[0]
            ewdel3 = np.diag(W3 @ Klam2 @ W3.T) + Sigma3 @ np.kron(deltas[d-1],lambdas[d-1])
            dNs[d] = d0s[d] + 0.5 * (ewdel1 + ewdel3)
            cNs[d] = c0s[d] + 0.5 * (deltas[d-1].size * lambdas[d-1].size + lambdas[d+1].size * deltas[d].size) 

            lambdas[d] = cNs[d] /dNs[d]

y_hat = compute_G_left(Phi, W_D, N)
x = np.arange(len(y))

# RMSE
rmse = np.sqrt(np.mean((y_hat - y) ** 2))
print("RMSE:", rmse)

# Scatter plot
plt.figure()
plt.scatter(x, y, label="y (true)")
plt.scatter(x, y_hat, label="y_hat (predicted)")

plt.xlabel("Observation #")
plt.ylabel("Value")
plt.title(f"y and y_hat vs observations (RMSE = {rmse:.4f})")
plt.legend()
plt.show()