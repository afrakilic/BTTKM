
"""
Non-linear model using a Tensor Train (TT) kernel machine of the form y = Φw + e,
where y is the output, Φ is the feature matrix, w is the weight vector, and e is the error term.

The `bttkm` class provides `train` and `predict` methods. Unlike the CP decomposition (btnkm),
the weight tensor is parameterized as a Tensor Train (TT) decomposition, where each TT-core
is updated sequentially via variational inference. Training iteratively updates TT-cores,
noise precision (tau), and regularization parameters (lambda, delta) with optional rank pruning
and R-squared monitoring. Predictions include uncertainty estimates via a Student-t predictive
distribution.

Dependencies and configurations are defined in `config.py`.
"""

import os, sys

sys.path.append(os.getcwd())
from config import *  # Import everything from config.py
from functions.TT_helpers import (generate_tt_cores, 
                        init_sigmas,
                        compute_G_left, 
                        compute_G_right,
                        compute_H_left,
                        compute_H_right,
                        khatri_rao_rows, 
                        reshape_permute_ggT_to_sigma,
                        pure_power_features_TT,
                        tt_core_unfolding, 
                        initialize_tt_cores,
                        legendre_basis)
from functions.utils import (safelog)




class bttkm:
    """A general non-linear using kernel machine y = \Phi w + e, where y is output, \Phi is the
    input matrix, w is the model weigths, and e is the error term."""

    def __init__(self, num_features: int) -> None:
        """Initialize the model.

        Parameters
        ----------
        """
        self.num_features = int

    def train(
        self,
        features: np.ndarray,
        target: np.ndarray,
        input_dimension: int,
        max_rank: int,
        shape_parameter_tau: float = 1,
        scale_parameter_tau: float = 1,
        shape_parameter_lambda: np.ndarray = None,
        scale_parameter_lambda: np.ndarray = None,
        shape_parameter_delta: np.ndarray = None,
        scale_parameter_delta: np.ndarray = None,
        max_iter: int = 100,
        seed: int = 2,
        lambda_update: bool = True,
        prune_rank: bool = False,  # Optional rank pruning
        rank_tol: int = 1e-1,  # this is threshold to keep the certain rank (i.e. columns in factor matrices)in terms of explained variance
        delta_update: bool = True,
        precision_update: bool = True,
        lower_bound_tol: int = 1e-4,
        plot_results: bool = True,
        classification: bool = False,
    ) -> None:
        np.random.seed(seed)
        I = input_dimension
        R = max_rank
        X = features
        Y = target
        N = X.shape[0]
        D = X.shape[1]
        Ms = [I] * D  
        Rs = [1] + [R] * (D - 1) + [1] 

        tau = shape_parameter_tau / scale_parameter_tau
        a0 = shape_parameter_tau
        b0 = scale_parameter_tau

        g0 = shape_parameter_delta; h0 = scale_parameter_delta; c0 = shape_parameter_lambda; d0= scale_parameter_lambda

        g0s = [g0 * np.ones(m) for m in Ms]; gNs = [g.copy() for g in g0s]
        h0s = [h0 * np.ones(m) for m in Ms]; hNs = [h.copy() for h in h0s]
        c0s = [c0 * np.ones(r) for r in Rs]; cNs = [c.copy() for c in c0s]
        d0s = [d0 * np.ones(r) for r in Rs]; dNs = [d.copy() for d in d0s]

        deltas0  = [g / h for g, h in zip(g0s, h0s)]; deltas  = [d.copy() for d in deltas0]
        lambdas0 = [c / d for c, d in zip(c0s, d0s)]; lambdas = [l.copy() for l in lambdas0]


       
        # W_D = initialize_tt_cores(N = N, D=D, I =I, R=R)
        Sigma = init_sigmas(D=D, Ms=Ms, Rs=Rs, N=N)

        W_D= generate_tt_cores(D=D, Ms=Ms, Rs=Rs, seed=1)
        Phi= pure_power_features_TT(X, Ms=Ms)

        # Phi = legendre_basis(X, I=I)
        pbar = trange(max_iter, desc="Running", leave=True)
        Fit_history = []
        LB = np.zeros(max_iter)  # lowerbound
        LBRelChan = 0  
        LB_terms = {f"temp{i}": np.zeros(max_iter) for i in range(1, 7)}


        G_left_env = [None] * (D + 1)
        H_left_env = [None] * (D + 1)
        G_left_env[0] = np.ones((N, 1))
        H_left_env[0] = np.ones((N, 1))

        for it in pbar:

            G_right_env = [None] * (D + 1)
            H_right_env = [None] * (D + 1)
            G_right_env[D] = np.ones((N, 1))
            H_right_env[D] = np.ones((N, 1))
            for k in range(D - 1, -1, -1):
                G_right_env[k] = compute_G_right([Phi[k]], [W_D[k]], N, G0=G_right_env[k + 1])
                H_right_env[k] = compute_H_right([Phi[k]], [W_D[k]], [Sigma[k]], N, H0=H_right_env[k + 1])  # G0 → H0
            for d in range(D):
                
                #     #tensor core update 
                # G_left = compute_G_left(Phi[:d], W_D[:d], N)
                # G_right = compute_G_right(Phi[d+1:], W_D[d+1:], N) 
                # H_left = compute_H_left(Phi[:d], W_D[:d], Sigma[:d], N)
                # H_right = compute_H_right(Phi[d+1:], W_D[d+1:], Sigma[d+1:],N)
                # Gd_1 = khatri_rao_rows(khatri_rao_rows(G_right, Phi[d]), G_left)

                G_left_curr  = G_left_env[d]
                G_right_curr = G_right_env[d + 1]
                H_left_curr  = H_left_env[d]
                H_right_curr = H_right_env[d + 1]

                Gd  = khatri_rao_rows(khatri_rao_rows(G_right_curr, Phi[d]), G_left_curr)

                ggT = reshape_permute_ggT_to_sigma( khatri_rao_rows(H_left_curr, Phi[d]).T @ khatri_rao_rows(H_right_curr, Phi[d]),
                                                R_d=W_D[d].shape[0],
                                                M_d=W_D[d].shape[1],
                                                R_next=W_D[d].shape[2])
            
                # Sigma[d] = np.linalg.pinv(tau * ggT + np.kron(np.kron(lambdas[d+1]* np.eye(Rs[d+1]), deltas[d]*np.eye(Ms[d])), lambdas[d]*np.eye(Rs[d])))

                kron_term = np.kron(np.kron(lambdas[d+1] * np.eye(Rs[d+1]),
                             deltas[d]    * np.eye(Ms[d])),
                             lambdas[d]   * np.eye(Rs[d]))

                kron_term = np.clip(kron_term, 1e-6, 1e3)

                A = tau * ggT + kron_term
 

                # A = tau * ggT + np.kron(np.kron(lambdas[d+1] * np.eye(Rs[d+1]),
                #                                 deltas[d]    * np.eye(Ms[d])),
                #                                 lambdas[d]   * np.eye(Rs[d]))

                eigvals, eigvecs = np.linalg.eigh(A)
                eigvals_clipped = np.maximum(eigvals, 1e-10 * eigvals.max())
                Sigma[d] = eigvecs @ np.diag(1.0 / eigvals_clipped) @ eigvecs.T
                
                W_D[d] = (tau * Sigma[d] @ Gd.T @ Y).reshape(W_D[d].shape, order="F")


                G_left_env[d + 1] = compute_G_left([Phi[d]], [W_D[d]], N, G0=G_left_env[d])
                H_left_env[d + 1] = compute_H_left([Phi[d]], [W_D[d]], [Sigma[d]], N, H0=H_left_env[d])  # G0 → H0
                 # #lambda updates 
                if it > 5 and lambda_update: 
                    if d > 0: 

                        Klam1= np.kron(lambdas[d+1]*np.eye(lambdas[d+1].size),deltas[d]*np.eye(deltas[d].size))
                        Klam2= np.kron(deltas[d-1]*np.eye(deltas[d-1].size), lambdas[d-1]*np.eye(lambdas[d-1].size))
                      
                        Sigma_cored = np.diag(Sigma[d]).reshape(W_D[d].shape, order="F")
                        Sigma1= tt_core_unfolding(Sigma_cored, mode=1)[0]
                        W1 =  tt_core_unfolding(W_D[d], mode=1)[0]
                        ewdel1 = np.diag(W1 @ Klam1 @ W1.T) + Sigma1 @ np.kron(lambdas[d+1],deltas[d]
                                                                               )
                        Sigma_coreb = np.diag(Sigma[d-1]).reshape(W_D[d-1].shape, order="F")
                        Sigma3 = tt_core_unfolding(Sigma_coreb, mode=3)[0]
                        W3 =  tt_core_unfolding(W_D[d-1], mode=3)[0]
                        ewdel3 = np.diag(W3 @ Klam2 @ W3.T) + Sigma3 @ np.kron(deltas[d-1],lambdas[d-1])
                    
                        cNs[d] = c0s[d] + 0.5 * (deltas[d-1].size * lambdas[d-1].size + lambdas[d+1].size * deltas[d].size) 
                        dNs[d] = d0s[d] + 0.5 * (ewdel1 + ewdel3)
    
                        
                        lambdas[d] = cNs[d] / dNs[d]
                        lambdas[d][lambdas[d] < 1e-5] = 1e-5

            
                #delta updates 
                if it > 5 and delta_update:
                    Kdel = np.kron(lambdas[d+1] * np.eye(lambdas[d+1].size), lambdas[d] * np.eye(lambdas[d].size))
                    Sigma_cored = np.diag(Sigma[d]).reshape(W_D[d].shape, order="F")
                    Sigma2 = tt_core_unfolding(Sigma_cored, mode=2)[0]
                    W2 = tt_core_unfolding(W_D[d], mode=2)[0]
                    ewdel2 = np.diag(W2 @ Kdel @ W2.T) + Sigma2 @ np.kron(lambdas[d+1], lambdas[d])
                    gNs[d] = g0s[d] + 0.5 * (lambdas[d].size * lambdas[d+1].size)
                    hNs[d] = h0s[d] + 0.5 * ewdel2
                    deltas[d]  = gNs[d] / hNs[d]
                    deltas[d][deltas[d] < 1e-5] = 1e-5
            
            y_hat = compute_G_left(Phi, W_D, N)
            
            if it > 5 and  precision_update:
                a_N = a0 + (N / 2)
                YY = Y.T @ Y
                YG = -2 * np.dot(Y.T, y_hat)
                GG = np.sum(compute_H_left(Phi, W_D, Sigma, N), axis=0)
                b_N = b0 + (0.5 * (YY + YG + GG))
            else:
                a_N = a0
                b_N = b0

            YY = Y.T @ Y
            YG = -2 * np.dot(Y.T, y_hat)
            GG = np.sum(compute_H_left(Phi, W_D, Sigma, N), axis=0)
            err = float(YY + YG + GG)

            tau = a_N / b_N

            # Rank pruning (optional)

            W_D_mode1 = tt_core_unfolding(W_D, mode=1)
            W_D_mode2 = tt_core_unfolding(W_D, mode=2)
            W_D_mode3 = tt_core_unfolding(W_D, mode=3)

            if it > 0 and it % 5 == 0 and prune_rank:
                for d in range(1, D):
                    stack = np.hstack((W_D_mode3[d-1], W_D_mode1[d]))
                    comPower = np.diag(stack @ stack.T)
                    var_explained = comPower / np.sum(comPower) * 100
                    keep_R = var_explained >= 1e-2

                    if np.all(keep_R):
                        continue

                    idx = np.where(keep_R)[0]

                    R_d_old = Rs[d]      # == len(keep_R) by construction
                    I_prev  = Ms[d-1]
                    R_prev  = Rs[d-1]
                    I_d     = Ms[d]
                    R_next  = Rs[d+1]

                    # trim cores 
                    W_D[d-1] = W_D[d-1][:, :, keep_R]   # mode-3 of core d-1
                    W_D[d]   = W_D[d][keep_R, :, :]      # mode-1 of core d

                    # update Rs
                    Rs[d] = int(np.sum(keep_R))

                    # trim all lambda-related vectors at boundary d 
                    # all have current length R_d_old
                    lambdas[d] = lambdas[d][keep_R]
                    cNs[d]     = cNs[d][keep_R]
                    dNs[d]     = dNs[d][keep_R]
                    c0s[d]     = c0s[d][keep_R]
                    d0s[d]     = d0s[d][keep_R]

                    # trim Sigma[d-1]: R_d is mode-3 
                    flat_dm1 = np.concatenate([
                        rb * I_prev * R_d_old + ib * R_d_old + idx
                        for rb in range(R_prev)
                        for ib in range(I_prev)
                    ])
                    Sigma[d-1] = Sigma[d-1][np.ix_(flat_dm1, flat_dm1)]

                    # trim Sigma[d]: R_d is mode-1 
                    flat_d = np.concatenate([
                        np.arange(r * I_d * R_next, (r + 1) * I_d * R_next) for r in idx
                    ])
                    Sigma[d] = Sigma[d][np.ix_(flat_d, flat_d)]


             # FIT METRIC
            if classification:
                Fit = accuracy_score(Y, np.sign(y_hat))
            else:
                tss = np.sum((Y.reshape(-1,1) - np.mean(Y))**2)
                rss = np.sum((Y.reshape(-1,1) - y_hat)**2)
                Fit = 1 - (rss / tss)  # R-squared

            Fit_history.append(float(Fit))


            # LOWER BOUND

            temp1 = -0.5 * (tau) * err    

            temp2 = 0
            for d in range(D):
                matrix_d = np.kron(np.kron(lambdas[d+1] * np.eye(Rs[d+1]), deltas[d] * np.eye(Ms[d])), lambdas[d] * np.eye(Rs[d])) @ (
                    np.outer(W_D[d].ravel(order="F"), W_D[d].ravel(order="F"))
                    + Sigma[d]
                )
                temp2 += -0.5 * np.trace(matrix_d)

            temp3 = 0.0
            for W in Sigma:
                sign, logdet = np.linalg.slogdet(W)
                if sign <= 0:
                    # Sigma should be PD; if not, pinv gave a bad result
                    logdet = -1e10  
                temp3 += 0.5 * logdet

            temp4 = sum(
                np.sum(cNs[d] * (1 - safelog(dNs[d]) - (d0s[d] / dNs[d])))
                for d in range(1,D)
            )
            temp5 = sum(
                np.sum(
                    np.array(gNs[d]) * (1 - safelog(np.array(hNs[d])) - (h0s[d] / np.array(hNs[d])))
                )
                for d in range(len(gNs))
            )
            temp6 = a_N * (1 - safelog(b_N) - (b0 / b_N))  # + gammaln(a_N)

            LB_terms["temp1"][it] = temp1
            LB_terms["temp2"][it] = temp2
            LB_terms["temp3"][it] = temp3
            LB_terms["temp4"][it] = temp4
            LB_terms["temp5"][it] = temp5
            LB_terms["temp6"][it] = temp6

            LB[it] = temp1 + temp2 + temp3 + temp4 + temp5 + temp6



            # CONVERGENCE CHECK
            if it > 2:
                LBRelChan = (LB[it] - LB[it - 1]) / LB[2]

            else:
                LBRelChan = np.nan

            # Display progress for every iteration
            pbar.set_postfix(
                {
                    "RelChan": f"{LBRelChan:.2e}",
                    "Fit": f"{Fit:.4f}",
                    "err": f"{err:.1e}"
                }
            )
            # Convergence checks
            if it > 5:
                if abs(LBRelChan) < lower_bound_tol:
                    print("======= Converged =======")
                    break

        self.LB = LB[:it+1]
        self.Fit_history = Fit_history

        if plot_results:
        
            self.LB = LB[:it+1]
            self.Fit_history = Fit_history
            self.LB_terms = {k: v[:it+1] for k, v in LB_terms.items()}

            if plot_results:
                labels = {
                    "temp1": r"$-\frac{\tau}{2}\|y - \hat{y}\|^2$" + "\n(likelihood)",
                    "temp2": r"$-\frac{1}{2}\mathrm{tr}(\Lambda W W^\top)$" + "\n(prior on W)",
                    "temp3": r"$\frac{1}{2}\log|\Sigma|$" + "\n(entropy of W)",
                    "temp4": r"$\sum c_N(1 - \log d_N)$" + "\n(lambda term)",
                    "temp5": r"$\sum g_N(1 - \log h_N)$" + "\n(delta term)",
                    "temp6": r"$a_N(1 - \log b_N)$" + "\n(tau term)",
                }
                colors = ["steelblue", "darkorange", "forestgreen", "orchid", "tomato", "goldenrod"]

                fig, axes = plt.subplots(2, 4, figsize=(18, 7))
                axes = axes.flatten()

                for i, (key, color) in enumerate(zip(LB_terms.keys(), colors)):
                    axes[i].plot(self.LB_terms[key], color=color, linewidth=2)
                    axes[i].set_title(labels[key], fontsize=9)
                    axes[i].set_xlabel("Iteration", fontsize=8)
                    axes[i].grid(True, alpha=0.3)

                # 7th panel: total ELBO
                axes[6].plot(self.LB, color="black", linewidth=2)
                axes[6].set_title("Total ELBO", fontsize=9)
                axes[6].set_xlabel("Iteration", fontsize=8)
                axes[6].grid(True, alpha=0.3)

                # 8th panel: R² fit
                axes[7].plot(Fit_history, color="teal", linewidth=2)
                axes[7].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
                axes[7].set_title("Training Fit (R²)", fontsize=9)
                axes[7].set_xlabel("Iteration", fontsize=8)
                axes[7].grid(True, alpha=0.3)

                plt.suptitle("ELBO Terms per Iteration", fontsize=12, fontweight="bold")
                plt.tight_layout()
                plt.savefig("elbo_terms.png", dpi=150)
                plt.show()
        
        self.W_D = W_D
        self.V = Sigma
        self.a = a_N
        self.b = b_N
        self.N = N 

        return W_D, lambdas, deltas, tau

    def predict(
            self,
            features: np.ndarray,
            input_dimension: int,
            rank: int,
            true_values=None,
            classification: bool = False,
        ):
            """
            Predict the output using the trained model and plot the predictions with uncertainty bounds.

            Parameters
            ----------
            features : np.ndarray
                Input data matrix of size (N, D), where N is the number of samples and D is the number of features.
            input_dimension : int
                The dimensionality of the input space.
            true_values : np.ndarray, optional
                True values for comparison (size N,). If provided, Standardized MSE will be returned.

            Returns
            -------
            np.ndarray, np.ndarray
                Predicted output vector (N,) and variance of predictions (N,).
            """
            if not hasattr(self, "W_D") or not hasattr(self, "V"):
                raise ValueError(
                    "Model has not been trained yet. Call 'train' before 'predict'."
                )

            I = input_dimension
            R = rank
            X = features
            N = X.shape[0]
            D = X.shape[1]
            Ms = [I] * D  

            # Feature map
            Phi = pure_power_features_TT(X,Ms=Ms) 
            # Phi = legendre_basis(X, I=I)
            predictions = compute_G_left(Phi, self.W_D, N)

        #  Uncertainty Quantification 
            GR_env = [None] * (D + 1)
            GR_env[D] = np.ones((N, 1))
            for k in range(D - 1, -1, -1):
                GR_env[k] = compute_G_right([Phi[k]], [self.W_D[k]], N, G0=GR_env[k + 1])

            GL_env = np.ones((N, 1))
            sum_matrix = np.zeros((N, N))

            for d in range(D):
                G_left  = GL_env          # (N, Rs[d])
                G_right = GR_env[d + 1]  # (N, Rs[d+1])

                x_d = khatri_rao_rows(khatri_rao_rows(G_right, Phi[d]), G_left)  # (N, Rs[d]*Ms[d]*Rs[d+1])

                # x_d @ V[d] @ x_d.T contributes the posterior variance for core d
                sum_matrix += x_d @ self.V[d] @ x_d.T  # (N, N)

                GL_env = compute_G_left([Phi[d]], [self.W_D[d]], N, G0=GL_env)

            # Student-t predictive variance 
            S = (2 * self.a / (2 * self.a - 2)) * ((self.b / self.a) + (sum_matrix / (D * I * R^2)))
            std_dev = np.sqrt(np.diag(S))

            if true_values is not None:
                if classification:
                    accuracy = accuracy_score(true_values, np.sign(predictions))
                    mse = 1 - accuracy
                    print(f"Missclassification rate = {mse}")
                else:
                    # Standardized MSE
                    mse = np.mean((true_values - predictions) ** 2)
                    print(f"MSE = {mse}")
            else:
                mse = None

            return predictions, std_dev, mse

