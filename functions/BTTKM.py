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

        for it in pbar:

            GR_env = [None] * (D + 1)
            HR_env = [None] * (D + 1)
            GR_env[D] = np.ones((N, 1))
            HR_env[D] = np.ones((N, 1))
            
            for k in range(D - 1, -1, -1):
                GR_env[k] = compute_G_right([Phi[k]], [W_D[k]], N, G0=GR_env[k + 1])
                HR_env[k] = compute_H_right([Phi[k]], [W_D[k]], [Sigma[k]], N, H0=HR_env[k + 1])
            GR_env[0] = np.ones((N, 1))
            HR_env[0] = np.ones((N, 1))
            GL_env = np.ones((N, 1))
            HL_env = np.ones((N, 1))

            for d in range(D):
                G_left = GL_env
                G_right = GR_env[d + 1]
                Gd = khatri_rao_rows(khatri_rao_rows(G_right, Phi[d]), G_left)

                H_left = HL_env
                H_right = HR_env[d + 1]
                ggT = reshape_permute_ggT_to_sigma(khatri_rao_rows(H_left, Phi[d]).T @ khatri_rao_rows(H_right, Phi[d]),
                                                R_d=W_D[d].shape[0],
                                                M_d=W_D[d].shape[1],
                                                R_next=W_D[d].shape[2])
                # Gd@Gd.T
                Sigma[d] = np.linalg.pinv(tau * ggT + np.kron(np.kron(lambdas[d+1]* np.eye(Rs[d+1]), deltas[d]*np.eye(Ms[d])), lambdas[d]*np.eye(Rs[d])))
                # Sigma[d] = np.linalg.pinv(ggT)
                W_D[d] = (tau * Sigma[d] @ Gd.T @ Y).reshape(W_D[d].shape, order="F")

                GL_env = compute_G_left([Phi[d]], [W_D[d]], N, G0=GL_env)
                HL_env = compute_H_left([Phi[d]], [W_D[d]], [Sigma[d]], N, H0=HL_env)


                 # #lambda updates 
                if it > 5 and lambda_update: 
                    if d > 0: 
                        Klam1= np.kron(lambdas[d+1]*np.eye(lambdas[d+1].size),deltas[d]*np.eye(deltas[d].size))
                        Klam2= np.kron(deltas[d-1]*np.eye(deltas[d-1].size), lambdas[d-1]*np.eye(lambdas[d-1].size))
                        Sigma_cored = np.diag(Sigma[d]).reshape(W_D[d].shape, order="F")
                        Sigma1= tt_core_unfolding(Sigma_cored, mode=1)[0]
                        W1 =  tt_core_unfolding(W_D[d], mode=1)[0]
                        ewdel1 = np.diag(W1 @ Klam1 @ W1.T) + Sigma1 @ np.kron(lambdas[d+1],deltas[d])
                        Sigma_coreb = np.diag(Sigma[d-1]).reshape(W_D[d-1].shape, order="F")
                        Sigma3 = tt_core_unfolding(Sigma_coreb, mode=3)[0]
                        W3 =  tt_core_unfolding(W_D[d-1], mode=3)[0]
                        if d == 0:
                            ewdel3 = 0
                            cNs[d] = c0s[d] + 0.5 * (lambdas[d+1].size * deltas[d].size) 
                        else:    
                            ewdel3 = np.diag(W3 @ Klam2 @ W3.T) + Sigma3 @ np.kron(deltas[d-1],lambdas[d-1])
                            cNs[d] = c0s[d] + 0.5 * (deltas[d-1].size * lambdas[d-1].size + lambdas[d+1].size * deltas[d].size) 

                        dNs[d] = d0s[d] + 0.5 * (ewdel1 + ewdel3)
        
                        lambdas[d] = cNs[d] /dNs[d]

    
            
                #delta updates 
                if it>5 and delta_update:
                    Kdel = np.kron(lambdas[d+1] * np.eye(lambdas[d+1].size), lambdas[d] * np.eye(lambdas[d].size))

                    Sigma_cored = np.diag(Sigma[d]).reshape(W_D[d].shape, order="F")
                    Sigma2 = tt_core_unfolding(Sigma_cored, mode=2)[0]
                    W2 = tt_core_unfolding(W_D[d], mode=2)[0]
                    ewdel2 = np.diag(W2 @ Kdel @ W2.T) + Sigma2 @ np.kron(lambdas[d+1], lambdas[d])
                    gNs[d] = g0s[d] + 0.5 * (lambdas[d].size * lambdas[d+1].size)
                    hNs[d] = h0s[d] + 0.5 * ewdel2
                    deltas[d] = gNs[d] / hNs[d]
            
            y_hat = GL_env
            err = np.dot((Y - y_hat).T, (Y - y_hat))

            def slab_norms(W_d):
                R_d, M_d, R_next = W_d.shape
                return np.array([np.linalg.norm(W_d[:, m, :]) for m in range(M_d)])
            
            

            if it > 5 and precision_update:
                a_N = a0 + (N / 2)
                YY = Y.T @ Y
                YG = -2 * np.dot(Y.T, y_hat)
                GG = np.sum(HL_env, axis=0)
                b_N = b0 + (0.5 * (YY + YG + GG))
            else:
                a_N = a0
                b_N = b0

            tau = a_N / b_N

            #rank pruning 
            W_D_mode1 = tt_core_unfolding(W_D, mode=1)
            W_D_mode2 = tt_core_unfolding(W_D, mode=2)
            W_D_mode3 = tt_core_unfolding(W_D, mode=3)

            if it > 5 and it % 10 == 0 and prune_rank:
                for d in range(1, D): 
                    comPower_delta =np.diag( W_D_mode2[d]@W_D_mode2[d].T)
                    var_explained = comPower_delta / np.sum(comPower_delta) * 100
                    dead = var_explained < 1
                    W_D[d][:, dead, :] = 0 
                    
                    stack = np.hstack((W_D_mode3[d-1], W_D_mode1[d]))
                    comPower = np.diag(stack@stack.T)
                    var_explained = comPower / np.sum(comPower) * 100
                    
                    dead = var_explained < 1

                    W_D[d-1][:, :, dead] = 0  # zero out mode-3 slabs of core d-1
                    W_D[d][dead, :, :] = 0    # zero out mode-1 slabs of core d

             # FIT METRIC
            if classification:
                Fit = accuracy_score(Y, np.sign(y_hat))
            else:
                tss = np.sum((Y.reshape(-1,1) - np.mean(Y))**2)
                rss = np.sum((Y.reshape(-1,1) - y_hat)**2)
                Fit = 1 - (rss / tss)  # R-squared

            
            
            pbar.set_postfix(Fit=f"{float(Fit):.4f}")

        for d in range(D):
                norms = slab_norms(W_D[d])
                rho, p = spearmanr(deltas[d], norms)  
                print(f"core {d}: spearman(delta, ||slab||) = {rho:.3f} (p={p:.3f})")
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

        # --- Uncertainty Quantification ---
            # Precompute all right envelopes
            GR_env = [None] * (D + 1)
            GR_env[D] = np.ones((N, 1))
            for k in range(D - 1, -1, -1):
                GR_env[k] = compute_G_right([Phi[k]], [self.W_D[k]], N, G0=GR_env[k + 1])

            GL_env = np.ones((N, 1))
            sum_matrix = np.zeros((N, N))

            for d in range(D):
                G_left  = GL_env          # (N, Rs[d])
                G_right = GR_env[d + 1]  # (N, Rs[d+1])

                # Effective regressor for core d: khatri-rao of (G_right ⊙ Phi[d] ⊙ G_left)
                # shape: (N, Rs[d+1] * Ms[d] * Rs[d])  — same as Gd in training
                x_d = khatri_rao_rows(khatri_rao_rows(G_right, Phi[d]), G_left)  # (N, Rs[d]*Ms[d]*Rs[d+1])

                # x_d @ V[d] @ x_d.T contributes the posterior variance for core d
                sum_matrix += x_d @ self.V[d] @ x_d.T  # (N, N)

                GL_env = compute_G_left([Phi[d]], [self.W_D[d]], N, G0=GL_env)

            # Student-t predictive variance (same scaling as CPD)
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

