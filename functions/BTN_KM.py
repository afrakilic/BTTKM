"""
This script implements a general non-linear model using kernel machines. The model is formulated as:

    y = Φw + e

Where:
- y is the output vector,
- Φ is the input matrix containing features of the data,
- w is the model's weight vector, and
- e is the error term.

The class `Modelx` encapsulates the model, providing methods for both training (`train`) and prediction (`predict`).

The training procedure involves:
- Initializing factor matrices for the model's latent variables,
- Iteratively updating the factor matrices, noise precision (tau), and regularization parameters (lambda),
- Optionally pruning the rank of the model, and
- Computing a lower bound (LB) and fit metric (R-squared) to assess convergence.

The model is trained using input-output data, and once trained, it can make predictions on new data, providing not only the predicted output but also the uncertainty associated with the predictions.

Dependencies and configurations are centralized in `config.py`.
"""

import os, sys

sys.path.append(os.getcwd())
from functions.utils import (
    safelog,
    pure_power_features_full,
    dotkron,
    temp,
    columnwise_kronecker,
    safe_division,
    dotkronX,
)

from config import *  # Import everything from config.py


class btnkm:
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
        seed: int = 0,
        lambda_update: bool = True,
        prune_rank: bool = True,  # Optional rank pruning
        rank_tol: int = 1e-5,  # this is threshold to keep the certain rank (i.e. columns in factor matrices)in terms of explained variance
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

        tau = shape_parameter_tau / scale_parameter_tau
        a0 = shape_parameter_tau
        b0 = scale_parameter_tau

        c0 = (
            shape_parameter_lambda
            if shape_parameter_lambda is not None
            else np.ones(max_iter)
        )
        c_N = c0
        d0 = (
            scale_parameter_lambda
            if scale_parameter_lambda is not None
            else np.ones(max_iter)
        )
        d_N = d0
        lambda_R = c0 / d0

        g0_m = (
            shape_parameter_delta
            if shape_parameter_delta is not None
            else np.ones(I)
        )
        g_N = [g0_m for _ in range(D)]
        h0_m = (
            scale_parameter_delta
            if scale_parameter_delta is not None
            else np.ones(I)
        )
        h_N = [h0_m for _ in range(D)]
        delta = [[g0_m / h0_m] for _ in range(D)]

        # initialize the factor matrices
        W_D = [np.random.randn(I, R) for _ in range(D)]  #  IXR
        # Initialize the covariance matrices
        WSigma_D = [0.1 * np.kron(np.eye(R), np.eye(I)) for d in range(D)]

        # Feature map
        Phi = pure_power_features_full(X, input_dimension) + 0.2

        LB = np.zeros(max_iter)  # lowerbound
        LBRelChan = 0
        pbar = trange(max_iter, desc="Running", leave=True)
        R_values = []  # To store R at each iteration

        # Compute the Hadamard product of matrices in
        hadamard_product_V = np.ones((N, R**2))  # Start with the first matrix
        hadamard_product_mean = np.ones((N, R))
        for d in range(len(W_D)):
            hadamard_product_V = hadamard_product_V * temp(
                Phi=Phi[d], V=WSigma_D[d], R=R
            )  # Element-wise multiplication
            hadamard_product_mean = hadamard_product_mean * (Phi[d] @ W_D[d])

        # MODEL LEARNING
        # FACTOR MATRICES UPDATE
        for it in pbar:

            for d in range(D):  # update the posterior q(vec(W^d)):

                hadamard_product_V = safe_division(
                    hadamard_product_V, (temp(Phi=Phi[d], V=WSigma_D[d], R=R))
                )
                hadamard_product_mean = safe_division(
                    hadamard_product_mean, ((Phi[d] @ W_D[d]))
                )

                W_K_PROD_V = (
                    dotkron(Phi[d], Phi[d]).T @ hadamard_product_V
                )  # G^{d}G^{d}T
                cc, cy = dotkronX(Phi[d], hadamard_product_mean, Y)
                V_temp = np.reshape(
                    np.transpose(
                        np.reshape(W_K_PROD_V, (I, I, R, R), order="F"),
                        axes=(0, 2, 1, 3),
                    ),
                    (I * R, I * R),
                    order="F",
                )

                A = tau * (cc + V_temp) + np.kron(
                    lambda_R * np.eye(R), delta[d] * np.eye(I)
                )
                try:
                    WSigma_D[d] = np.linalg.inv(A)
                except np.linalg.LinAlgError:
                    WSigma_D[d] = np.linalg.pinv(A)

                W_D[d] = np.reshape((tau * WSigma_D[d] @ cy), (I, R), order="F")

                hadamard_product_V = hadamard_product_V * temp(
                    Phi=Phi[d], V=WSigma_D[d], R=R
                )
                hadamard_product_mean = hadamard_product_mean * (Phi[d] @ W_D[d])

            # LAMBDA UPDATES

            # Lambda_M Update
            if delta_update:
                for d in range(D):
                    mtemp = np.diag(W_D[d] @ (lambda_R * np.eye(R)) @ W_D[d].T)
                    vtemp = np.diag(
                        np.reshape(
                            WSigma_D[d]
                            .reshape(I, R, I, R)
                            .transpose(0, 2, 1, 3)
                            .reshape(I**2, R**2)
                            @ (lambda_R * np.eye(R)).ravel(order="F"),
                            (I, I),
                        )
                    )
                    g_N[d] = g0_m + R / 2
                    h_N[d] = h0_m * np.ones(I) + (mtemp + vtemp) / 2
                    delta[d] = g_N[d] / h_N[d]
                    delta[d][delta[d] < 1e-5] = 1e-5

            # Lambda_R Update
            if lambda_update:

                c_N = (0.5 * D * I) + c0
                d_N = 0

                for d in range(D):
                    np.transpose(
                        np.reshape(WSigma_D[d], (I, R, I, R), order="F"),
                        axes=(0, 2, 1, 3),
                    )
                    mtemp = np.diag(W_D[d].T @ (delta[d] * np.eye(I)) @ W_D[d])
                    vtemp = np.diag(
                        np.reshape(
                            (delta[d] * np.eye(I)).ravel(order="F").T
                            @ WSigma_D[d]
                            .reshape(I, R, I, R)
                            .transpose(0, 2, 1, 3)
                            .reshape(I**2, R**2),
                            (R, R),
                        )
                    )
                    d_N += mtemp + vtemp

                d_N = d0 + (0.5 * d_N)
                lambda_R = c_N / d_N
                lambda_R[lambda_R < 1e-5] = 1e-5

            # Error Precision Update
            ss_error = np.dot(
                (Y - np.sum(hadamard_product_mean, axis=1)),
                (Y - np.sum(hadamard_product_mean, axis=1)),
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

            # FIT METRIC
            if classification:
                Fit = accuracy_score(Y, np.sign(np.sum(hadamard_product_mean, axis=1)))
            else:
                tss = np.sum(abs(Y - np.mean(Y)))
                rss = np.sum(abs(Y - np.sum(hadamard_product_mean, axis=1)))
                Fit = 1 - (rss / tss)  # R-squared

            # LOWER BOUND
            temp1 = -0.5 * (a_N / b_N) * err

            temp22 = np.zeros((I * R, I * R))
            for d in range(D):
                temp22 += np.kron(lambda_R * np.eye(R), delta[d] * np.eye(I)) @ (
                    np.outer(W_D[d].ravel(order="F"), W_D[d].ravel(order="F"))
                    + WSigma_D[d]
                )
            temp2 = -0.5 * np.trace(temp22)

            temp3 = 0.5 * np.sum([safelog(np.linalg.slogdet(W)) for W in WSigma_D])

            temp4 = np.sum(
                np.multiply(c_N, (1 - safelog(d_N) - (d0 / d_N)))
            )  # + np.sum(gammaln(c_N))

            temp5 = sum(
                np.sum(
                    gammaln(np.array(g_N[d]))
                    + np.array(g_N[d])
                    * (1 - safelog(np.array(h_N[d])) - (h0_m / np.array(h_N[d])))
                )
                for d in range(len(g_N))
            )
            temp6 = a_N * (1 - safelog(b_N) - (b0 / b_N))  # + gammaln(a_N)

            LB[it] = temp1 + temp2 + temp3 + temp4 + temp5 + temp6

            rankest = R

            # Rank pruning (optional)

            if prune_rank:
                Wall = np.vstack([W for W in W_D])
                comPower = np.diag(Wall.T @ Wall)
                var_explained = comPower / np.sum(comPower) * 100
                rankest = np.sum(var_explained > rank_tol)
                if np.max(rankest) == 0:
                    print("Rank becomes 0 !!!")
                    break
                if R != np.max(rankest):
                    indices = var_explained > rank_tol
                    false_indices = np.where(~indices)[0]
                    lambda_R = np.delete(lambda_R, false_indices)
                    for d in range(D):
                        W_D[d] = np.delete(W_D[d], false_indices, axis=1)
                        ranges = np.r_[
                            [
                                np.arange(
                                    I * false_indices[i], I * false_indices[i] + I
                                )
                                for i in range(len(false_indices))
                            ]
                        ]
                        WSigma_D[d] = np.delete(WSigma_D[d], ranges, axis=1)
                        WSigma_D[d] = np.delete(WSigma_D[d], ranges, axis=0)

                    hadamard_product_mean = np.delete(
                        hadamard_product_mean, false_indices, axis=1
                    )

                    row_indices = {
                        (r - 1) * R + j
                        for r in false_indices + 1
                        for j in range(1, R + 1)
                    }
                    col_indices = {
                        (i - 1) * R + c
                        for c in false_indices + 1
                        for i in range(1, R + 1)
                    }
                    removed = sorted(row_indices | col_indices)
                    removed = np.array(sorted(row_indices | col_indices)) - 1
                    hadamard_product_V = np.delete(
                        hadamard_product_V, removed, axis=1
                    )
                    c0 = c0[:rankest]
                    d0 = d0[:rankest]

                    R = np.max(rankest)

            R_values.append(R)

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
                    "R": rankest,
                    "err": f"{err:.1e}",
                }
            )
            # Convergence checks
            if it > 5:
                if abs(LBRelChan) < lower_bound_tol:
                    print("======= Converged =======")
                    break

        M = np.array(
            [
                np.sum(
                    (np.diag(W_D[d] @ W_D[d].T) / np.sum(np.diag(W_D[d] @ W_D[d].T)))
                    * 100
                    >= 0.25
                    # (1 / np.array(delta[d]))/np.sum(1 / np.array(delta[d]))*100 >= 0.5
                )
                for d in range(D)
            ]
        )
        M_mean, M_std = np.mean(M), np.std(M)
        # Store the estimated parameters for predictionen
        self.W_D = W_D
        self.V = WSigma_D
        self.a = a_N
        self.b = b_N

        print("Model training complete. Parameters stored for prediction.")

        # PLOTS
        if plot_results:
            fig, axs = plt.subplots(2, 1, figsize=(5, 6))

            # First subplot: Plot R_values vs. iteration
            axs[0].plot(R_values, marker="o", color="green", label="Effective R")
            axs[0].set_xlabel("Iteration")
            axs[0].set_ylabel("Rank R")
            axs[0].set_title("Evolution of Rank R during Training")
            axs[0].grid(True)
            axs[0].legend()

            # Second subplot: Scatter plot for LB vs. iterations
            axs[1].scatter(
                np.arange(1, it + 1), LB[:it], color="b", alpha=0.6, label="Lower Bound"
            )
            axs[1].set_xlabel("Iteration")
            axs[1].set_ylabel("Lower Bound (LB)")
            axs[1].set_title("Lower Bound during Training")
            axs[1].legend()

            plt.tight_layout()
            plt.show()

        return R, M_mean, M_std, M, W_D, R_values, LB[:it]

    def predict(
        self,
        features: np.ndarray,
        input_dimension: int,
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

        # Feature map
        Phi = pure_power_features_full(features, input_dimension) + 0.2

        # Combine the factor matrices to compute predictions
        W_D_PROD = np.ones(
            (Phi[0].shape[0], self.W_D[0].shape[1])
        )  # Initialize product matrix
        for d in range(len(self.W_D)):
            W_D_PROD = np.multiply(
                W_D_PROD, (Phi[d] @ self.W_D[d])
            )  # Element-wise multiplication

        predictions = np.sum(W_D_PROD, axis=1)  # Mean predictions

        # Uncertainty quantification
        N = features.shape[0]
        R = self.W_D[0].shape[1]
        M = input_dimension
        D = len(self.W_D)
        sum_matrix = 0
        for d in range(len(self.W_D)):
            W_K = [W for idx, W in enumerate(self.W_D) if idx != d]
            Phi_K = [W for idx, W in enumerate(Phi) if idx != d]
            hadamard_product = np.ones((N, R))
            for i in range(len(W_K)):
                hadamard_product *= Phi_K[i] @ W_K[i]
            x = columnwise_kronecker(Phi[d].T, hadamard_product.T)
            xVx = x.T @ self.V[d] @ x
            sum_matrix += xVx

        S = (2 * self.a / (2 * self.a - 2)) * ((self.b / self.a) + (sum_matrix/(D*M*R)))
        std_dev = np.sqrt(np.diag(S))  # Standard deviation for each prediction
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
