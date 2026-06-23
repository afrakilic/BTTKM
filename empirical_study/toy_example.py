import os, sys

sys.path.append(os.getcwd())
from config import *  # Import everything from config.py
from functions.TT_helpers import (generate_tt_cores, 
                        init_sigmas,
                        compute_G_left, 
                        pure_power_features_TT,
                        tt_core_unfolding)
from functions.BTTKM import bttkm

D = 3
I_true = 3
R_true= 3

Ms = [I_true] * D  
Rs = [1] + [R_true] * (D - 1) + [1] 
N = 100

rng = np.random.default_rng(42)
X = rng.random((N, D))
W_D_true = 10* generate_tt_cores(D=D, Ms=Ms, Rs=Rs, seed=1)
Phi = pure_power_features_TT(X,Ms=Ms) 
y = compute_G_left(Phi, W_D_true, N) + 1e-5 * rng.standard_normal((N, 1))

#initialization

I_init=5
R_init =5

a, b = 1e-3, 1e-3
c, d = 1e-6 , 1e-6
g, h = 1e-6, 1e-6

# Loop 10 times to run the training and evaluation
start_time = time.time()
for i in range(1):

    # train the model
    model =bttkm(X.shape[1])
    W_D, lambdas, deltas, tau = model.train(
        features=X,
        target=y,
        input_dimension=I_init,
        max_rank=R_init,
        shape_parameter_tau=a,
        scale_parameter_tau=b,
        shape_parameter_lambda=c,
        scale_parameter_lambda=d,
        shape_parameter_delta=g,
        scale_parameter_delta=h,
        max_iter=100,
        precision_update=True,
        lambda_update=True,
        delta_update=True,
        plot_results=False,
        prune_rank=False,
    )

    # Predict (mse is returned by the predict function)
    prediction_mean, mse = model.predict(
        features=X, input_dimension=I_init, rank = R_init
    )

    prediction_mean_unscaled = prediction_mean #* y_std + y_mean
    # prediction_std_unscaled = prediction_std * y_std

    # rmse
    rmse = np.sqrt(np.mean((prediction_mean_unscaled - y) ** 2))


    # rmse = np.sqrt(np.mean((y_test - prediction_mean) ** 2))
    print(f"RMSE: {rmse:.4f}")


x = np.arange(len(y))

# Scatter plot
plt.figure()
plt.scatter(x, y, label="y (true)")
plt.scatter(x, prediction_mean, label="y_hat (predicted)")

plt.xlabel("Observation #")
plt.ylabel("Value")
plt.title(f"y and y_hat vs observations (RMSE = {rmse:.4f})")
plt.legend()
plt.show()

W_D_mode1 = tt_core_unfolding(W_D, mode=1)
W_D_mode2 = tt_core_unfolding(W_D, mode=2)
W_D_mode3 = tt_core_unfolding(W_D, mode=3)