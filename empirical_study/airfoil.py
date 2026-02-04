"""
This script loads and preprocesses a dataset, trains a machine learning model, and evaluates its performance.
Dependencies and configurations are centralized in `config.py`.
"""

import os, sys

sys.path.append(os.getcwd())
from config import *  # Import everything from config.py
from functions.BTN_KM import btnkm


df = pd.read_csv("data/airfoil.csv", header=None)
df.columns = df.iloc[0]
df = df[1:]
df.reset_index(drop=True, inplace=True)
df = df.values
df = df.astype(float)

X = df[:, :5]
y = df[:, 5]

# Initialize hyperparameters
input_dimension = 20
max_rank = 25

a, b = 1e-3, 1e-3
c, d = 1e-5 * np.ones(max_rank), 1e-6 * np.ones(max_rank)
g, h = 1e-6 * np.ones(input_dimension), 1e-6 * np.ones(input_dimension)


R_effective = []
rmse_values = []
nll_values = []

# Loop 10 times to run the training and evaluation
start_time = time.time()
for i in range(10):
    # Split the data
    np.random.seed(i)
    indices = np.random.permutation(len(X))
    split_index = int(0.90 * len(X))  # 90% for training, 10% for testing
    X_train, X_test = X[indices[:split_index]], X[indices[split_index:]]
    y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]

    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std == 0] = 1
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std  # Use train stats

    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train = (y_train - y_mean) / y_std

    # train the model
    model = btnkm(X_train.shape[1])
    R, _, _, _, _, _, _ = model.train(
        features=X_train,
        target=y_train,
        input_dimension=input_dimension,
        max_rank=max_rank,
        shape_parameter_tau=a,
        scale_parameter_tau=b,
        shape_parameter_lambda=c,
        scale_parameter_lambda=d,
        shape_parameter_delta=g,
        scale_parameter_delta=h,
        max_iter=50,
        precision_update=True,
        lambda_update=True,
        delta_update=True,
        plot_results=False,
        prune_rank=True,
    )

    # Predict (mse is returned by the predict function)
    prediction_mean, prediction_std, _ = model.predict(
        features=X_test, input_dimension=input_dimension
    )

    prediction_mean_unscaled = prediction_mean * y_std + y_mean
    prediction_std_unscaled = prediction_std * y_std

    # nll
    nll = 0.5 * np.log(2 * np.pi * prediction_std_unscaled**2) + 0.5 * (
        (y_test - prediction_mean_unscaled) ** 2
    ) / (prediction_std_unscaled**2)
    nll_values.append(np.mean(nll))

    # rmse
    rmse = np.sqrt(np.mean((prediction_mean_unscaled - y_test) ** 2))
    rmse_values.append(rmse)

    R_effective.append(R)


end_time = time.time()

total_runtime_seconds = end_time - start_time
effective_r = np.mean(R_effective)
effective_r_std = np.std(R_effective)

print(f"Total runtime for 10 runs: {total_runtime_seconds:.2f} seconds")
print(
    f"Mean RMSE: {np.mean(rmse_values)}, Standard Deviation of RMSE: {np.std(rmse_values)}"
)
print(
    f"Mean NLL : {np.mean(nll_values)}, Standard Deviation of NLL: {np.std(nll_values)}"
)
print(f"Effective R: {effective_r}, std: {effective_r_std}")

# REPORTED
# input_dimension = 20
# # max_rank = 50
# Total runtime for 10 runs: 45.65 seconds
# Mean RMSE: 1.7376706531939032, Standard Deviation of RMSE: 0.14943724700909902
# Mean NLL : 1.9771612560155287, Standard Deviation of NLL: 0.09675963679007242
# Effective R: 9.5, std: 0.5
############################################################
