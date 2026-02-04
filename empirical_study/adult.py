"""
This script loads and preprocesses a dataset, trains a machine learning model, and evaluates its performance.
Dependencies and configurations are centralized in `config.py`.
"""

import os, sys

sys.path.append(os.getcwd())
from config import *  # Import everything from config.py
from functions.BTN_KM import btnkm

# Load the dataset
df = pd.read_csv(
    "data/adult.csv",
    header=None,
    low_memory=False,
)
df.columns = df.iloc[0]
df = df[1:]
df.reset_index(drop=True, inplace=True)
df = df.values
df = df.astype(float)

X = df[:, 0:96]  # features
y = df[:, 96]  # target
y = np.where(y > 0, 1, -1)

# hyper-parameters
input_dimension = 40
max_rank = 10

a, b = 1e-2, 1e-3
c, d = 1e-6 * np.ones(max_rank), 1e-6 * np.ones(max_rank)
g, h = 1e-6 * np.ones(input_dimension), 1e-6 * np.ones(input_dimension)


mse_values = []
R_effective = []
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
        classification=True,
    )

    # Predict (mse is returned by the predict function)
    prediction_mean, prediction_std, mse = model.predict(
        features=X_test,
        input_dimension=input_dimension,
        true_values=y_test,
        classification=True,
    )

    mse_values.append(mse)
    R_effective.append(R)

    # NLL
    probs_gt_zero = norm.sf(0, loc=prediction_mean, scale=prediction_std)  # P(y > 0)
    y_test_binary = (y_test + 1) // 2
    eps = 1e-15
    y_pred_prob = np.clip(probs_gt_zero, eps, 1 - eps)
    nll = -np.mean(
        y_test_binary * np.log(y_pred_prob)
        + (1 - y_test_binary) * np.log(1 - y_pred_prob)
    )
    nll_values.append(nll)


end_time = time.time()


total_runtime_seconds = end_time - start_time
print(f"Total runtime for 10 runs: {total_runtime_seconds:.2f} seconds")


mean_mse, std_mse = np.mean(mse_values), np.std(mse_values)
effective_r, effective_r_std = np.mean(R_effective), np.std(R_effective)
mean_nll, std_nll = np.mean(nll_values), np.std(nll_values)


print(
    f"Mean Missclassication: {mean_mse}, Standard Deviation of Missclassication: {std_mse}"
)
print(f"Mean Effective R: {effective_r}, std: {effective_r_std}")
print(f"Mean NLL: {mean_nll}, std_nll: {std_nll} ")


# Total runtime for 10 runs: 64300.72 seconds
# Mean Missclassication: 0.1433561795268627, Standard Deviation of Missclassication: 0.004530178545165831
# Mean Effective R: 6.3, std: 0.45825756949558405
# Mean NLL: 0.3359441836595124, std_nll: 0.010410498181907212 