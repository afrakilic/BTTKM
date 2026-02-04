"""
This script generates results for Table 1 and Table 2, analyzing the effect of initial maximum rank
(R_max) on the BTN-Kernel machines' predictive performance and effective rank.

Experiments are conducted on three benchmark regression datasets, each evaluated over 10 random splits.

For a fixed input feature dimension (D=20), the model is trained with three different rank settings
(R_max = 10, 25, 50) while adapting hyperparameters for both rank- and input-dimension-specific precisions.

Metrics reported include RMSE, negative log-likelihood (NLL), and the effective rank (R_eff), averaged
across trials to assess model accuracy and complexity control as a function of rank initialization.

This evaluation highlights how varying initial model capacity affects generalization and rank sparsity.
"""

import os, sys, pprint

sys.path.append(os.getcwd())
from config import *  # Import everything from config.py
from functions.BTN_KM import btnkm

datasets = ["concrete.csv", "energy.csv", "airfoil.csv"]
titles = ["Concrete", "Energy", "Airfoil"]

input_dimension = 20
max_rank = np.array([10, 25, 50])
num_trials = 10

# Collect metrics
results = {
    title: {r: {"rmse": [], "nll": [], "r_eff": []} for r in max_rank}
    for title in titles
}

for dataset_idx, dataset_name in enumerate(datasets):
    df = pd.read_csv(f"data/{dataset_name}", header=None)
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True).astype(float)

    X_all, y_all = df.iloc[:, :-1].values, df.iloc[:, -1].values

    for trial in range(num_trials):
        np.random.seed(trial)
        indices = np.random.permutation(len(X_all))
        split_index = int(0.9 * len(X_all))
        X_train, X_test = X_all[indices[:split_index]], X_all[indices[split_index:]]
        y_train, y_test = y_all[indices[:split_index]], y_all[indices[split_index:]]

        # Normalize
        X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
        X_std[X_std == 0] = 1
        X_train = (X_train - X_mean) / X_std
        X_test = (X_test - X_mean) / X_std

        y_mean, y_std = y_train.mean(), y_train.std()
        y_train = (y_train - y_mean) / y_std

        for i, r in enumerate(max_rank):
            a, b = 1e-3, 1e-3
            c, d = 1e-5 * np.ones(r), 1e-6 * np.ones(r)
            g, h = 1e-6 * np.ones(input_dimension), 1e-6 * np.ones(input_dimension)

            model = btnkm(X_train.shape[1])
            _, _, _, _, _, R_values, _ = model.train(
                features=X_train,
                target=y_train,
                input_dimension=input_dimension,
                max_rank=r,
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

            pred_mean, pred_std, _ = model.predict(
                features=X_test, input_dimension=input_dimension
            )
            pred_mean = pred_mean * y_std + y_mean
            pred_std = pred_std * y_std

            rmse = np.sqrt(np.mean((pred_mean - y_test) ** 2))
            nll = np.mean(
                0.5 * np.log(2 * np.pi * pred_std**2)
                + 0.5 * ((y_test - pred_mean) ** 2) / (pred_std**2)
            )
            r_eff = R_values[-1]

            title = titles[dataset_idx]
            results[title][r]["rmse"].append(rmse)
            results[title][r]["nll"].append(nll)
            results[title][r]["r_eff"].append(r_eff)

# Print average ± std
summary = {}
for title in titles:
    summary[title] = {}
    for r in max_rank:
        rmse_vals = results[title][r]["rmse"]
        nll_vals = results[title][r]["nll"]
        r_vals = results[title][r]["r_eff"]

        summary[title][r] = {
            "RMSE": f"{np.mean(rmse_vals):.3f} ± {np.std(rmse_vals):.3f}",
            "NLL": f"{np.mean(nll_vals):.3f} ± {np.std(nll_vals):.3f}",
            "R_eff": f"{np.mean(r_vals):.1f} ± {np.std(r_vals):.1f}",
        }

pprint.pprint(summary)
