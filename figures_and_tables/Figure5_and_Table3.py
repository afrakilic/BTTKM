"""
This script generates Figure 5 and Table 3, evaluating BTN-Kernel machines  performance on three real-world regression datasets.

Experiments compare predictive accuracy and feature-wise sparsity with and without
the input-dimension-specific precision hyperparameter λ_M enabled.

For each dataset, models are trained across two different maximum feature dimensions (M_max = 20, 50),
repeated over 10 random train/test splits.

The reported metrics include RMSE, negative log-likelihood (NLL), and effective rank R.
Feature-wise average precision values (M) are visualized in grouped bar plots to illustrate
how λ_M influences model sparsity and complexity.

These results highlight the role of λ_M in adaptively controlling feature relevance and improving generalization.
"""

import os, sys

sys.path.append(os.getcwd())
from config import *
from functions.BTN_KM import btnkm

# Datasets
datasets = ["concrete.csv", "energy.csv", "airfoil.csv"]
titles = ["Concrete", "Energy", "Airfoil"]
input_dimensions = [20, 50]
max_rank = 25
colors = ["#A40000", "#00008B"]
n_trials = 10

results = {
    title: {
        "on": defaultdict(lambda: {"metrics": [], "M_values": []}),
        "off": defaultdict(lambda: {"metrics": [], "M_values": []}),
    }
    for title in titles
}

# Run experiments
for dataset_idx, dataset_name in enumerate(datasets):
    df = pd.read_csv(f"data/{dataset_name}", header=None)
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True).astype(float)

    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values

    for trial in range(n_trials):
        np.random.seed(trial)
        indices = np.random.permutation(len(X))
        split_index = int(0.90 * len(X))
        X_train, X_test = X[indices[:split_index]], X[indices[split_index:]]
        y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]

        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0)
        X_std[X_std == 0] = 1
        X_train = (X_train - X_mean) / X_std
        X_test = (X_test - X_mean) / X_std

        y_mean = y_train.mean()
        y_std = y_train.std()
        y_train = (y_train - y_mean) / y_std

        for input_dimension in input_dimensions:
            a, b = 1e-3, 1e-3
            c, d = 1e-5 * np.ones(max_rank), 1e-6 * np.ones(max_rank)
            g, h = 1e-6 * np.ones(input_dimension), 1e-6 * np.ones(input_dimension)

            # λM off
            model_off = btnkm(X_train.shape[1])
            R_off, _, _, M_off, _, _, _ = model_off.train(
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
                delta_update=False,
                plot_results=False,
                prune_rank=True,
            )
            pred_mean_off, pred_std_off, _ = model_off.predict(
                features=X_test, input_dimension=input_dimension
            )
            pred_mean_off = pred_mean_off * y_std + y_mean
            pred_std_off = pred_std_off * y_std
            rmse_off = np.sqrt(np.mean((pred_mean_off - y_test) ** 2))
            nll_off = np.mean(
                0.5 * np.log(2 * np.pi * pred_std_off**2)
                + 0.5 * ((y_test - pred_mean_off) ** 2) / (pred_std_off**2)
            )

            results[titles[dataset_idx]]["off"][input_dimension]["metrics"].append(
                (rmse_off, nll_off, R_off)
            )
            results[titles[dataset_idx]]["off"][input_dimension]["M_values"].append(
                M_off
            )

            # λM on
            model_on = btnkm(X_train.shape[1])
            R_on, _, _, M_on, _, _, _ = model_on.train(
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
            pred_mean_on, pred_std_on, _ = model_on.predict(
                features=X_test, input_dimension=input_dimension
            )
            pred_mean_on = pred_mean_on * y_std + y_mean
            pred_std_on = pred_std_on * y_std
            rmse_on = np.sqrt(np.mean((pred_mean_on - y_test) ** 2))
            nll_on = np.mean(
                0.5 * np.log(2 * np.pi * pred_std_on**2)
                + 0.5 * ((y_test - pred_mean_on) ** 2) / (pred_std_on**2)
            )

            results[titles[dataset_idx]]["on"][input_dimension]["metrics"].append(
                (rmse_on, nll_on, R_on)
            )
            results[titles[dataset_idx]]["on"][input_dimension]["M_values"].append(M_on)

# Report results
print("\nAveraged Results over 10 trials:")
for dataset in titles:
    print(f"\n{dataset}:")
    for mode in ["on", "off"]:
        for D in input_dimensions:
            rmse_vals = [r[0] for r in results[dataset][mode][D]["metrics"]]
            nll_vals = [r[1] for r in results[dataset][mode][D]["metrics"]]
            R_vals = [r[2] for r in results[dataset][mode][D]["metrics"]]
            print(
                f"  M-max = {D:3d}, λM {'on ' if mode == 'on' else 'off'} → "
                f"RMSE = {np.mean(rmse_vals):.4f} ± {np.std(rmse_vals):.4f}, "
                f"NLL = {np.mean(nll_vals):.4f} ± {np.std(nll_vals):.4f}, "
                f"R = {np.mean(R_vals):.4f} ± {np.std(R_vals):.4f}"
            )

# Plot
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
width = 0.35

for dataset_idx, dataset in enumerate(titles):
    ax = axs[dataset_idx]

    M_sample = next(iter(results[dataset]["off"][input_dimensions[0]]["M_values"]))
    n_features = len(M_sample)
    x_indices = np.arange(n_features)

    for i, D in enumerate(input_dimensions):
        color = colors[i]
        # average M values
        M_list_off = results[dataset]["off"][D]["M_values"]
        M_list_on = results[dataset]["on"][D]["M_values"]
        M_avg_off = np.mean(np.vstack(M_list_off), axis=0)
        M_avg_on = np.mean(np.vstack(M_list_on), axis=0)

        offset = i * width

        ax.bar(
            x_indices + offset,
            M_avg_off,
            width=width,
            color=color,
            alpha=0.3,
            label=r"$\delta_d$: off, $M_{{\max}} = {}$".format(D),
        )

        ax.bar(
            x_indices + offset,
            M_avg_on,
            width=width,
            color=color,
            alpha=1,
            label=r"$\delta_d$: on, $M_{{\max}} = {}$".format(D),
        )

    ax.axhline(y=50, color="#00008B", linestyle="--", linewidth=2)
    ax.axhline(y=20, color="#A40000", linestyle="--", linewidth=2)
    ax.set_title(dataset, fontsize=24)
    ax.set_xlabel("Features", fontsize=22, labelpad=15)
    ax.set_xticks(x_indices + width / 2)
    ax.set_xticklabels(
        [str(i) for i in range(1, n_features + 1)], fontsize=22
    )
    ax.tick_params(axis="both")
    ax.tick_params(axis="y", labelsize=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if dataset_idx != 0:
        ax.spines["left"].set_visible(False)

axs[0].set_ylabel(r"$\bar{M}_{\text{eff}}$", fontsize=22)

handles, labels = axs[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(
    handles=by_label.values(),
    labels=by_label.keys(),
    loc="center left",
    bbox_to_anchor=(0.9, 0.5),
    fontsize=22,
    frameon=False,
)

plt.tight_layout(rect=[0, 0, 0.88, 1])
#plt.savefig("plot3.pdf", format='pdf', bbox_inches='tight')
plt.show()
