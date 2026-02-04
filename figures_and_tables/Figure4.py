"""
This script produces Figure 4, illustrating the convergence behavior of the BTN-Kernel machines
on three regression datasets: Concrete, Energy, and Airfoil.

For each dataset, it:
  - Loads and normalizes the data.
  - Trains BTN-Kernel machines with three different maximum rank settings.
  - Tracks the effective rank (R_eff) and variational lower bound (LB) during training.

The plots show how the model automatically prunes redundant components and converges,
demonstrating its efficiency and robustness across varying complexities.
"""

# FIGURE 4

import os, sys

sys.path.append(os.getcwd())
from config import *  # Import everything from config.py
from functions.BTN_KM import btnkm

datasets = ["concrete.csv", "energy.csv", "airfoil.csv"]
titles = ["Concrete", "Energy", "Airfoil"]

# Set experiment parameters
input_dimension = 20
max_rank = np.array([10, 25, 50])
line_styles = ["-", "--", "-."]
markers = ["o", "s", "d"]
line_colors = ["#A40000", "#00008B", "#006B3C"]


fig, axs = plt.subplots(2, 3, figsize=(15.5, 7))


for dataset_idx, dataset_name in enumerate(datasets):

    df = pd.read_csv(f"data/{dataset_name}", header=None)
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True).astype(float)

    X_all, y_all = df.iloc[:, :-1].values, df.iloc[:, -1].values

    np.random.seed(1)
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

    all_R_values_energy = []
    all_LB_energy = []

    for i in range(3):
        a, b = 1e-3, 1e-3
        c, d = 1e-5 * np.ones(max_rank[i]), 1e-6 * np.ones(max_rank[i])
        g, h = 1e-6 * np.ones(input_dimension), 1e-6 * np.ones(input_dimension)

        model = btnkm(X_train.shape[1])
        _, _, _, _, _, R_values, LB = model.train(
            features=X_train,
            target=y_train,
            input_dimension=input_dimension,
            max_rank=max_rank[i],
            shape_parameter_tau=a,
            scale_parameter_tau=b,
            shape_parameter_lambda=c,
            scale_parameter_lambda=d,
            shape_parameter_delta=g,
            scale_parameter_delta=h,
            max_iter=150,
            precision_update=True,
            lambda_update=True,
            delta_update=True,
            plot_results=False,
            prune_rank=True,
            lower_bound_tol=1e-10,
        )

        all_R_values_energy.append(R_values)
        all_LB_energy.append(LB)

    # Plot R values
    for i in range(3):
        axs[0, dataset_idx].plot(
            all_R_values_energy[i],
            linestyle=line_styles[i],
            marker=markers[i],
            color=line_colors[i],
            markersize=3,
            label=r"$R_{{\max}} = {}$".format(max_rank[i]),
        )
    axs[0, dataset_idx].set_title(f"{titles[dataset_idx]}", fontsize=24)
    axs[0, 0].set_ylabel(r"$R_{\text{eff}}$", fontsize=22)
    axs[0, dataset_idx].spines["top"].set_visible(False)
    axs[0, dataset_idx].spines["right"].set_visible(False)
    axs[0, dataset_idx].set_facecolor("white")
    axs[0, dataset_idx].tick_params(axis="both", which="major", labelsize=20)
    max_len_r = max(len(r) for r in all_R_values_energy)
    axs[0, dataset_idx].set_xticks(np.arange(0, max_len_r + 1, 50))

    # Plot Lower Bounds
    for i in range(3):
        iterations = np.arange(1, len(all_LB_energy[i]) + 1)
        axs[1, dataset_idx].plot(
            iterations,
            all_LB_energy[i],
            linestyle=line_styles[i],
            marker=markers[i],
            color=line_colors[i],
            markersize=3,
            alpha=0.8,
        )
    axs[1, dataset_idx].set_xlabel("Iteration", fontsize=22)
    axs[1, 0].set_ylabel("LB", fontsize=22)
    axs[1, dataset_idx].spines["top"].set_visible(False)
    axs[1, dataset_idx].spines["right"].set_visible(False)
    axs[1, dataset_idx].set_facecolor("white")
    axs[1, dataset_idx].tick_params(axis="both", which="major", labelsize=20)
    axs[1, dataset_idx].set_xticks(np.arange(0, max_len_r + 1, 50))

handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="center left",
    bbox_to_anchor=(0.93, 0.5),
    fontsize=22,
    frameon=False,
)
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.subplots_adjust(wspace=0.5, hspace=0.3)
#plt.savefig("plot2.pdf", format='pdf', bbox_inches='tight')
plt.show()
