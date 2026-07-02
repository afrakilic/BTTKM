import os, sys

sys.path.append(os.getcwd())
from config import *  # Import everything from config.py
from functions.TT_helpers import (generate_tt_cores, 
                        compute_G_left, 
                        pure_power_features_TT)
from functions.BTTKM import bttkm

D = 2
I_true = 3
R_true= 3

Ms = [I_true] * D  
Rs = [1] + [R_true] * (D - 1) + [1] 
N = 100

rng = np.random.default_rng(42)
X = rng.random((N, D))
W_D_true = 10* generate_tt_cores(D=D, Ms=Ms, Rs=Rs, seed=1)
Phi = pure_power_features_TT(X,Ms=Ms) 
y = compute_G_left(Phi, W_D_true, N) + 1e-3 * rng.standard_normal((N, 1))

#initialization

I_init= 10
R_init =10

a, b = 1e+2, 1e-1
c, d = 1e-1, 1e-1
g, h = 1e-1, 1e-1

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
        plot_results=True,
        prune_rank=False
    )

    # Predict (mse is returned by the predict function)
    prediction_mean, prediction_std, mse = model.predict(
        features=X, input_dimension=I_init, rank = R_init
    )

    prediction_mean_unscaled = prediction_mean
    # rmse
    rmse = np.sqrt(np.mean((prediction_mean_unscaled - y) ** 2))
    print(f"RMSE: {rmse:.4f}")

BLUE   = "#4878CF"
RED    = "#C44E52"
ORANGE = "#DD8452"

x = np.arange(len(y))
pred = prediction_mean.ravel()
std  = prediction_std.ravel()

sort_idx = np.argsort(y.ravel())
y_sorted    = y.ravel()[sort_idx]
pred_sorted = pred[sort_idx]
std_sorted  = std[sort_idx]
x_sorted    = np.arange(len(y_sorted))

fig, ax = plt.subplots(figsize=(10, 4))

ax.fill_between(x_sorted, pred_sorted - 2*std_sorted, pred_sorted + 2*std_sorted,
                color='gray', alpha=0.2, label=r"$\pm 2\sigma$")
ax.fill_between(x_sorted, pred_sorted -   std_sorted, pred_sorted +   std_sorted,
                color='gray', alpha=0.4, label=r"$\pm 1\sigma$")
ax.scatter(x_sorted, y_sorted,    color='black', s=15, alpha=0.9,
           label="True", zorder=3)
ax.scatter(x_sorted, pred_sorted, color='white', s=15, alpha=0.9,
           edgecolors='black', linewidths=0.8, label="Predicted", zorder=4)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel("Observation index (sorted by true value)")
ax.set_ylabel("Value")
ax.set_title(rf"Predictions with uncertainty  —  RMSE $=$ {rmse:.4f}", pad=8)
ax.legend(frameon=False, fontsize=10)

plt.tight_layout()
plt.show()


mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 15,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "legend.fontsize": 12,
})



# ── Figure 1 : Lambda ─────────────────────────────────────────────────────────
fig1, axes = plt.subplots(1, D-1, figsize=(4*(D-1), 3.5))
if D-1 == 1:
    axes = [axes]

all_norms1 = [np.linalg.norm(W_D[d-1], axis=(0,1)) + np.linalg.norm(W_D[d], axis=(1,2)) for d in range(1, D)]
all_lams   = [np.log(lambdas[d]) for d in range(1, D)]
norm1_max  = max(n.max() for n in all_norms1) + 1
lam_max    = max(l.max() for l in all_lams)   + 1
lam_min    = min(l.min() for l in all_lams)   - 1

for d in range(1, D):
    ax  = axes[d-1]
    ax2 = ax.twinx()
    x   = np.arange(len(lambdas[d]))

    bar = ax.bar(x, all_norms1[d-1], color=BLUE, alpha=0.75, width=0.5)
    line, = ax2.plot(x, np.log(lambdas[d]), color=RED, marker='o',
                     markersize=4, linewidth=1.2, linestyle='--')

    ax.set_ylim(0, norm1_max)
    ax2.set_ylim(lam_min, lam_max)
    ax.set_xlim(-0.5, len(x) - 0.5)

    ax.set_title(rf"$\lambda_{d}$", pad=4)
    ax.set_xlabel("Rank index")

    # norm y-axis: only leftmost subplot shows tick labels
    ax.tick_params(axis='y', colors=BLUE)
    if d == 1:
        ax.set_ylabel("Component norm", color=BLUE)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)

    # precision y-axis: only rightmost subplot shows tick labels
    ax2.tick_params(axis='y', colors=RED)
    ax2.spines['right'].set_color(RED)
    ax2.spines['left'].set_color(BLUE)
    ax.spines['right'].set_visible(False)
    if d == D-1:
        ax2.set_ylabel(r"$\log\,\lambda$ precision", color=RED)
    else:
        ax2.set_yticklabels([])
        ax2.tick_params(axis='y', length=0)

# single shared legend below the figure
fig1.legend(
    handles=[bar, line],
    labels=[r'$\|\mathcal{W}\|_3 + \|\mathcal{W}\|_1$', r'$\log\,\lambda$'],
    loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.08),
)
fig1.suptitle(r"$\lambda$ precision vs.\ TT-core rank norms", fontsize=12)
plt.tight_layout()
plt.show()


# ── Figure 2 : Delta ──────────────────────────────────────────────────────────
fig2, axes = plt.subplots(1, D, figsize=(4*D, 3.5))
if D == 1:
    axes = [axes]

all_norms2 = [np.linalg.norm(W_D[d], axis=(0, 2)) for d in range(D)]
all_deltas = [np.log(deltas[d]) for d in range(D)]
norm2_max  = max(n.max() for n in all_norms2) + 1
delta_max  = max(l.max() for l in all_deltas) + 1
delta_min  = min(l.min() for l in all_deltas) - 1

for d in range(D):
    ax  = axes[d]
    ax2 = ax.twinx()
    x   = np.arange(len(deltas[d]))

    bar = ax.bar(x, all_norms2[d], color=ORANGE, alpha=0.75, width=0.5)
    line, = ax2.plot(x, np.log(deltas[d]), color=RED, marker='o',
                     markersize=4, linewidth=1.2, linestyle='--')

    ax.set_ylim(0, norm2_max)
    ax2.set_ylim(delta_min, delta_max)
    ax.set_xlim(-0.5, len(x) - 0.5)

    ax.set_title(rf"$\delta_{d}$", pad=4)
    ax.set_xlabel("Feature index")

    # norm y-axis: only leftmost subplot shows tick labels
    ax.tick_params(axis='y', colors=ORANGE)
    if d == 0:
        ax.set_ylabel("Component norm", color=ORANGE)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)

    # precision y-axis: only rightmost subplot shows tick labels
    ax2.tick_params(axis='y', colors=RED)
    ax2.spines['right'].set_color(RED)
    ax2.spines['left'].set_color(ORANGE)
    ax.spines['right'].set_visible(False)
    if d == D-1:
        ax2.set_ylabel(r"$\log\,\delta$ precision", color=RED)
    else:
        ax2.set_yticklabels([])
        ax2.tick_params(axis='y', length=0)

# single shared legend below the figure
fig2.legend(
    handles=[bar, line],
    labels=[r'$\|\mathcal{W}\|_2$', r'$\log\,\delta$'],
    loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.08),
)
fig2.suptitle(r"$\delta$ precision vs.\ TT-core feature norms", fontsize=12)
plt.tight_layout()
plt.show()