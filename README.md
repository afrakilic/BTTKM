# Bayesian Tensor Train Kernel Machines with Automatic Model Complexity Inference

[![Source Code License](https://img.shields.io/badge/license-GPL-blueviolet)](https://github.com/afrakilic/BTTKM/blob/main/LICENSE)

Code for the paper [Bayesian Tensor Train Kernel Machines with Automatic Model Complexity Inference](https://arxiv.org/abs/2611.07129) (Afra Kilic and Kim Batselier). A shorter version of this paper has been submitted to ESANN 2027.

BTT-KM is a Bayesian tensor network kernel machine in which the model weights form a tensor
train (TT). Hierarchical ARD-type priors on the TT cores infer the TT ranks and the feature
dimensions from the data, and mean-field variational inference gives a Bayesian alternating
least squares algorithm with the same cost as its deterministic counterpart, together with a
closed-form predictive distribution.

## Installation

Python 3.10 was used.

```bash
git clone https://github.com/afrakilic/BTTKM.git
cd BTTKM
pip install -r requirements.txt
```

## Repository structure

```
bttkm/                  the model
  model.py                BTTKM class: train, predict
  tt.py                   feature map, core unfoldings and TT contractions
experiments/            one script per experiment of the paper
  synthetic.py            Section 4.1: synthetic experiment (Figures 3 and 4)
  regression.py           Section 4.2: UCI regression (Tables 1 and 2)
  regression_summary.py   prints Tables 1 and 2 (regression) from the saved results
  classification.py       Section 4.2: UCI classification (Table 1)
  utils.py                data loading, train/test split and metrics
data/                   UCI data sets
results/                saved results
```

## Usage

```python
from bttkm import BTTKM

model = BTTKM().train(X_train, y_train, M=20, R=5,
                      a0=1e-3, b0=1e-3, c0=1e-6, d0=1e-6, g0=1e-6, h0=1e-6, max_iter=50)
mean, std = model.predict(X_test)        # Student's-t predictive, nu = 2 * model.a
print(model.effective_ranks)             # inferred TT ranks R_2 ... R_D
```

`y_train` is a column vector of shape (N, 1). Set `prune_rank=True` to remove rank components
that explain less than `rank_tol` % of the variance during training.

## Reproducing the paper

| Paper                              | Script                                                                 |
|------------------------------------|------------------------------------------------------------------------|
| Figures 3 and 4 (synthetic)        | `python experiments/synthetic.py`                                      |
| Tables 1 and 2 (regression)        | `python experiments/regression.py`, then `python experiments/regression_summary.py` |
| Table 1 (classification)           | `python experiments/classification.py spambase` (or `adult`)           |

`regression.py` runs Airfoil, Energy, Yacht and Concrete (or the data sets given as arguments)
and saves the per-split results to `results/per_run_results.csv`; add `--no-mc` to skip the
Monte Carlo check. The BTN-KM and GP results in Tables 1 and 2 are taken from
[BTN-Kernel-Machines](https://github.com/afrakilic/BTN-Kernel-Machines).

All experiments use fixed random seeds. They were run on a MacBook Pro (Apple M2 Pro, 16 GB,
macOS 15.5); other operating systems, hardware or library versions may give slightly
different numbers. Spambase takes about 45 minutes and Adult about 7 hours.

## Results

UCI benchmarks, 10 random 90/10 splits, mean ± standard deviation over the splits. Times are per
split and cover training and prediction; the Monte Carlo check is not included.

| Data set | RMSE | Analytic NLL | MC NLL | Analytic cov. (%) | MC cov. (%) | Effective rank | Time per split (s) |
|----------|------|--------------|--------|-------------------|-------------|----------------|--------------------|
| Airfoil  | 1.526 ± 0.161 | 1.810 ± 0.095 | 1.812 ± 0.093 | 95.2 ± 1.5 | 95.4 ± 1.6 | 4.88 ± 0.12 | 6.89 ± 0.20 |
| Concrete | 5.083 ± 0.611 | 3.052 ± 0.132 | 3.050 ± 0.126 | 93.7 ± 2.5 | 93.8 ± 2.2 | 3.90 ± 0.14 | 9.41 ± 0.39 |
| Energy   | 0.480 ± 0.054 | 0.708 ± 0.091 | 0.722 ± 0.081 | 96.1 ± 2.7 | 97.3 ± 1.5 | 4.76 ± 0.11 | 10.60 ± 0.22 |
| Yacht    | 0.587 ± 0.144 | 0.814 ± 0.254 | 0.818 ± 0.256 | 95.8 ± 2.9 | 96.1 ± 3.2 | 4.30 ± 0.18 | 5.28 ± 0.41 |

| Data set | Misclassification | NLL | Effective rank | Time per split (s) |
|----------|-------------------|-----|----------------|--------------------|
| Spambase | 0.094 ± 0.036 | 0.257 ± 0.045 | 2.83 ± 0.20 | 266.55 ± 12.58 |
| Adult    | 0.145 ± 0.005 | 0.329 ± 0.008 | 3.00 ± 0.00 | 2446.24 ± 69.06 |

The effective rank is the mean inner TT rank after truncation.

## Citation

```bibtex
@misc{kilic2026bayesiantensortrainkernel,
      title={Bayesian Tensor Train Kernel Machines with Automatic Model Complexity Inference},
      author={Afra Kilic and Kim Batselier},
      year={2026},
      eprint={2611.07129},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2611.07129},
}
```

## License and author

GNU General Public License, see [LICENSE](LICENSE).

[Afra Kilic](https://www.tudelft.nl/staff/h.a.kilic/), PhD candidate, [Delft Center for Systems and Control](https://www.tudelft.nl/en/3me/about/departments/delft-center-for-systems-and-control/), Delft University of Technology (H.A.Kilic@tudelft.nl | hafra.kilic@gmail.com).
This research is part of the project Sustainable learning for Artificial Intelligence from noisy large-scale data (project number VI.Vidi.213.017), financed by the Dutch Research Council (NWO).

Copyright (c) 2025 Afra Kilic. Technische Universiteit Delft hereby disclaims all copyright interest.