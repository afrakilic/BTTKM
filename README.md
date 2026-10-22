# Regularized Bayesian Tensor Train Kernel Machines

[![Source Code License](https://img.shields.io/badge/license-GPL-blueviolet)](https://github.com/afrakilic/BTTKM/blob/main/LICENSE)

Code for the paper *Regularized Bayesian Tensor Train Kernel Machines* (Afra Kilic and Kim Batselier).

BTT-KM is a Bayesian tensor-network kernel machine in which the model weights form a tensor
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
bttkm/          the model
  model.py        BTTKM class: train, predict
  tt.py           feature map, core unfoldings and TT contractions
experiments/
  synthetic.py    Section 4: synthetic experiment (Figures 3 and 4)
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

```bash
python experiments/synthetic.py
```

This prints the variance explained and the posterior precisions of every rank and feature
component, and shows Figures 3 and 4 together with the lower bound and the predictions.
The experiment uses fixed random seeds and was run on a MacBook Pro (Apple M2 Pro, 16 GB,
macOS 15.5); other operating systems, hardware or library versions may give slightly
different numbers.

## License and author

GNU General Public License, see [LICENSE](LICENSE).

[Afra Kilic](https://www.tudelft.nl/staff/h.a.kilic/), PhD candidate, [Delft Center for Systems and Control](https://www.tudelft.nl/en/3me/about/departments/delft-center-for-systems-and-control/), Delft University of Technology (H.A.Kilic@tudelft.nl | hafra.kilic@gmail.com).
This research is part of the project Sustainable learning for Artificial Intelligence from noisy large-scale data (project number VI.Vidi.213.017), financed by the Dutch Research Council (NWO).

Copyright (c) 2025 Afra Kilic. Technische Universiteit Delft hereby disclaims all copyright interest.
