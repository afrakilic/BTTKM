# Interpretable Bayesian Tensor Network Kernel Machines with Automatic Rank and Feature Selection

[![Source Code License](https://img.shields.io/badge/license-GPL-blueviolet)](https://github.com/afrakilic/BTN-Kernel-Machines/blob/main/LICENSE)


This repository contains the source code used to produce the results obtained in [Interpretable Bayesian Tensor Network Kernel Machines with Automatic Rank and Feature Selection](https://arxiv.org/abs/2409.12789) submitted to [Journal of Machine Learning Research](https://www.jmlr.org/). This project sets fixed random seeds to promote reproducibility. All experiments were conducted on the following computer:

- **Device**: MacBook Pro 
- **Chip**: Apple M2 Pro 
- **Memory**: 16 GB LPDDR5
- **Operating System**: macOS 15.5 

However, please note that some computations may still yield slightly different results across operating systems (e.g., macOS vs Windows), hardware architectures, or Python library versions.


In this work we propose a fully probabilistic framework that uses sparsity-inducing hierarchical priors on Tensor Network factors to automatically infer tensor rank and feature dimensions, while also identifying the most relevant features for prediction, thereby enhancing model interpretability.

If you find the paper or this repository helpful in your publications, please consider citing it.

```bibtex
@misc{kilic2025interpretablebayesiantensornetwork,
      title={Interpretable Bayesian Tensor Network Kernel Machines with Automatic Rank and Feature Selection}, 
      author={Afra Kilic and Kim Batselier},
      year={2025},
      eprint={2507.11136},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2507.11136}, 
}
```

---

## Installation



Make sure you have Python **3.10.16** installed.

Install dependencies:

```bash
git clone https://github.com/afrakilic/BTN-Kernel-Machines
cd BTN-Kernel-Machines
```

and then install the required packages by, e.g., running

```bash
pip install -r requirements.txt
```

### Structure

The repository code is structured in the following way

- **`data/`**  
  Contains all the UCI benchmark datasets used in the experiments.

- **`figure_and_tables/`**  
  Includes the code for running experiments that generate the figures and tables shown in **Section 4** of the paper, **except for Table 4**.

- **`empirical_study/`**  
  Contains the experiments described in **Section 4.4**, which are reported in **Table 4**.

- **`functions/`**  
  Includes the implementation of **Bayesian Tensor Network Kernel Machines (BTN-KM)** and helper functions organized in the `utils` module.
## License

The repository is provided under the GNU General Public License. See the [LICENSE](https://github.com/afrakilic/BTN-Kernel-Machines/blob/main/LICENSE) file included with this repository.

---

## Author

[Afra KILIC](https://www.tudelft.nl/staff/h.a.kilic/), PhD Candidate [H.A.Kilic@tudelft.nl | hafra.kilic@gmail.com]

> [Delft Center for Systems and Control](https://www.tudelft.nl/en/3me/about/departments/delft-center-for-systems-and-control/) in [Delft University of Technology](https://www.tudelft.nl/en/)

> This research is part of the project Sustainable learning for Artificial Intelligence from noisy large-scale data (with project number VI.Vidi.213.017) which is financed by the Dutch Research Council (NWO).

Copyright (c) 2025 Afra Kilic.

Copyright notice: Technische Universiteit Delft hereby disclaims all copyright interest. 
