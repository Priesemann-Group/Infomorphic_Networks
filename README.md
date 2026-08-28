# Infomorphic Networks

This repository contains the PyTorch implementation of `im_net`, a neuron/layer type (`IM_Layer`) that optimizes local objective functions built from **Partial Information Decomposition (PID)** instead of end-to-end gradient descent. It hosts the code for two papers, each in its own subfolder with its own README:

- [`paper_trivariate/`](paper_trivariate/README.md) — [*"What Should a Neuron Aim For? Designing Local Objective Functions Based on Information Theory"*](https://arxiv.org/abs/2412.02482), feedforward classifiers trained with trivariate PID goals.
- [`paper_hopfield/`](paper_hopfield/README.md) — [*Redundancy Maximization as a Principle of Associative Memory Learning in Hopfield Networks*](https://arxiv.org/pdf/2511.02584) Infomorphic Hopfield Networks, associative-memory networks whose recurrent update rule is trained with local PID goals.

## Contributors
This repository was developed by [Valentin Neuhaus](https://github.com/vneuhaus), [Andreas C. Schneider](https://github.com/ac-schneider), [David A. Ehrlich](https://github.com/daehrlich) and [Mark Blümel](https://github.com/Mark-Bluemel), and is actively maintained by the authors.

## System Requirements
### Operating system
Developed and tested mostly on Linux. No Windows nor macOS-specific testing has been performed.

### Software dependencies
All Python dependencies are listed in `env.yml` and installed via `conda`. Key dependencies include PyTorch, Hydra (1.2), Optuna (2.10), and scikit-learn — see `env.yml` for the complete list.

### Hardware requirements
No non-standard hardware is required. An NVIDIA GPU is optional and only used to accelerate training; training also runs on CPU via `params.pref_gpu=False` (or `exp_params.pref_gpu=False` in `paper_trivariate`).

## Installation guide
### Clone the repository
```bash
git clone <repository-url>
cd infomorph_networks
```

### Optional: alternative PID estimators
`im_net` can optionally compute PID atoms with two additional third-party estimators (`dit`, `BROJA_2PID`) for comparison, vendored as git submodules under `external/`. They are not required for training or reproducing the papers' main results. To add them:
```bash
git submodule update --init --recursive
```

### Create the environment
```bash
conda env create -f env.yml
conda activate infomorphic_env
```

### Typical install time
Around 10 minutes on a standard desktop (excluding submodule/package download time, which depends on network speed).

## Running the code
All commands must be run from the repository root, since `im_net` loads `moebius.pkl` (provided at the repository root) using a path relative to the current working directory.

Both papers configure and override runs the same way, via [Hydra](https://hydra.cc/docs/advanced/override_grammar/basic/): parameters live in each paper's `conf/` folder as YAML, and any value can be overridden on the command line, e.g. to change the number of training epochs. See each paper's own README for its entry points, available models, and example commands:

- [`paper_trivariate/README.md`](paper_trivariate/README.md)
- [`paper_hopfield/README.md`](paper_hopfield/README.md)

All results (performance, PID atom sizes, and the parameters used) are saved to an `experiments` folder, in a subfolder named after the run's timestamp.

## License
This project is licensed under the BSD 3-Clause License — see [`LICENSE`](LICENSE) for details.

## Contacts
The corresponding author for the code is `valentin.neuhaus@ds.mpg.de`.
For questions, also reach out to us at `mark.bluemel@ds.mpg.de`, `andreas.schneider@ds.mpg.de`.