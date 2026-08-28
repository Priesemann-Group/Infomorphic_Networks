# Paper: What Should a Neuron Aim For?

This folder contains the code used for [*"What Should a Neuron Aim For? Designing Local Objective Functions Based on Information Theory"*](https://arxiv.org/abs/2412.02482), which trains classifiers on MNIST/CIFAR-10 out of `im_net.IM_Layer` neurons whose local objective is a weighted combination of **trivariate** Partial Information Decomposition (PID) atoms (context, label, and — depending on the model — a lateral or feedback source).

See the top-level [README](../README.md) for environment setup; all commands below are run from the repository root.

## Entry points and available models

There are three entry points, one per network depth:

| Entry point | Trains | Available `+models=` |
|---|---|---|
| `base_model_singlelayer.py` | a single readout layer | `infomorphic_readout` |
| `base_model.py` | a hidden layer + readout layer | `global_backprop`, `infomorphic_context`, `infomorphic_context_lateral`, `infomorphic_context_lateral_with_feedback`, `infomorphic_lateral`, `infomorphic_random_projection` |
| `base_model_multilayer.py` | two hidden layers + readout layer | `infomorphic_context_lateral_multilayer` |

The model configs live in `conf/models/`; each selects the layer sizes, activation function and (for most models) which sources feed each neuron's PID.

## Running a model

```bash
python paper_trivariate/base_model_singlelayer.py +models=infomorphic_readout
```

Most models under `base_model.py` and `base_model_multilayer.py` also need a goal, which fixes the gamma weights of the PID atoms in the loss. The available goals per model are the YAML files under `conf/goals/<model_name>/`:

```bash
python paper_trivariate/base_model.py +models=infomorphic_context_lateral +goals/infomorphic_context_lateral=naive
python paper_trivariate/base_model_multilayer.py +models=infomorphic_context_lateral_multilayer +goals/infomorphic_context_lateral=hierarchical
```

`global_backprop`, `infomorphic_context_lateral_with_feedback`, `infomorphic_random_projection` and `infomorphic_readout` don't have a dedicated goal folder — they either fix gamma directly in the model config or don't use a PID-based loss.

To perform multiple runs at once (e.g. across seeds or hyperparameters), list the swept parameters in `conf/hydra/hydra_base_config.yaml` and add `--multirun`:

```bash
python paper_trivariate/base_model.py --multirun +models=infomorphic_context_lateral +goals/infomorphic_context_lateral=naive
```

To run a hyperparameter optimization over the goal weights with Optuna, use the `optuna_sweep` goal together with an Optuna Hydra sweeper:

```bash
python paper_trivariate/base_model.py --multirun +models=infomorphic_context_lateral +goals/infomorphic_context_lateral=optuna_sweep hydra=hydra_optuna_config
```
Check `conf/hydra/hydra_optuna_config.yaml` (or `hydra_optuna_cma-es_config.yaml` for CMA-ES) beforehand to set the study name and search space.

## Modifying parameters

Datasets (`conf/dataset/`: `mnist`, `cifar10`, `rhm`) and storage presets (`conf/storage/`: `base`, `custom`, `debugging`, `full` — controlling which quantities get written to `data.h5`) are also selected via Hydra groups, and any leaf value can be overridden on the command line, e.g.:

```bash
python paper_trivariate/base_model.py +models=infomorphic_context_lateral +goals/infomorphic_context_lateral=naive dataset=cifar10 exp_params.epochs=500
```

## Results and figures

Runs are written to `experiments/<dataset>/<model>/<goal>/<singleruns|multiruns>/<day_month_year>/<hour_minute_second>[_<job_idx>]`, containing `data.h5` (performance, PID atoms, weights — depending on the storage preset), `config.yaml` and `run_properties.yaml`. Load them for analysis with `im_net.datamanager.DataManager(path, mode='analysis')`.

The notebooks in `Notebooks/` (`MainFigures.ipynb`, `goal_sensitivity.ipynb`) reproduce the paper's figures from these runs; exported figures are saved to `Figures/`.
