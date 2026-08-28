# Paper: Infomorphic Hopfield Networks

This folder contains the code for the infomorphic Hopfield network paper: associative-memory networks built from `im_net.IM_Layer` neurons, whose recurrent update rule is trained by optimizing local Partial Information Decomposition (PID) goals over each neuron's external (pattern) and internal (lateral/recurrent) inputs, instead of a Hebbian or gradient-descent rule.

See the top-level [README](../README.md) for environment setup; all commands below are run from the repository root.

## Demo

`demo.ipynb` provides a minimal running model and expected results, and should run in under 5 minutes. As a smoke test from the console:

```bash
python paper_hopfield/src/training.py storage=full
```

## Entry points

The code is in `src/`. There are three entry points, depending on which learning rule you want to train with:

- **`training.py`** — the main entry point, training infomorphic Hopfield networks (`im_net.IM_Layer`-based):
  ```bash
  python paper_hopfield/src/training.py goal=goal_name params.alpha=x ...
  ```
  where `goal_name` is one of the configs in `conf/goal/` (e.g. `redundancy`, `synergy_and_x`, `coinfo`, `entropy`, `joint_MI`, `optimized`, or the `candidates/*` and `sweeps/*` goals) and `x` is the desired memory load (patterns = `alpha * neurons`).

- **`hebbian.py`** — trains a classical Hebbian-rule Hopfield network for comparison:
  ```bash
  python paper_hopfield/src/hebbian.py
  ```

- **`eval_rule.py`** — trains with a learning rule from the external comparison projects:
  ```bash
  python paper_hopfield/src/eval_rule.py learning_rule=rule_name eval_model=basic
  ```
  where `rule_name` is one of `conf/learning_rule/` (`descent_l2`, `descent_exp_si`, `gardner`, `gardner_km`, `hebb`, `mpf`, `trivial`).

## External projects

This code compares against two other learning rules, whose implementations are vendored/adapted rather than original to this project:
- **Descent L2** ([Tolmachev & Manton, 2020](https://arxiv.org/abs/2010.01472)): the rule implementations in `src/tolmachev.py` are adapted from [ptolmachev/Hopfield_Nets](https://github.com/ptolmachev/Hopfield_Nets). The stability plots were produced with [our fork](https://github.com/markbluemel02/Hopfield_Nets/tree/Infomorphic) (`Infomorphic` branch). The upstream repository specifies no license; its code is reused here for academic comparison with attribution to the paper above.
- **MPF** ([Hillar et al., 2012](https://arxiv.org/abs/1204.2916)): implemented in `src/mpf.py`.

`im_net` can additionally compute PID atoms with two third-party estimators, vendored as git submodules under `external/` (see the top-level README for how to fetch them) rather than required by default:
- [`dit`](https://github.com/dit/dit) (BSD 3-Clause)
- [`BROJA_2PID`](https://github.com/Abzinger/BROJA_2PID) (Apache License 2.0)

## Modifying parameters

Parameters are configured via Hydra; config files live in `conf/` (`model/`, `goal/`, `binning/`, `dataset/`, `storage/`, `learning_rule/`, `eval_model/`, `mods/`). Any value can be overridden on the command line:

```bash
python paper_hopfield/src/training.py goal=goal_name params.epochs=500
```

### Common overrides

Sweep over memory loads (training and testing performance at each), with the range set by `preset`:
```bash
python paper_hopfield/src/training.py +mods/capacity=preset ...
```
If `capacity.interrupt` is true, the sweep stops once storage fails at a given memory load. This is also the pattern used to optimize a goal's gamma weights across memory loads.

Run a full Optuna optimization over the goal weights:
```bash
python paper_hopfield/src/training.py --multirun +mods/capacity=optimization goal=sweep storage=cluster hydra=optimizer
```
Check the `hydra` config first to set the parameters and name of the optimization study.

## Storage

Results (performance, PID atom sizes, and the parameters used) are saved to the `experiments` folder, in a subfolder `experiment_name/year_month_day/hour_minute_second[_millisecond]`.

## Results and figures

The notebooks in `plotting/` (`Figure_1.ipynb`–`Figure_5.ipynb`, main figures; `plotting/appendix/`, supplementary figures) reproduce the paper's figures and list which runs are needed for each.
