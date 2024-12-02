# Infomorphic Networks

## Setup environment
All packages needed to run the code are listed in `env.yaml`. To create a conda environment with all the necessary packages and dependencies, and to activate the environment, run the following commands:
```bash
conda env create -f env.yaml
conda activate infomorphic_env
```

## Running the code
To run a specific infomorpic network model, you can use the following command:

```bash
python paper_trivariate/base_model.py +models=model_name +goals/model_name=goal_name
```

Where `model_name` is the name of the model you want to run and `goal_name` is the name of the goal you want to run. The available models are:
- `infomorphic_context`
- `infomorphic_lateral`
- `infomorphic_context_lateral`
- `infomorphic_context_lateral_feedback`
- `infomorphic_random_projection`*

The available goal functions for each of the models can be found in the `goals` folder.
The models marked with * don't have a specific goal function.

To run a single layer model, you can use the following command:

```bash
python paper_trivariate/base_model_singlelayer.py +models=infomorphic_readout
```

All results, like performance and atom sizes of the neurons, as well as the parameters used, are saved to the `experiments` folder. The results are saved in a subfolder with the structure `dataset_name/model_name/goal_name/singleruns/day_month_year/hour_minute_second`.

To perform multiple runs at once, you can list the parameters that are different in the runs in `/conf/hydra_base_config.yaml`. The parameters that are different in the runs are listed in the `multirun` section of the configuration file. To perform multiple runs, you can use the following command:

```bash
python paper_trivariate/base_model.py --multirun +models=model_name +goals/model_name=goal_name
```
The results are saved in a subfolder with the structure `dataset_name/model_name/goal_name/multiruns/day_month_year/hour_minute_second/job_idx`.


To perform an optimization, you can use the following command:

```bash
python paper_trivariate/base_model.py --multirun +models=model_name +goals/model_name=optuna_sweep.yaml hydra=hydra_optuna_config.yaml
```
Before performing the optimization, check the `hydra_optuna_config` file to set the parameters of the optimization. The optimization runs are saved in a subfolder with the structure 
`dataset_name/model_name/goal_name/optuna_runs/day_month_year/hour_minute_second` and the optuna study is saved in a database in `dataset_name/model_name/goal_name/optuna_runs/optuna_results.sqlite3`. The study name can be changed in the `hydra_optuna_config` file.

## Modifying the Model Parameters
To modify the parameters of the model, you can use the `hydra` configuration file. The configuration files are located in the `conf` folder. The configuration files are written in YAML format. To modify the parameters of the model, you can change the values in the configuration file. Additionally, one can add overrides to the command line to change the parameters of the model. For example, to change the number of epochs of the model, you can use the following command:

```bash
python paper_trivariate/base_model.py +models=model_name +goals/model_name=goal_name exp_params.epochs=500
```

## Results of the Paper
To reproduce the figures of the paper, look at the juptyer notebooks in the `notebooks` folder. The notebooks list the runs that are necessary to reproduce the figure as well as the code to generate the figure. The figures of the paper are saved in the `figures` folder.