#!/bin/bash

#SBATCH -p grete:shared
#SBATCH -G A100:2
#SBATCH --job-name=joblib-optuna
#SBATCH --output=output-%x.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00


# Load modules or your own conda environment here
module load cuda/12.1
cd /user/ehrlich5/u11172/infomorph_networks/

export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4

source ~/.bashrc
conda activate infomorph

python paper_trivariate/base_model.py --multirun +models=infomorphic_context_lateral +goals/infomorphic_context_lateral=optuna_sweep.yaml hydra=hydra_optuna_cma-es_config.yaml
