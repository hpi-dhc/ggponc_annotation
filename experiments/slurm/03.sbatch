#!/bin/bash -eux
#SBATCH --job-name=hydra
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=florian.borchert@hpi.de
#SBATCH --partition=longrun # -p
#SBATCH --output=logs/01_hydra_%j.log # %j is job id

eval "$(conda shell.bash hook)"

conda activate ggponc

cd ..

python run_experiment.py -cn=03_ggponc_coarse_long.yaml link=false cuda=0 learning_rate=5e-6,1e-5,5e-5 label_smoothing_factor=0.0,0.05,0.1 weight_decay=0.0,0.05 hydra/launcher=submitit_slurm -m
