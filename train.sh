#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

##SBATCH --mail-type=ALL
#SBATCH --output=%j-%x.%u.out
#SBATCH --job-name=default
#SBATCH --account=m3691

conda activate /global/homes/z/zangwei/.conda/envs/rodnet

srun python $@
