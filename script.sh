#!/bin/bash

#SBATCH --time=48:00:00 
#SBATCH --job-name=gan+_pesq
#SBATCH --account=PAS2301

#SBATCH --mem=96gb
#SBATCH --output /users/PAS2301/kibria5/Research/quality_enhancement/metricgan+/objective_pesq/training.out

#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1

module load miniconda3
source activate gan_env
python train.py