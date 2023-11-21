#!/bin/bash
#SBATCH --job-name=nxw500SBayesSearchCV
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48000
#SBATCH --time=5-00:00:00

echo "Training Started"
module load cuda/11.3

eval "$(conda shell.bash hook)"
conda activate imageprocessing


echo "Training started by nxw500"
python polygon_classifier_small.py ~/data/
