#!/bin/bash
#SBATCH --job-name=O3D
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80000
#SBATCH --time=5-00:00:00

echo "Training Started"
module load cuda/11.3

eval "$(conda shell.bash hook)"
conda activate imageprocessing

echo "Training started by nxw500"
python outlier_removal_bayes_o3d.py ~/data/
