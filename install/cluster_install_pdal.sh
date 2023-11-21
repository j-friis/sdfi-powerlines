#!/bin/bash
#BATCH --job-name=install_pdal
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=16000M
#SBATCH --time=4-00:00:00

module load miniconda/4.12.0
conda create -n pdal python=3.8 -y
eval "$(conda shell.bash hook)"
conda activate pdal


conda install -c conda-forge pdal python-pdal gdal
pip install laspy[lazrs]
