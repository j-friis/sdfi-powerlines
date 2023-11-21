#!/bin/bash

n_cpu=$(nproc --all)
echo ${n_cpu}
n_cpu=$((n_cpu-1))

echo "Runing pdal pipeline"
eval "$(conda shell.bash hook)"

conda activate pdal
python torch_points3d/core/data_transform/pdal/run_pipeline.py ~/data/denmark/raw/train $n_cpu 3.5
python torch_points3d/core/data_transform/pdal/run_pipeline.py ~/data/denmark/raw/test $n_cpu 3.5
python torch_points3d/core/data_transform/pdal/run_pipeline.py ~/data/denmark/raw/val $n_cpu 3.5

conda activate powerlines
bash seg_sh/sb_train_denmank_pl_hough.sh
