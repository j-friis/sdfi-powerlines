conda create -n pdal python=3.8 -y
eval "$(conda shell.bash hook)"
conda activate pdal
conda install -c conda-forge pdal python-pdal gdal
pip install laspy[lazrs]
