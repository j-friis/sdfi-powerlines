conda create -n pdal_test python=3.8 -y
eval "$(conda shell.bash hook)"
conda activate pdal_test
conda install -c conda-forge pdal python-pdal gdal
pip install laspy[lazrs]
