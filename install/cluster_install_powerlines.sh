#!/bin/bash
#BATCH --job-name=install
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=10000M
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --time=4-00:00:00
TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX"
echo "$TORCH_CUDA_ARCH_LIST"
module load miniconda/4.12.0
conda create -n powerlines python=3.8 -y
eval "$(conda shell.bash hook)"
conda activate powerlines

echo "Load module cuda/11.3"
module load cuda/11.3
echo "Install started by xdr476"

export CUDA_HOME=/opt/software/cuda/11.3
echo $CUDA_HOME

pip install numpy==1.20.3
pip install pandas
pip install wandb
pip install opencv-python
pip install hydra-core
pip install shapely
pip install rasterio
pip install laspy[lazrs,laszip]
pip install open3d
pip install tensorboard
pip install h5py
pip install plyfile
pip install gdown
pip install ipdb
pip install argparse

conda install openblas-devel -c anaconda
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-points-kernels==0.6.10
pip install torch-scatter==2.1.0 torch-sparse==0.6.16 -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
pip install torch-geometric
pip install torchnet
pip install pytorch-metric-learning
pip install torch ninja
MAX_JOBS=1 pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"


python -c "import torch; print('Cuda is available: ', torch.cuda.is_available())"
python -c "import torch; print('Torch version: ',torch.__version__)"
python -c "import torch; print('Torch version cuda: ', torch.version.cuda)"
