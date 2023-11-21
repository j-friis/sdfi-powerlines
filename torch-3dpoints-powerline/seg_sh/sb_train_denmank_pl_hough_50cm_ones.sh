
CUDA_VISIABLE_DEVICES=1 HYDRA_FULL_ERROR=1 python -u train.py task=segmentation models=segmentation/minkowski  \
model_name=SEUNet18 data=segmentation/den_pl_hough_small_50cm_ones training.wandb.log=True training.wandb.project=NewDenmark-Seg-modified \
training.wandb.name=minkowski_powerline_segmentation_SEUNet18_50cm_ones training.batch_size=8


