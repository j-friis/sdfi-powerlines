
CUDA_VISIABLE_DEVICES=1 HYDRA_FULL_ERROR=1 python -u train.py task=segmentation models=segmentation/minkowski  \
model_name=SEUNet18 data=segmentation/den_pl_hough_small_10cm_ones \
training.wandb.log=True training.wandb.project=NewDenmark-Seg-modified training.wandb.name=HOUGH_SEUNet18_10cm_ones \
training=denmark/minkowski training.batch_size=32 lr_scheduler=exponential lr_scheduler.params.gamma=0.998

