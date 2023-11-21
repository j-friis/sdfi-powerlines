
CUDA_VISIABLE_DEVICES=1 HYDRA_FULL_ERROR=1 python -u train.py task=segmentation models=segmentation/minkowski  \
model_name=SEUNet18 data=segmentation/den_pl_hough_small_20cm \
training.wandb.log=True training.wandb.project=NewDenmark-Seg-modified training.wandb.name=HOUGH_SEUNet18_20cm \
training=denmark/minkowski training.batch_size=32 lr_scheduler=exponential lr_scheduler.params.gamma=0.998

