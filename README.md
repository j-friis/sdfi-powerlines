#  sdfi-powerlines

## Using the solution
The first thing you need to do is to install the required Python libraries. Our solution requires two separate Python virtual environments. This is because the PDAL library is incompatible with the rest of the libraries we use for deep earning. The environment with the PDAL library is called pdal, and the environment for the deep learning is called powerlines.

## Installation 
<!-- Install the two python envoriments using the install files in ./install -->
We provide two files for each of the installations of the environments, one for installation on the DIKU-GPU cluster and one for a local installation. To install the pdal env, use the file install `pdal_env.sh`. To install the powerlines nv, use the file install `powerlines_env.sh`.
Change directory into the folder 
```bash
cd install
```

Install the first environment(pdal)

```bash
bash install_pdal_env.sh 
```

Install the second environment(powerlines). For this env you need to have cuda installed on your PC, and set the correct path in file like this `export CUDA_HOME=/usr/local/cuda` 

```bash
bash install_powerlines_env.sh 
```
## Predicting on new data
To apply the models on some new data, you need to place the laz files in the ”raw/test” folder and make sure that the ”test” folder in the corresponding ”processed*” folder is empty, as otherwise, the model will use those blocks. To use our rule-based model, you can apply the script predict rule based.sh, which takes care of first applying the rule-based preprocessing, transforming them into blocks, predicting with the 3D CNN on the blocks, adding the predictions to a las file and saving it. Our multi-view model with the script predict multi view.sh can do the same thing. The las files with predictions will be saved in a folder ”denmark/eval”, with a new dimension ”prediction” equal to 1 if we classify the point as powerline and 0 otherwise.


Change directory to the *torch-3dpoints-powerline* folder 
```bash
cd torch-3dpoints-powerline
```
and run either the pipeline with the 2DCNN and Houghline
```bash
bash predict_multi_view.sh #2DCNN
bash predict_rule_based.sh #Houghline
```
###
The dataset is organized hierarchically, with the primary folder titled ”Denmark.” Within this primary directory is a subordinate folder named ”raw,” which further houses three subfolders: ’train’, ’val’, and ’test’. Each subfolder contains the laz files determining the data splits employed for the training, validation, and testing. 

When training or evaluating the model, the laz files need to be processed, including actions such as dividing them into smaller blocks and normalizing, among others. These blocks are stored in a separate folder labeled ”processed*” under the primary folder. The asterisk (*) in the folder name signifies what configuration of the dataset has been used. Under the ”processed*”, there will also be a ’train’, ’val’, and ’test’ folder, containing the block files for each split.


The library Hydra manages the configuration and is in the ”conf” folder. The configura-
tion used by our final rule-based model is ”data/segmentation/den pl hough small 20cm ones”.