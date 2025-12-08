# BTKD

# Environment
## Our environment is: python==3.9, numpy==1.21.0, pytorch==1.9.0, cuda==11.3, cudnn==8.0, opencv-python==4.1.2.30, matplotlib==3.4.3. These are all basic libraries.
## To run the YOLO models, you need to install the `ultralytics` library.
## To run the MiT segmentation models, you need to install the `timm` library.
## Both Ubuntu and Windows are supported; only the file path formats are different.
## Download the classification pretrained weights and place them in the `pth` folder under the project root directory.
## Download the segmentation pretrained weights and trained weights, and place them in the `segment/pth` folder.
## My GPU setup is 2 × RTX 4090.

# How to Run

## Classification
## In `dataload_cf100.py`, modify the dataset path to your own.
## Files named `train_XXX_YYY.py` are for training from scratch, where `XXX` is the model type and `YYY` is the dataset.
## Files starting with `dis_` are for knowledge distillation. Before running them, first modify the paths for loading the `.pth` files.

## Object Detection
## In the `detect` folder, run the scripts whose names start with `train`.

## Segmentation
## In the `segment` folder, run the scripts whose names start with `train`.

## Pretrained Weights
## Teacher: `deeplabv3_resnet101.pth`  ||  Student: `deeplabv3_resnet18.pth`  ||  Best model: `btkd_deeplabv3_resnet18.pth`
## Teacher: `mit_B4.pth`               ||  Student: `mit_B0.pth`               ||  Best model: `btkd_mit_B0.pth`

# Notes
## Due to limited time and resources, the codebase will be updated gradually in the future.



# To Be Continue...
