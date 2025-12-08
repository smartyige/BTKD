# BTKD

This repository provides the official implementation of **BTKD**, a knowledge distillation framework for image classification, object detection, and semantic segmentation.

---

## Environment

Our experiments were conducted with the following software stack:

- Python 3.9  
- PyTorch 1.9.0 (CUDA 11.3, cuDNN 8.0)  
- NumPy 1.21.0  
- OpenCV-Python 4.1.2.30  
- Matplotlib 3.4.3  

These are standard dependencies and can be installed via `pip` or `conda`.

Additional task-specific dependencies:

- For YOLO-based detection models: `ultralytics`
- For MiT-based segmentation models: `timm`

The codebase supports both **Ubuntu** and **Windows**; the only difference is the style of file paths.

### Pretrained Weights & Checkpoints

- **Classification**:  
  Download the pretrained classification weights and place them in the root directory under `./pth/`.

- **Segmentation**:  
  Download the segmentation pretrained weights and trained checkpoints and place them in `./segment/pth/`.

All experiments were run on **2 × RTX 4090 GPUs**.

---

## Usage

### 1. Classification

1. In `dataload_cf100.py`, update the dataset path to your own data directory.
2. Scripts named `train_XXX_YYY.py` perform *from-scratch* training, where:
   - `XXX` denotes the model/backbone (e.g., ResNet, ViT, etc.),
   - `YYY` denotes the dataset (e.g., CIFAR10, CIFAR100).
3. Scripts whose filenames start with `dis_` implement BTKD-based knowledge distillation.
   
   The default teacher–student settings and best checkpoints are:
   
   - Teacher: [h_resnet50_cf10.pth](https://drive.google.com/file/d/1Z1D65CAkQZSwZv0zrhk5JypnfC84hjm9/view?usp=drive_link)

   - Teacher: [h_resnet50_cf100.pth](https://drive.google.com/file/d/1C5eYZYx_yRb-XWeSeva3vktp_pfFrLFM/view?usp=drive_link)

---

### 2. Object Detection

For object detection, navigate to the `detect` directory and run the scripts whose names start with `train_`.  
These scripts train detection models (e.g., YOLO) under the BTKD framework.

---

### 3. Semantic Segmentation

For semantic segmentation, navigate to the `segment` directory and run the scripts whose names start with `train_`.

The default teacher–student settings and best checkpoints are:

- Teacher: [deeplabv3_resnet101.pth]()  
  Student: [deeplabv3_resnet18.pth]()  
  Best checkpoint: [btkd_deeplabv3_resnet18.pth]()

- Teacher: `mit_B4.pth`  
  Student: `mit_B0.pth`  
  Best checkpoint: `btkd_mit_B0.pth`

All of the above should be placed under `./segment/pth/`.

---

## Notes

Due to limited time and resources, this repository is still evolving.  
More polished code, additional experiments, and extra pretrained models will be released progressively.


# To Be Continue...
