# BTKD

This repository provides the official implementation of **BTKD**. BTKD is a knowledge distillation framework for image classification, object detection, and semantic segmentation.

---

## Environment Setup

Our experiments were conducted with the following software environment:

- Python 3.9
- PyTorch 1.9.0 (CUDA 11.3, cuDNN 8.0)
- NumPy 1.21.0
- OpenCV-Python 4.1.2.30
- Matplotlib 3.4.3

These are common dependencies and can be installed via `pip` or `conda`.

Additional dependencies are required for different tasks:

- For YOLO-based object detection models: `ultralytics`
- For MiT-based semantic segmentation models: `timm`

This codebase supports both **Ubuntu** and **Windows**. The main difference between the two systems is the file path format.

### Pretrained Weights and Checkpoints

- **Image Classification**:  
  Please download the pretrained weights required for classification experiments and place them in the `./pth/` folder under the project root directory.

- **Semantic Segmentation**:  
  Please download the pretrained weights and trained checkpoints required for semantic segmentation experiments and place them in the `./segment/pth/` folder.

All experiments were conducted on **2 × RTX 4090 GPUs**.

---

## Usage

### 1. Image Classification

1. In `dataload_cf100.py`, modify the dataset path to your own data directory.
2. Scripts named `train_XXX_YYY.py` are used for from-scratch training, where:
   - `XXX` denotes the model or backbone network, such as ResNet or ViT;
   - `YYY` denotes the dataset, such as CIFAR-10 or CIFAR-100.
3. Scripts whose filenames start with `dis_` are used for BTKD-based knowledge distillation training.

   The corresponding best checkpoints are listed below:

   - Teacher: [h_resnet50_cf10.pth](https://drive.google.com/file/d/1Z1D65CAkQZSwZv0zrhk5JypnfC84hjm9/view?usp=drive_link)

   - Teacher: [h_resnet50_cf100.pth](https://drive.google.com/file/d/1C5eYZYx_yRb-XWeSeva3vktp_pfFrLFM/view?usp=drive_link)

   - Teacher: [h_resnet50_imagenet.pth](https://drive.google.com/file/d/11MJe7-bfDDt9oGVWxRwDcPwOJJ7NAiAP/view?usp=drive_link)

4. Run the following commands:

   - CIFAR-10: ResNet-50 → ResNet-18

     ```bash
     python dis_rn18_cf10.py
     ```

   - CIFAR-100: ResNet-50 → ResNet-18

     ```bash
     python dis_rn18_cf100.py
     ```

   - CIFAR-100: ResNet-50 → MobileNet-V1

     ```bash
     python dis_mobilenet_cf100.py
     ```

   - ImageNet: ResNet-50 → MobileNet

     ```bash
     python dis_mobilenet_imagenet.py
     ```

---

### 2. Object Detection

For object detection experiments, please enter the `detect` directory and run the scripts whose filenames start with `train_`.

These scripts are used to train object detection models, such as YOLO, under the BTKD framework.

---

### 3. Semantic Segmentation

For semantic segmentation experiments, please enter the `segment` directory and run the scripts whose filenames start with `train_`.

The default teacher–student settings and the corresponding best checkpoints are listed below:

- Teacher: [deeplabv3_resnet101.pth](https://drive.google.com/file/d/1qGLpc-RsLroaeW3hcFaZhmxIvSYdNloL/view?usp=sharing)  
  Student: [deeplabv3_resnet18.pth](https://drive.google.com/file/d/1KTh6FSdJHuq4KCFrPEsHZOHRqo1L0gAj/view?usp=sharing)  
  Best checkpoint: [btkd_deeplabv3_resnet18.pth](https://drive.google.com/file/d/1afcmZ30mT7FboKtxRdHP98U58xnfUbIf/view?usp=sharing)

- Teacher: [mit_B4.pth]()  
  Student: [mit_B0.pth](https://drive.google.com/file/d/1Ave0JlLWIPHZifBCjzr68FyOG9N9BHht/view?usp=sharing)  
  Best checkpoint: [btkd_mit_B0.pth](https://drive.google.com/file/d/1aG02z4vxBE89nxpOJ33VltQMQhrE1bMj/view?usp=sharing)

All the above weights and checkpoint files should be placed in the `./segment/pth/` folder.

---

### 4. Configuration Files and Hyperparameter Settings

This repository currently does not provide additional standalone YAML or JSON configuration files. The main training configurations have been directly written into the corresponding training scripts.

In other words, each released script corresponds to a fixed experimental setting, and the script itself serves as the executable configuration file for that experiment. It includes:

- teacher model architecture;
- student model architecture;
- dataset name;
- batch size;
- optimizer;
- learning rate;
- number of training epochs;
- distillation temperature;
- BTKD / BTKD++ related loss weights;
- checkpoint saving path;
- evaluation settings.

To modify the hyperparameters, please directly edit the default settings in the corresponding script.

### 5. Random Seed Settings

To improve the stability and reproducibility of experimental results, this repository recommends using the following five random seeds:

```text
1, 2, 3, 4, 5
```

## Notes

Due to limited time and computational resources, this repository is still being continuously improved.  
We will gradually release more standardized and complete code, more experimental results, and additional pretrained models.

---

## Citation

If you find this repository helpful for your research, please consider citing our paper:

```bibtex
@inproceedings{zhang2025can,
  title={Can students beyond the teacher? distilling knowledge from teacher’s bias},
  author={Zhang, Jianhua and Gao, Yi and Liu, Ruyu and Cheng, Xu and Zhang, Houxiang and Chen, Shengyong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={21},
  pages={22434--22442},
  year={2025}
}
```
