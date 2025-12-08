# ------------------------------------------------------------
# Supports DeepLab (769×769) and SegFormer (1024×1024)
# ------------------------------------------------------------

import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
import torchvision.transforms.functional as TF
from torchvision import transforms
import random
import numpy as np


# ============================================================
# Utility Functions
# ============================================================

def random_resize(image, target, scale_range=(0.5, 2.0)):
    """Random rescale image and target."""
    w, h = image.size
    scale = random.uniform(scale_range[0], scale_range[1])
    new_w = int(w * scale)
    new_h = int(h * scale)

    image = TF.resize(image, (new_h, new_w), interpolation=TF.InterpolationMode.BILINEAR)
    target = TF.resize(target, (new_h, new_w), interpolation=TF.InterpolationMode.NEAREST)

    return image, target


def random_crop(image, target, crop_size):
    """Random crop keeping valid segmentation labels."""
    w, h = image.size
    cw, ch = crop_size

    if w < cw or h < ch:
        pad_w = max(cw - w, 0)
        pad_h = max(ch - h, 0)
        image = TF.pad(image, (0, 0, pad_w, pad_h))
        target = TF.pad(target, (0, 0, pad_w, pad_h))

        w, h = image.size

    x = random.randint(0, w - cw)
    y = random.randint(0, h - ch)

    image = TF.crop(image, y, x, ch, cw)
    target = TF.crop(target, y, x, ch, cw)
    return image, target


def random_flip(image, target):
    """Random horizontal flip."""
    if random.random() < 0.5:
        image = TF.hflip(image)
        target = TF.hflip(target)
    return image, target


def normalize_image(image):
    """Normalize image with ImageNet mean/std."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    image = TF.to_tensor(image)
    image = TF.normalize(image, mean, std)
    return image


def to_tensor_target(target):
    """Convert segmentation mask to tensor."""
    target = np.array(target, dtype=np.int64)
    target = torch.from_numpy(target)
    return target


# ============================================================
# Custom Dataset Wrapper
# ============================================================

class CityscapesDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train", mode="fine",
                 crop_size=(769, 769), is_train=True):
        super().__init__()

        self.dataset = Cityscapes(
            root=root,
            split=split,
            mode=mode,
            target_type="semantic"
        )
        self.crop_size = crop_size
        self.is_train = is_train

    def __getitem__(self, idx):
        image, target = self.dataset[idx]  # PIL images

        if self.is_train:
            # Random resize
            image, target = random_resize(image, target, scale_range=(0.5, 2.0))

            # Random crop
            image, target = random_crop(image, target, self.crop_size)

            # Random flip
            image, target = random_flip(image, target)

        # Normalize image
        image = normalize_image(image)

        # Target mask to tensor
        target = to_tensor_target(target)  # [H, W]

        return image, target

    def __len__(self):
        return len(self.dataset)


# ============================================================
# Loader Factory
# ============================================================

def get_cityscapes_loaders(root="../../datasets/cityscapes",
                           model_type="deeplab",
                           batch_size=4,
                           num_workers=4):

    if model_type.lower() == "deeplab":
        crop_size = (769, 769)
    elif model_type.lower() == "segformer":
        crop_size = (1024, 1024)
    else:
        raise ValueError("model_type must be 'deeplab' or 'segformer'")

    # Train Dataset
    train_dataset = CityscapesDataset(
        root=root,
        split="train",
        mode="fine",
        crop_size=crop_size,
        is_train=True
    )

    # Validation Dataset
    val_dataset = CityscapesDataset(
        root=root,
        split="val",
        mode="fine",
        crop_size=crop_size,
        is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"[Cityscapes] Train: {len(train_dataset)} images, "
          f"Val: {len(val_dataset)} images, "
          f"Crop Size: {crop_size}, Model Type: {model_type}")

    return train_loader, val_loader
