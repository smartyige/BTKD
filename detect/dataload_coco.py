# ------------------------------------------------------------
# 适用于 FasterRCNN + COCO 的数据加载模块（Windows 路径）
# ------------------------------------------------------------

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CocoDetection
import os
import numpy as np


# ============================================================
#  COCO 路径配置
# ============================================================

COCO_ROOT = "G:\\Datasets\\COCO\\"

TRAIN_IMAGES = os.path.join(COCO_ROOT, "train2017")
VAL_IMAGES   = os.path.join(COCO_ROOT, "val2017")

TRAIN_ANN = os.path.join(COCO_ROOT, "annotations", "instances_train2017.json")
VAL_ANN   = os.path.join(COCO_ROOT, "annotations", "instances_val2017.json")


# ============================================================
#  FasterRCNN 的 transforms
# ============================================================

def get_transform(train=True):
    transforms = []

    # 转 tensor
    transforms.append(T.ToTensor())

    # 训练增强
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)


# ============================================================
#  COCO 标注格式转换为 FasterRCNN 所需格式
# ============================================================

def convert_coco_annotations(target):
    """
    输入：CocoDetection 返回的 list[dict]
    输出：符合 FasterRCNN 要求的 target dict
    """

    boxes = []
    labels = []

    for obj in target:
        # 跳过 crowd
        if obj.get("iscrowd", 0) == 1:
            continue

        # COCO bbox = [x, y, w, h]
        x, y, w, h = obj["bbox"]
        if w <= 0 or h <= 0:
            continue

        boxes.append([x, y, x + w, y + h])
        labels.append(obj["category_id"])

    if len(boxes) == 0:
        # 保证至少一个框，否则 FasterRCNN 会报错
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)
    else:
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

    return {"boxes": boxes, "labels": labels}


# ============================================================
#  自定义 COCO Dataset 封装
# ============================================================

class CocoDatasetWrapper(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms=None):
        super().__init__(img_folder, ann_file)
        self.transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        # 转换为 FasterRCNN 需要的格式
        target = convert_coco_annotations(target)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target


# ============================================================
#  collate_fn
# ============================================================

def collate_fn(batch):
    return tuple(zip(*batch))


# ============================================================
#  dataloader 构建
# ============================================================

def get_coco_dataloaders(batch_size=4, num_workers=0):
    print("[COCO] Loading dataset from:", COCO_ROOT)

    train_dataset = CocoDatasetWrapper(
        TRAIN_IMAGES, TRAIN_ANN,
        transforms=get_transform(train=True)
    )

    val_dataset = CocoDatasetWrapper(
        VAL_IMAGES, VAL_ANN,
        transforms=get_transform(train=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, train_dataset, val_dataset


# # ============================================================
# #  Quick Test
# # ============================================================
#
# if __name__ == "__main__":
#     train_loader, val_loader, _, _ = get_coco_dataloaders(batch_size=2)
#
#     for imgs, targets in train_loader:
#         print("Image batch size:", len(imgs))
#         print("Target example:", targets[0])
#         break
#
#     print("[COCO] dataload_coco test completed.")
