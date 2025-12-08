# ------------------------------------------------------------
# 完整可运行的 FasterRCNN + BTKD++ 训练脚本
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import math
from tqdm import tqdm

from dis_model_det import (
    build_fasterrcnn_resnet101,
    build_fasterrcnn_resnet18,
    build_fasterrcnn_resnet50,
    build_fasterrcnn_mobilenetv2
)

from dataload_coco import get_coco_dataloaders
from btkd_fasterrcnn_loss import btkd_loss


# ============================================================
# γ 动态课程学习
# ============================================================

def gamma_schedule(epoch, epochs):
    """前期强调 easy，后期强调 hard。"""
    x = epoch / epochs
    return 1 - math.cos((math.pi / 2) * x) ** 2   # 0 → 1 递增


# ============================================================
# 训练脚本
# ============================================================

def train_btkd(
    teacher_model,
    student_model,
    epochs=200,
    lr=1e-4,
    lambda_stu=0.5,
    T=4.0,
    save_path="./pth/btkd_fasterrcnn_student.pth",
    batch_size=8
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)

    # Teacher
    teacher_model.eval()

    # optimizer
    optimizer = optim.Adam(student_model.parameters(), lr=lr)

    # Dataloaders
    train_loader, val_loader, train_set, val_set = get_coco_dataloaders(
        batch_size=batch_size
    )

    best_val_loss = float("inf")

    for epoch in range(epochs):
        student_model.train()
        running_loss = 0.0

        gamma = gamma_schedule(epoch, epochs)  # 动态 γ
        print(f"\n[Epoch {epoch+1}/{epochs}] gamma={gamma:.4f}")

        train_loader_tqdm = tqdm(train_loader)

        for images, targets in train_loader_tqdm:

            images = [img.to(device) for img in images]
            targets_gpu = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # ---------------------
            # 获取 GT 信息（labels + boxes）
            # ---------------------
            gt_labels_all = torch.cat([t["labels"] for t in targets_gpu], dim=0)
            gt_boxes_all = torch.cat([t["boxes"] for t in targets_gpu], dim=0)

            # ---------------------
            # Teacher forward
            # ---------------------
            with torch.no_grad():
                teacher_out = teacher_model(images)
            teacher_logits = torch.cat([o["scores"].unsqueeze(1) * torch.nn.functional.one_hot(o["labels"], 91)
                                         for o in teacher_out], dim=0)

            teacher_boxes = torch.cat([o["boxes"] for o in teacher_out], dim=0)

            # ---------------------
            # Student forward
            # ---------------------
            student_out = student_model(images)
            student_logits = torch.cat([o["scores"].unsqueeze(1) * torch.nn.functional.one_hot(o["labels"], 91)
                                         for o in student_out], dim=0)
            student_boxes = torch.cat([o["boxes"] for o in student_out], dim=0)

            # ---------------------
            # FasterRCNN 内部的 GT 回归目标（正样本 matched）
            # ---------------------
            # FasterRCNN 内部已经自动生成 matched regression_targets 和 labels
            # student_model(images, targets) 返回一个 loss dict
            loss_dict_stu = student_model(images, targets_gpu)

            if "bbox_regression" in loss_dict_stu:
                box_gt = loss_dict_stu["bbox_regression"]
            else:
                box_gt = student_boxes.clone()   # fallback

            # ---------------------
            # BTKD++ loss
            # ---------------------
            loss_total, loss_info = btkd_loss(
                cls_s=student_logits,
                cls_t=teacher_logits,
                box_s=student_boxes,
                box_t=teacher_boxes,
                box_gt=box_gt,
                gt_labels=gt_labels_all,
                gt_boxes=gt_boxes_all,
                iou_th=0.5,
                T=T,
                gamma=gamma,
                lambda_stu=lambda_stu
            )

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            running_loss += loss_total.item()

            train_loader_tqdm.set_description(
                f"Train Loss: {loss_total.item():.4f}"
            )

        # ---------------------
        # Validation（简单验证 loss）
        # ---------------------
        student_model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation"):
                images = [img.to(device) for img in images]
                targets_gpu = [{k: v.to(device) for k, v in t.items()} for t in targets]

                out = student_model(images, targets_gpu)
                val_loss += sum(loss for loss in out.values()).item()

        val_loss /= len(val_loader)
        print(f"[Epoch {epoch+1}] TrainLoss={running_loss/len(train_loader):.4f}, ValLoss={val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(student_model.state_dict(), save_path)
            print(f"Best model saved to {save_path}")

    print("Training completed!")


# ============================================================
# Main Run Example
# ============================================================

if __name__ == "__main__":

    # 选择教师与学生模型
    teacher = build_fasterrcnn_resnet101(num_classes=91)
    student = build_fasterrcnn_resnet18(num_classes=91)

    train_btkd(
        teacher_model=teacher,
        student_model=student,
        epochs=12,
        batch_size=2,
        lr=1e-4,
        T=4.0,
        lambda_stu=0.5,
        save_path="./pth/fasterrcnn_res18_btkd.pth"
    )
