# ------------------------------------------------------------
# YOLOv8-m → YOLOv8-n 的 BTKD++ 训练脚本
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import math

from dis_model_yolo import build_yolo_teacher, build_yolo_student
from dataload_coco import get_coco_dataloaders
from btkd_yolo_loss import btkd_yolo_loss


# ============================================================
# γ 动态课程学习（Easy→Hard）
# ============================================================

def gamma_schedule(epoch, epochs):
    """
    前期学习 easy，后期学习 hard。
    γ: 0 → 1
    """
    x = epoch / epochs
    return 1 - math.cos((math.pi/2) * x)**2


# ============================================================
# YOLO BTKD++ 训练函数
# ============================================================

def train_yolo_btkd(
    teacher_model,
    student_model,
    epochs=200,
    batch_size=4,
    lr=1e-4,
    T=4.0,
    lambda_stu=0.5,
    save_path="./pth/yolo_n_btkd.pth"
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher_model.to(device)
    student_model.to(device)

    teacher_model.eval()   # Teacher 永远不更新
    optimizer = optim.Adam(student_model.parameters(), lr=lr)

    train_loader, val_loader, train_set, val_set = get_coco_dataloaders(batch_size=batch_size)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        student_model.train()

        gamma = gamma_schedule(epoch, epochs)
        print(f"\n[Epoch {epoch+1}/{epochs}] gamma={gamma:.4f}")

        running_loss = 0.0

        for imgs, targets in tqdm(train_loader, desc="Training"):

            imgs = torch.stack(imgs).to(device)

            # YOLO raw outputs
            bbox_t, obj_t, cls_t = teacher_model(imgs)
            bbox_s, obj_s, cls_s = student_model(imgs)

            # YOLO raw outputs shape:
            # bbox_s: [B, N, 4]
            # obj_s:  [B, N, 1]
            # cls_s:  [B, N, 80]

            loss_total_batch = 0.0

            # 按“每张图片”单独蒸馏
            for i in range(len(targets)):
                t = targets[i]
                # 每张图的 predictions
                bbox_t_i = bbox_t[i]
                obj_t_i  = obj_t[i]
                cls_t_i  = cls_t[i]

                bbox_s_i = bbox_s[i]
                obj_s_i  = obj_s[i]
                cls_s_i  = cls_s[i]

                # Compute distillation loss for this image
                loss_i, info_i = btkd_yolo_loss(
                    bbox_s=bbox_s_i,
                    obj_s=obj_s_i,
                    cls_s=cls_s_i,
                    bbox_t=bbox_t_i,
                    obj_t=obj_t_i,
                    cls_t=cls_t_i,
                    targets=t,
                    T=T,
                    gamma=gamma,
                    lambda_stu=lambda_stu
                )

                loss_total_batch += loss_i

            optimizer.zero_grad()
            loss_total_batch.backward()
            optimizer.step()

            running_loss += loss_total_batch.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"TrainLoss: {avg_train_loss:.4f}")

        # ----------------------------------------------------
        # 验证
        # ----------------------------------------------------
        student_model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for imgs, targets in tqdm(val_loader, desc="Validation"):
                imgs = torch.stack(imgs).to(device)

                bbox_t, obj_t, cls_t = teacher_model(imgs)
                bbox_s, obj_s, cls_s = student_model(imgs)

                batch_val_loss = 0.0

                for i in range(len(targets)):
                    t = targets[i]

                    loss_i, _ = btkd_yolo_loss(
                        bbox_s=bbox_s[i], obj_s=obj_s[i], cls_s=cls_s[i],
                        bbox_t=bbox_t[i], obj_t=obj_t[i], cls_t=cls_t[i],
                        targets=t,
                        T=T,
                        gamma=gamma,
                        lambda_stu=lambda_stu
                    )
                    batch_val_loss += loss_i.item()

                val_loss += batch_val_loss

        val_loss /= len(val_loader)
        print(f"ValLoss: {val_loss:.4f}")

        # ----------------------------------------------------
        # 保存 best student
        # ----------------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(student_model.state_dict(), save_path)
            print(f"Best YOLO BTKD++ model saved to {save_path}")

    print("\nTraining Completed! Best loss =", best_val_loss)
