# ------------------------------------------------------------
# Full Training Script for BTKD++ + CIRKD Segmentation KD
# ------------------------------------------------------------

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from dataload_cityscapes import get_cityscapes_loaders
from btkd_seg_loss import BTKD_SegLoss

# === Import your corrected model builders ===
from dis_model_seg import (
    build_deeplabv3_r101,
    build_deeplabv3_r18,
    build_deeplabv3_mbv2,
    build_segformer_b4,
    build_segformer_b0,
)


# ============================================================
# Train One Epoch
# ============================================================

def train_one_epoch(model_s, model_t, optimizer, loss_fn, train_loader, device, epoch, epochs):

    model_s.train()
    model_t.eval()

    total_loss = 0

    pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")

    for img, label in pbar:
        img = img.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        # Forward teacher (no grad)
        with torch.no_grad():
            logits_t, feat_t = model_t(img)

        # Forward student
        logits_s, feat_s = model_s(img)

        # Compute KD loss
        loss, loss_dict = loss_fn(
            logits_s,
            logits_t,
            feat_s,
            feat_t,
            label,
            epoch,
            epochs
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "btkd": f"{loss_dict['btkd']:.3f}",
            "p2p": f"{loss_dict['p2p_batch']:.3f}",
            "intra": f"{loss_dict['intra']:.3f}",
            "gamma": f"{loss_dict['gamma']:.3f}"
        })

    return total_loss / len(train_loader)


# ============================================================
# Validation (Pixel Accuracy)
# ============================================================

@torch.no_grad()
def validate(model_s, val_loader, device):
    model_s.eval()

    total_correct = 0
    total_pixel = 0

    for img, label in tqdm(val_loader, desc="Validating"):
        img = img.to(device)
        label = label.to(device)

        logits, _ = model_s(img)
        pred = torch.argmax(logits, dim=1)

        mask = (label != 255)
        total_correct += (pred[mask] == label[mask]).sum().item()
        total_pixel += mask.sum().item()

    return total_correct / total_pixel


# ============================================================
# Main KD Training Function
# ============================================================

def train_kd_seg(
        model_type="deeplabv3",    # "deeplab" or "mit"
        student_type="resnet18", # resnet18 / mbv2 / b0
        epochs=200,
        batch_size=4,
        lr=1e-4,
        device="cuda:0"
):

    print(f"\n=== BTKD++ + CIRKD Segmentation KD Training ===")
    print(f" Model: {model_type}, Student: {student_type}")
    print("==============================================\n")

    # -------------------------------
    #  DataLoader
    # -------------------------------
    train_loader, val_loader = get_cityscapes_loaders(
        root="./datasets/cityscapes",
        model_type=model_type,
        batch_size=batch_size,
        num_workers=4
    )

    # -------------------------------
    # Build Teacher & Student
    # -------------------------------
    if model_type == "deeplab":

        teacher = build_deeplabv3_r101(num_classes=19)

        if student_type == "resnet18":
            student = build_deeplabv3_r18(num_classes=19)
        elif student_type == "mbv2":
            student = build_deeplabv3_mbv2(num_classes=19)
        else:
            raise ValueError("Unknown student type for DeepLab")

    elif model_type == "segformer":

        teacher = build_segformer_b4(num_classes=19)

        if student_type == "b0":
            student = build_segformer_b0(num_classes=19)
        else:
            raise ValueError("Unknown student type for SegFormer")

    teacher = teacher.to(device)
    student = student.to(device)

    teacher.eval()

    # -------------------------------
    #  Loss + Optimizer (参数自己调)
    # -------------------------------
    loss_fn = BTKD_SegLoss(
        num_classes=19,
        pixel_dim=256,
        queue_pixel_len=256,
        queue_region_len=64,
        tau=0.5,
        lambda_btkd=1.0,
        lambda_batch=1.0,
        lambda_mem_pixel=1.0,
        lambda_mem_region=1.0,
        lambda_intra=1.0,
        lambda_inter=1.0
    ).to(device)

    optimizer = Adam(student.parameters(), lr=lr)

    # -------------------------------
    #  Training Loop
    # -------------------------------
    best_acc = 0
    save_path = f"./pth/btkd_{model_type}_{student_type}.pth"

    for epoch in range(epochs):

        loss = train_one_epoch(
            student,
            teacher,
            optimizer,
            loss_fn,
            train_loader,
            device,
            epoch,
            epochs
        )

        acc = validate(student, val_loader, device)

        print(f"[Epoch {epoch+1}/{epochs}] Loss={loss:.4f}, PixelAcc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(student.state_dict(), save_path)
            print(f"✓ New best model saved: {save_path} (Acc={best_acc:.4f})\n")

    print(f"Training Finished. Best PixelAcc={best_acc:.4f}")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    train_kd_seg(
        model_type="deeplab",   # deeplab / segformer
        student_type="resnet18",# resnet18 / mbv2 / b0
        epochs=200,
        batch_size=4,
        lr=1e-4,
        device="cuda:0"
    )
