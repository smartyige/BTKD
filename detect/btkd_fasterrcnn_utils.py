# ------------------------------------------------------------
# FasterRCNN BTKD++ 辅助工具函数
# 包括：ROI 提取、IoU、bbox 离散化、rectification
# ------------------------------------------------------------

import torch
import torch.nn.functional as F
import math


# ============================================================
#  IoU 计算
# ============================================================

def box_iou(boxes1, boxes2):
    """
    boxes1: [N, 4]
    boxes2: [M, 4]
    output: IoU matrix [N, M]
    """

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * \
            (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * \
            (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # top-left
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # bottom-right

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-7)


# ============================================================
#  获取 FasterRCNN ROIHeads 的正样本（positive proposals）
# ============================================================

def extract_positive_samples(roi_losses, roi_features):
    """
    从 ROIHeads 的返回中提取 student/teacher 使用的 positive proposals。
    FasterRCNN 在内部会将匹配后的信息存放于 roi_losses['labels'], roi_losses['regression_targets'] 中。
    """

    labels = roi_losses['labels']          # List[Tensors]
    regression_targets = roi_losses['regression_targets']

    # 取 batch 中所有正样本
    pos_cls = []
    pos_reg = []
    pos_feat = []

    for i in range(len(labels)):
        mask_pos = labels[i] > 0
        pos_cls.append(labels[i][mask_pos])
        pos_reg.append(regression_targets[i][mask_pos])
        pos_feat.append(roi_features[i][mask_pos])

    return pos_cls, pos_reg, pos_feat


# ============================================================
#  预测值高斯离散化（ Gaussian-like Bins）
# ============================================================

def discretize_bbox_targets(t, t_min, t_max, num_bins=128, sigma=2.0):
    """
    t: continuous regression value (tensor)
    t_min, t_max: normalization range
    output: probability distribution [num_bins]
    """
    # Normalize
    t_norm = (t - t_min) / (t_max - t_min)
    t_norm = t_norm.clamp(0, 1)

    # Create bins
    bin_centers = torch.linspace(0, 1, num_bins, device=t.device)

    # Gaussian-like distribution
    dist = torch.exp(-(bin_centers - t_norm[..., None]) ** 2 / (2 * sigma ** 2))
    dist = dist / (dist.sum(dim=-1, keepdim=True) + 1e-7)

    return dist


# ============================================================
#  Hard-task rectification
#  使用分类 logits rectification + 回归分布 rectification
# ============================================================

def rectify_logits_teacher(teacher_logits, gt_labels):
    """
    对 Hard-task 分类 logits 进行比例校正（与分类 calibrate3 对齐）
    """

    teacher_logits = teacher_logits.clone()
    idx_max = torch.argmax(teacher_logits, dim=1)

    for i in range(len(gt_labels)):
        gt = gt_labels[i]
        wrong = idx_max[i]
        if wrong != gt:
            sum_old = teacher_logits[i][gt] + teacher_logits[i][wrong]

            teacher_logits[i][gt] = (teacher_logits[i][gt] + 1) / 2
            teacher_logits[i][wrong] = (teacher_logits[i][wrong] + 0) / 2

            sum_new = teacher_logits[i][gt] + teacher_logits[i][wrong]
            scale = sum_old / (sum_new + 1e-7)

            teacher_logits[i][gt] *= scale
            teacher_logits[i][wrong] *= scale

    return teacher_logits


# ============================================================
#  根据分类 + IoU 判定 Easy / Hard 样本
# ============================================================

def check_det(teacher_logits, teacher_boxes, gt_labels, gt_boxes, iou_th=0.5):
    """
    teacher_logits: [N, C]
    teacher_boxes:  [N, 4]
    gt_labels:      [N]
    gt_boxes:       [N, 4]
    output: mask_easy (True=easy), mask_hard
    """

    pred_cls = torch.argmax(teacher_logits, dim=1)
    correct_cls = pred_cls == gt_labels

    iou = box_iou(teacher_boxes, gt_boxes).diag()
    good_box = iou >= iou_th

    mask_easy = correct_cls & good_box
    mask_hard = ~mask_easy

    return mask_easy, mask_hard, iou


# ============================================================
#  将 Hard-task 的教师回归分布进行校正
# ============================================================

def rectify_bbox_distribution(teacher_dist, gt_onehot):
    """
    teacher_dist: [N, B] — teacher predicted soft distribution
    gt_onehot:    [N, B] — GT one-hot distribution
    """

    new_dist = (teacher_dist + gt_onehot) / 2
    new_dist = new_dist / (new_dist.sum(dim=1, keepdim=True) + 1e-7)
    return new_dist


# ============================================================


# ============================================================

print("[BTKD DET UTILS] Loaded all helper functions.")
