# ------------------------------------------------------------
# YOLOv8 的 BTKD++ 知识蒸馏损失（分类 + obj + bbox）
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from btkd_yolo_utils import (
    match_predictions_with_gt,
    check_yolo,
    discretize_bbox,
    rectify_cls_logits,
    rectify_obj_logits
)


# ============================================================
# KL Divergence（batchmean）
# ============================================================

def kl_div(p_s, p_t):
    return F.kl_div(
        F.log_softmax(p_s, dim=-1),
        F.softmax(p_t, dim=-1),
        reduction="batchmean"
    )


# ============================================================
# YOLO BTKD++ 主损失
# ============================================================

def btkd_yolo_loss(
    bbox_s, obj_s, cls_s,
    bbox_t, obj_t, cls_t,
    targets,
    iou_th=0.5,
    T=4.0,
    gamma=0.5,
    lambda_stu=0.5
):
    """
    输入：
        bbox_s/bbox_t: [N, 4]
        obj_s/obj_t:   [N, 1]
        cls_s/cls_t:   [N, 80]
        targets: GT dict {'boxes':[M,4], 'labels':[M]}

    输出：
        总损失 + 统计项
    """

    device = cls_s.device

    # ========================================================
    #  匹配 GT
    # ========================================================

    (
        pred_boxes_t,
        pred_cls_t,
        pred_obj_t,
        gt_boxes,
        gt_labels,
        iou_vals
    ) = match_predictions_with_gt(bbox_t, cls_t, obj_t, targets, iou_th)

    # Student 用相同 matched index
    pred_boxes_s, pred_cls_s, pred_obj_s, _, _, _ = match_predictions_with_gt(
        bbox_s, cls_s, obj_s, targets, iou_th
    )

    M = gt_labels.shape[0]     # 正样本数量


    # ========================================================
    #  Easy / Hard
    # ========================================================

    mask_easy, mask_hard = check_yolo(
        pred_cls_t, pred_boxes_t,
        gt_labels, gt_boxes,
        iou_vals,
        iou_th=iou_th
    )


    # ========================================================
    #  分类蒸馏（easy / hard）
    # ========================================================

    # ---------- easy ----------
    if mask_easy.sum() > 0:
        cls_s_easy = pred_cls_s[mask_easy] / T
        cls_t_easy = pred_cls_t[mask_easy] / T
        loss_cls_easy = kl_div(cls_s_easy, cls_t_easy) * (T * T)
    else:
        loss_cls_easy = torch.tensor(0., device=device)

    # ---------- hard ----------
    if mask_hard.sum() > 0:
        cls_t_rect = rectify_cls_logits(pred_cls_t[mask_hard], gt_labels[mask_hard])

        cls_s_hard = pred_cls_s[mask_hard] / T
        cls_t_hard = cls_t_rect / T

        loss_cls_hard = kl_div(cls_s_hard, cls_t_hard) * (T * T)
    else:
        loss_cls_hard = torch.tensor(0., device=device)


    # ========================================================
    #  objectness 蒸馏（easy / hard）
    # ========================================================

    # ---------- easy ----------
    if mask_easy.sum() > 0:
        loss_obj_easy = F.mse_loss(pred_obj_s[mask_easy], pred_obj_t[mask_easy])
    else:
        loss_obj_easy = torch.tensor(0., device=device)

    # ---------- hard ----------
    if mask_hard.sum() > 0:
        obj_t_rect = rectify_obj_logits(pred_obj_t, mask_hard)
        loss_obj_hard = F.mse_loss(pred_obj_s[mask_hard], obj_t_rect[mask_hard])
    else:
        loss_obj_hard = torch.tensor(0., device=device)


    # ========================================================
    #  bbox 蒸馏（离散化 + rectification）
    # ========================================================

    loss_bbox_easy = torch.tensor(0., device=device)
    loss_bbox_hard = torch.tensor(0., device=device)

    num_bins = 64
    t_ranges = [(-2, 2), (-2, 2), (-2, 2), (-2, 2)]

    for k in range(4):
        tmin, tmax = t_ranges[k]

        # -------- easy --------
        if mask_easy.sum() > 0:
            dist_s = discretize_bbox(pred_boxes_s[mask_easy][:, k], tmin, tmax, num_bins)
            dist_t = discretize_bbox(pred_boxes_t[mask_easy][:, k], tmin, tmax, num_bins)
            loss_bbox_easy += kl_div(dist_s, dist_t)

        # -------- hard --------
        if mask_hard.sum() > 0:
            dist_s = discretize_bbox(pred_boxes_s[mask_hard][:[:, k]], tmin, tmax, num_bins)
            dist_t = discretize_bbox(pred_boxes_t[mask_hard][:, k], tmin, tmax, num_bins)

            # 增强 GT → rectified
            gt_bins = torch.clamp(
                ((gt_boxes[mask_hard][:, k] - tmin) / (tmax - tmin) * num_bins).long(),
                0, num_bins - 1
            )
            gt_onehot = F.one_hot(gt_bins, num_bins).float().to(device)

            dist_t_rect = (dist_t + gt_onehot) / 2
            dist_t_rect = dist_t_rect / (dist_t_rect.sum(dim=1, keepdim=True) + 1e-7)

            loss_bbox_hard += kl_div(dist_s, dist_t_rect)


    # ========================================================
    #  合并 Easy / Hard（动态 γ）
    # ========================================================

    loss_kd_easy = loss_cls_easy + loss_obj_easy + loss_bbox_easy
    loss_kd_hard = loss_cls_hard + loss_obj_hard + loss_bbox_hard

    loss_kd = (1 - gamma) * loss_kd_easy + gamma * loss_kd_hard


    # ========================================================
    #  YOLO CE（与 FasterRCNN 前面的 CE 类似）
    #          YOLO 没有 CE 结构，只训练蒸馏项。
    # ========================================================

    loss_total = (1 - lambda_stu) * loss_kd


    # ========================================================
    # 输出日志打印信息
    # ========================================================

    return loss_total, {
        "cls_easy": loss_cls_easy.item(),
        "cls_hard": loss_cls_hard.item(),
        "obj_easy": loss_obj_easy.item(),
        "obj_hard": loss_obj_hard.item(),
        "bbox_easy": loss_bbox_easy.item(),
        "bbox_hard": loss_bbox_hard.item(),
        "kd_easy": loss_kd_easy.item(),
        "kd_hard": loss_kd_hard.item(),
        "loss_kd": loss_kd.item(),
        "loss_total": loss_total.item()
    }
