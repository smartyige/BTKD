# ------------------------------------------------------------
# BTKD++ detection loss for FasterRCNN
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from btkd_fasterrcnn_utils import (
    discretize_bbox_targets,
    rectify_logits_teacher,
    rectify_bbox_distribution,
    check_det,
)

# ============================================================
# KL divergence for probability distributions
# ============================================================

def kl_loss(p_stu, p_tea):
    """
    KLDivLoss with batchmean reduction
    """
    return F.kl_div(
        F.log_softmax(p_stu, dim=-1),
        F.softmax(p_tea, dim=-1),
        reduction="batchmean"
    )


# ============================================================
#  Classification BTKD Loss
# ============================================================

def classification_btkd_loss(
    logits_student, logits_teacher,
    gt_labels, gt_boxes, pred_boxes_teacher,
    iou_th=0.5, T=4.0
):
    """
    logits: [N, C]
    gt_labels: [N]
    """

    mask_easy, mask_hard, iou = check_det(
        logits_teacher, pred_boxes_teacher, gt_labels, gt_boxes, iou_th
    )

    # easy 部分
    stu_easy = logits_student[mask_easy]
    tea_easy = logits_teacher[mask_easy]

    if stu_easy.shape[0] > 0:
        loss_easy = kl_loss(stu_easy / T, tea_easy / T)
    else:
        loss_easy = torch.tensor(0.0, device=logits_student.device)

    # hard 部分
    stu_hard = logits_student[mask_hard]
    tea_hard = logits_teacher[mask_hard]

    if stu_hard.shape[0] > 0:
        # rectify teacher's logits
        tea_rectified = rectify_logits_teacher(tea_hard, gt_labels[mask_hard])

        loss_hard = kl_loss(stu_hard / T, tea_rectified / T)
    else:
        loss_hard = torch.tensor(0.0, device=logits_student.device)

    return loss_easy, loss_hard, mask_easy, mask_hard


# ============================================================
#  Regression BTKD Loss (bbox)
# ============================================================

def regression_btkd_loss(
    bbox_s, bbox_t,
    bbox_gt,              # [N, 4]
    mask_easy, mask_hard,
    num_bins=128,
    t_ranges=[(-1.5, 1.5), (-1.5, 1.5), (-2.0, 2.0), (-2.0, 2.0)],
    sigma=2.0
):
    """
    bbox_s: student predicted regression [N, 4]
    bbox_t: teacher predicted regression [N, 4]
    bbox_gt: matched gt regression [N, 4]
    """

    # easy -----------------------------------------------------------------
    if mask_easy.sum() > 0:
        easy_s = bbox_s[mask_easy]
        easy_t = bbox_t[mask_easy]
        easy_gt = bbox_gt[mask_easy]

        loss_easy = 0.0
        for k in range(4):
            tmin, tmax = t_ranges[k]

            # teacher / student distributions
            dist_t = discretize_bbox_targets(easy_t[:, k], tmin, tmax, num_bins, sigma)
            dist_s = discretize_bbox_targets(easy_s[:, k], tmin, tmax, num_bins, sigma)
            dist_gt = F.one_hot(
                torch.clamp(((easy_gt[:, k] - tmin) / (tmax - tmin)
                * num_bins).long(), 0, num_bins - 1),
                num_bins
            ).float()

            loss_easy += kl_loss(dist_s, dist_t)

    else:
        loss_easy = torch.tensor(0.0, device=bbox_s.device)

    # hard -----------------------------------------------------------------
    if mask_hard.sum() > 0:
        hard_s = bbox_s[mask_hard]
        hard_t = bbox_t[mask_hard]
        hard_gt = bbox_gt[mask_hard]

        loss_hard = 0.0
        for k in range(4):
            tmin, tmax = t_ranges[k]

            dist_t = discretize_bbox_targets(hard_t[:, k], tmin, tmax, num_bins, sigma)
            dist_s = discretize_bbox_targets(hard_s[:, k], tmin, tmax, num_bins, sigma)

            # GT onehot
            dist_gt = F.one_hot(
                torch.clamp(((hard_gt[:, k] - tmin) / (tmax - tmin)
                * num_bins).long(), 0, num_bins - 1),
                num_bins
            ).float()

            # rectified distribution
            dist_t_rect = rectify_bbox_distribution(dist_t, dist_gt)

            loss_hard += kl_loss(dist_s, dist_t_rect)

    else:
        loss_hard = torch.tensor(0.0, device=bbox_s.device)

    return loss_easy, loss_hard


# ============================================================
#  BTKD++ 总损失（分类 + 回归）
# ============================================================

def btkd_loss(
    cls_s, cls_t,
    box_s, box_t,
    box_gt,
    gt_labels, gt_boxes,
    iou_th=0.5,
    T=4.0,
    gamma=0.5,
    lambda_stu=0.5,
    ce_loss_fn=nn.CrossEntropyLoss()
):
    """
    综合 BTKD++ 检测损失
    """

    #  分类损失（easy/hard）
    loss_cls_easy, loss_cls_hard, mask_easy, mask_hard = classification_btkd_loss(
        cls_s, cls_t, gt_labels, gt_boxes, box_t, iou_th, T
    )

    # 学生 CE
    loss_ce = ce_loss_fn(cls_s, gt_labels)

    #  回归损失（easy/hard）
    loss_reg_easy, loss_reg_hard = regression_btkd_loss(
        box_s, box_t, box_gt, mask_easy, mask_hard
    )

    #  组合 loss
    loss_kd_easy = loss_cls_easy + loss_reg_easy
    loss_kd_hard = loss_cls_hard + loss_reg_hard

    loss_kd = (1 - gamma) * loss_kd_easy + gamma * loss_kd_hard

    loss_total = lambda_stu * loss_ce + (1 - lambda_stu) * (T * T) * loss_kd

    return loss_total, {
        "ce": loss_ce.item(),
        "cls_easy": loss_cls_easy.item(),
        "cls_hard": loss_cls_hard.item(),
        "reg_easy": loss_reg_easy.item(),
        "reg_hard": loss_reg_hard.item(),
        "kd_easy": loss_kd_easy.item(),
        "kd_hard": loss_kd_hard.item(),
    }
