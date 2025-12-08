# ------------------------------------------------------------
# YOLOv8 BTKD++ 辅助函数
# ------------------------------------------------------------

import torch
import torch.nn.functional as F


# ============================================================
#  IoU 计算（与 FasterRCNN 一致）
# ============================================================

def box_iou(boxes1, boxes2):
    """
    boxes1: [N, 4]
    boxes2: [M, 4]
    output: [N, M]
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(0) * \
            (boxes1[:, 3] - boxes1[:, 1]).clamp(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(0) * \
            (boxes2[:, 3] - boxes2[:, 1]).clamp(0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])    # top-left
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])    # bottom-right

    wh = (rb - lt).clamp(0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2 - inter

    return inter / (union + 1e-7)


# ============================================================
#  YOLO → xyxy 格式转换
# ============================================================

def xywh_to_xyxy(box):
    """
    box: (..., 4) in YOLO (x_center, y_center, w, h)
    """
    x, y, w, h = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


# ============================================================
#  匹配 GT（正样本提取）
# ============================================================

def match_predictions_with_gt(pred_boxes_xywh, pred_cls, pred_obj, targets, iou_th=0.5):
    """
    pred_boxes_xywh: [N, 4]
    pred_cls: [N, 80]
    pred_obj: [N, 1]
    targets: dict {'boxes':[M,4], 'labels':[M]}
    返回：
        match_idx: 每个 GT 匹配到的预测 index
    """

    device = pred_boxes_xywh.device

    gt_boxes = targets["boxes"].to(device)
    gt_labels = targets["labels"].to(device)

    # Convert YOLO xywh → xyxy
    pred_boxes = xywh_to_xyxy(pred_boxes_xywh)

    # 计算 IoU
    ious = box_iou(pred_boxes, gt_boxes)   # [N, M]

    # 为每个 GT 找到 IoU 最大的预测
    match_idx = torch.argmax(ious, dim=0)  # [M]

    matched_preds_boxes = pred_boxes[match_idx]
    matched_preds_cls = pred_cls[match_idx]
    matched_preds_obj = pred_obj[match_idx]

    iou_vals = torch.max(ious, dim=0).values  # [M]

    return matched_preds_boxes, matched_preds_cls, matched_preds_obj, gt_boxes, gt_labels, iou_vals


# ============================================================
#  Easy / Hard 判定
# ============================================================

def check_yolo(teacher_cls, teacher_boxes, gt_labels, gt_boxes, iou_vals, iou_th=0.5):
    """
    teacher_cls: [M, 80]
    teacher_boxes: [M, 4]
    gt_labels: [M]
    iou_vals: [M] IoU(pred, gt)
    """

    # teacher 分类预测（取 argmax）
    t_pred = torch.argmax(teacher_cls, dim=1)

    cls_correct = (t_pred == gt_labels)
    box_ok = (iou_vals >= iou_th)

    mask_easy = cls_correct & box_ok
    mask_hard = ~mask_easy

    return mask_easy, mask_hard


# ============================================================
#  回归离散化（与 FasterRCNN 一致）
# ============================================================

def discretize_bbox(val, t_min, t_max, num_bins=128, sigma=2.0):
    """
    单 scalar → 概率分布
    """
    val_norm = (val - t_min) / (t_max - t_min)
    val_norm = val_norm.clamp(0, 1)

    centers = torch.linspace(0, 1, num_bins, device=val.device)
    dist = torch.exp(-(centers - val_norm.unsqueeze(1)) ** 2 / (2 * sigma ** 2))
    dist = dist / (dist.sum(dim=1, keepdim=True) + 1e-7)

    return dist


# ============================================================
#  分类 rectification（比例校正）
# ============================================================

def rectify_cls_logits(teacher_cls, gt_labels):
    """
    与 calibrate3 类似：交换 + 缩放
    """
    tea = teacher_cls.clone()
    pred_wrong = torch.argmax(tea, dim=1)

    for i in range(len(gt_labels)):
        gt = gt_labels[i]
        wrong = pred_wrong[i]

        if wrong != gt:
            s_old = tea[i, gt] + tea[i, wrong]
            tea[i, gt] = (tea[i, gt] + 1) / 2
            tea[i, wrong] = (tea[i, wrong] + 0) / 2
            s_new = tea[i, gt] + tea[i, wrong] + 1e-7
            tea[i, gt] *= (s_old / s_new)
            tea[i, wrong] *= (s_old / s_new)

    return tea


# ============================================================
#  objectness rectification
# ============================================================

def rectify_obj_logits(teacher_obj, mask_hard):
    """
    Hard-task：提高 teacher 的 obj，让学生更关注困难目标
    """
    tea_obj = teacher_obj.clone()
    tea_obj[mask_hard] = (tea_obj[mask_hard] + 1.0) / 2.0
    return tea_obj


# ============================================================
# ============================================================

print("[BTKD-YOLO] Utils loaded successfully.")
