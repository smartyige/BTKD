# ------------------------------------------------------------
# YOLOv8-m → YOLOv8-n 的 BTKD++ 教师/学生模型定义
# 使用 raw outputs（model.model(img)）进行蒸馏
# ------------------------------------------------------------

import torch
import torch.nn as nn
from ultralytics import YOLO


# ============================================================
# YOLOv8 Raw Output Wrapper
# ============================================================

class YOLOv8Raw(nn.Module):
    """
    包装 YOLOv8，使其能够返回 raw output：
    pred: [B, N, 85] = [x, y, w, h, obj, class80]
    """

    def __init__(self, model_path, device="cuda"):
        super().__init__()
        self.model = YOLO(model_path).model   # 加载 raw model
        self.device = device
        self.to(device)

    def forward(self, images):
        """
        input:
            images: Tensor [B, 3, H, W]

        output:
            pred_bbox: [B, N, 4]
            pred_obj:  [B, N, 1]
            pred_cls:  [B, N, 80]
        """

        preds = self.model(images)[0]      # raw output from head
        # preds shape: [B, N, 85]

        pred_bbox = preds[..., :4]         # x,y,w,h
        pred_obj = preds[..., 4:5]         # objectness
        pred_cls = preds[..., 5:]          # class predictions

        return pred_bbox, pred_obj, pred_cls


# ============================================================
# Teacher & Student constructors
# ============================================================

def build_yolo_teacher(model_size="m", device="cuda"):
    """
    Teacher YOLOv8-m
    """
    model_path = f"yolov8{model_size}.pt"
    model = YOLOv8Raw(model_path, device=device)
    model.eval()
    return model


def build_yolo_student(model_size="n", device="cuda"):
    """
    Student YOLOv8-n
    """
    model_path = f"yolov8{model_size}.pt"
    model = YOLOv8Raw(model_path, device=device)
    return model


# # ============================================================
# # Quick Test
# # ============================================================
#
# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#
#     teacher = build_yolo_teacher("m", device)
#     student = build_yolo_student("n", device)
#
#     x = torch.randn(1, 3, 640, 640).to(device)
#
#     bbox_t, obj_t, cls_t = teacher(x)
#     bbox_s, obj_s, cls_s = student(x)
#
#     print("Teacher bbox:", bbox_t.shape)
#     print("Student bbox:", bbox_s.shape)
