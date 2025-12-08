# ------------------------------------------------------------
# CIRKD + BTKD++ + Intra/Inter + FG/BG Distillation
# ------------------------------------------------------------

import torch
import torch.nn.functional as F
import math


# ============================================================
#  Embedding reshape
# ============================================================

def reshape_embedding(feat):
    """
    feat: [B, C, H, W]
    return: [B, HW, C]
    """
    B, C, H, W = feat.shape
    feat = feat.view(B, C, -1).transpose(1, 2)  # [B, HW, C]
    return feat


# ============================================================
#  Easy / Hard mask & Teacher Calibration (BTKD++)
# ============================================================

def compute_easy_hard_mask(logits_teacher, labels):
    """
    logits_teacher: [B, C, H, W]
    labels:         [B, H, W]
    """
    pred = torch.argmax(logits_teacher, dim=1)
    easy = (pred == labels)
    hard = ~easy
    return easy, hard


def rectify_teacher_logits(logits_teacher, labels, hard_mask):
    """
    BTKD-style wrong knowledge correction:
    swap teacher wrong-class logits with GT-class logits only on hard pixels.
    """
    lt = logits_teacher.clone()  # [B, C, H, W]
    B, C, H, W = lt.shape

    lt = lt.permute(0, 2, 3, 1)  # → [B, H, W, C]

    for b in range(B):
        mask = hard_mask[b]  # [H, W]
        if mask.sum() == 0:
            continue

        wrong_logits = lt[b][mask]  # [K, C]
        gt_cls = labels[b][mask]    # [K]
        pred_cls = torch.argmax(wrong_logits, dim=1)

        for i in range(wrong_logits.shape[0]):
            y = pred_cls[i].item()  # wrong predicted class
            t = gt_cls[i].item()    # true class
            # swap
            wrong_logits[i][y], wrong_logits[i][t] = wrong_logits[i][t].clone(), wrong_logits[i][y].clone()

        lt[b][mask] = wrong_logits

    return lt.permute(0, 3, 1, 2).contiguous()  # back to [B, C, H, W]


# ============================================================
#  Foreground / Background Split
# ============================================================

def get_fg_bg_mask(labels, bg_class=0):
    """
    Cityscapes: background = 0 (road, etc.)
    labels: [B, H, W]
    """
    fg = labels != bg_class
    bg = labels == bg_class
    return fg, bg


# ============================================================
#  CIRKD mini-batch pixel-to-pixel (FG only)
# ============================================================

def pixel_relation_loss(feat_s, feat_t, tau=0.5):
    """
    feat_s: [B, HW, C]
    feat_t: [B, HW, C]
    """
    B, N, C = feat_s.shape

    # normalize
    fs = F.normalize(feat_s, dim=-1)
    ft = F.normalize(feat_t, dim=-1)

    # relation matrix
    Rs = torch.bmm(fs, fs.transpose(1, 2)) / tau
    Rt = torch.bmm(ft, ft.transpose(1, 2)) / tau

    Rs = F.log_softmax(Rs, dim=-1)
    Rt = F.softmax(Rt, dim=-1)

    return F.kl_div(Rs, Rt, reduction="batchmean")


# ============================================================
#  Pixel & Region Memory Queue (FG only)
# ============================================================

class PixelQueue:
    """Pixel memory bank: queue per class."""
    def __init__(self, num_classes=19, dim=256, queue_len=256):
        self.queue = torch.zeros(num_classes, queue_len, dim)
        self.ptr = torch.zeros(num_classes, dtype=torch.long)
        self.num_classes = num_classes
        self.queue_len = queue_len

    def to(self, device):
        self.queue = self.queue.to(device)
        self.ptr = self.ptr.to(device)
        return self

    def enqueue(self, feats, labels):
        """
        feats: [B, HW, C]
        labels: [B, HW]
        """
        B, N, C = feats.shape
        feats = feats.detach()

        for b in range(B):
            for cls in torch.unique(labels[b]):
                cls = int(cls.item())
                mask = (labels[b] == cls)
                if mask.sum() == 0:
                    continue

                f = feats[b][mask]
                k = min(f.size(0), self.queue_len)

                ptr = self.ptr[cls].item()
                end = min(ptr + k, self.queue_len)
                self.queue[cls, ptr:end] = f[:end - ptr]
                self.ptr[cls] = (ptr + k) % self.queue_len


class RegionQueue:
    """Region-level memory bank (prototype-level)."""
    def __init__(self, num_classes=19, dim=256, queue_len=64):
        self.queue = torch.zeros(num_classes, queue_len, dim)
        self.ptr = torch.zeros(num_classes, dtype=torch.long)
        self.num_classes = num_classes
        self.queue_len = queue_len

    def to(self, device):
        self.queue = self.queue.to(device)
        self.ptr = self.ptr.to(device)
        return self

    def enqueue(self, feats, labels):
        """
        feats: [B, HW, C]
        labels: [B, HW]
        """
        B, N, C = feats.shape
        feats = feats.detach()

        for b in range(B):
            f_b = feats[b]
            y_b = labels[b]

            for cls in torch.unique(y_b):
                cls = int(cls.item())
                mask = (y_b == cls)
                if mask.sum() == 0:
                    continue

                region = f_b[mask].mean(dim=0)
                ptr = self.ptr[cls].item()
                self.queue[cls, ptr] = region
                self.ptr[cls] = (ptr + 1) % self.queue_len


# ============================================================
#  Class Prototypes + Intra/Inter Loss (BTKD segmentation version)
# ============================================================

def compute_class_prototypes(feats, labels, num_classes=19):
    """
    feats: [B, HW, C]
    labels: [B, HW]
    return: [num_classes, C]
    """
    B, N, C = feats.shape
    proto = torch.zeros(num_classes, C, device=feats.device)

    for cls in range(num_classes):
        mask = (labels == cls)  # [B, HW]
        if mask.sum() == 0:
            continue
        proto[cls] = feats[mask].mean(dim=0)

    return proto


def intra_class_loss(feat_s, proto_t, labels):
    """
    Student feature should be close to Teacher prototype of its class.
    """
    B, N, C = feat_s.shape
    feat_flat = feat_s.reshape(-1, C)
    label_flat = labels.reshape(-1)

    proto_expand = proto_t[label_flat]  # [BN, C]

    return F.mse_loss(feat_flat, proto_expand)


def inter_class_loss(proto_t, margin=1.0):
    """
    Increase separation between teacher prototypes.
    proto_t: [num_classes, C]
    """
    num_classes = proto_t.size(0)
    loss = 0.
    count = 0

    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            dist = torch.norm(proto_t[i] - proto_t[j], p=2)
            loss += F.relu(margin - dist)
            count += 1

    if count > 0:
        loss /= count

    return loss


# ============================================================
#  Dynamic gamma (easy->hard)
# ============================================================

def gamma_schedule(epoch, total_epoch):
    """
    early: small gamma → focus easy knowledge
    late: large gamma → focus hard pixels
    """
    t = epoch / total_epoch
    gamma = math.cos(math.pi / 2 * (1 - t)) ** 2
    return gamma
