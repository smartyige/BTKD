# ------------------------------------------------------------
# btkd_seg_loss.py
# Fusion Loss for BTKD++ + CIRKD + Intra/Inter + FG/BG
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from btkd_seg_utils import (
    reshape_embedding,
    compute_easy_hard_mask,
    rectify_teacher_logits,
    get_fg_bg_mask,
    pixel_relation_loss,
    PixelQueue,
    RegionQueue,
    compute_class_prototypes,
    intra_class_loss,
    inter_class_loss,
    gamma_schedule
)


class BTKD_SegLoss(nn.Module):
    """
    Total segmentation KD loss:
        CE
      + BTKD++ logits loss (easy-hard with teacher correction)
      + CIRKD relation losses (batch, pixel memory, region memory)
      + Intra/Inter class feature constraints
      + FG/BG split strategy
    """

    def __init__(self,
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
                 lambda_inter=1.0,
                 margin_inter=1.0):

        super().__init__()

        self.tau = tau

        self.lambda_btkd = lambda_btkd
        self.lambda_batch = lambda_batch
        self.lambda_mem_pixel = lambda_mem_pixel
        self.lambda_mem_region = lambda_mem_region
        self.lambda_intra = lambda_intra
        self.lambda_inter = lambda_inter
        self.margin_inter = margin_inter

        # CE loss for segmentation
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)

        # queues
        self.pixel_queue_t = PixelQueue(num_classes, dim=pixel_dim, queue_len=queue_pixel_len)
        self.pixel_queue_s = PixelQueue(num_classes, dim=pixel_dim, queue_len=queue_pixel_len)

        self.region_queue_t = RegionQueue(num_classes, dim=pixel_dim, queue_len=queue_region_len)
        self.region_queue_s = RegionQueue(num_classes, dim=pixel_dim, queue_len=queue_region_len)

    # --------------------------------------------------------
    # Register device
    # --------------------------------------------------------
    def to(self, device):
        super().to(device)
        self.pixel_queue_t.to(device)
        self.pixel_queue_s.to(device)
        self.region_queue_t.to(device)
        self.region_queue_s.to(device)
        return self

    # --------------------------------------------------------
    # Main Loss Function
    # --------------------------------------------------------
    def forward(self,
                logits_s,
                logits_t,
                feat_s,
                feat_t,
                labels,
                epoch,
                total_epoch):

        """
        logits_s: [B, C, H, W]
        logits_t: [B, C, H, W]
        feat_s:   [B, C', H', W']
        feat_t:   [B, C', H', W']
        labels:   [B, H, W]
        """

        B = logits_s.size(0)

        # ======================
        # 1. Task CE loss
        # ======================
        loss_ce = self.ce_loss(logits_s, labels)

        # ======================
        # 2. Easy/Hard Mask & BTKD Correction
        # ======================
        easy_mask, hard_mask = compute_easy_hard_mask(logits_t, labels)

        # teacher logits corrected (only hard pixels)
        logits_t_corr = rectify_teacher_logits(logits_t, labels, hard_mask)

        # KD loss
        kd_easy = F.kl_div(
            F.log_softmax(logits_s / 4.0, dim=1),
            F.softmax(logits_t / 4.0, dim=1),
            reduction='batchmean'
        )

        kd_hard = F.kl_div(
            F.log_softmax(logits_s / 4.0, dim=1),
            F.softmax(logits_t_corr / 4.0, dim=1),
            reduction='batchmean'
        )

        gamma = gamma_schedule(epoch, total_epoch)
        loss_btkd = (1 - gamma) * kd_easy + gamma * kd_hard

        # ======================
        # 3. Foreground / Background
        # ======================
        fg_mask, bg_mask = get_fg_bg_mask(labels)

        # reshape features
        fs = reshape_embedding(feat_s)   # [B, HW, C]
        ft = reshape_embedding(feat_t)

        label_flat = labels.view(B, -1)  # [B, HW]

        fs_fg = []
        ft_fg = []
        label_fg = []

        for b in range(B):
            mask = fg_mask[b].view(-1)  # flatten
            fs_fg.append(fs[b][mask])
            ft_fg.append(ft[b][mask])
            label_fg.append(label_flat[b][mask])

        # pack FG samples
        fs_fg = torch.cat(fs_fg, dim=0)
        ft_fg = torch.cat(ft_fg, dim=0)
        label_fg = torch.cat(label_fg, dim=0)

        # if no FG pixels, skip relation KD
        if fs_fg.size(0) == 0:
            loss_p2p_batch = 0.0
            loss_p2p_mem = 0.0
            loss_region = 0.0
            loss_intra = 0.0
            loss_inter = 0.0
        else:
            # ======================
            # 4. CIRKD Mini-batch Pixel Relation KD (FG only)
            # ======================
            # we need to reshape back to [B, HW_fg, C]
            # but for simplicity we simulate batchmean by dividing
            num_fg = fs_fg.size(0)
            p2p_batch = pixel_relation_loss(fs_fg.unsqueeze(0), ft_fg.unsqueeze(0), tau=self.tau)
            loss_p2p_batch = p2p_batch * (num_fg / (B * fs.size(1)))  # scale to original batch size

            # ======================
            # 5. Memory Pixel KD (FG only)
            # ======================
            # enqueue teacher & student FG pixels into queue
            self.pixel_queue_t.enqueue(ft.unsqueeze(0), label_flat.unsqueeze(0))
            self.pixel_queue_s.enqueue(fs.unsqueeze(0), label_flat.unsqueeze(0))

            # get teacher memory as target
            loss_p2p_mem = 0.
            for cls in torch.unique(label_fg):
                cls = int(cls.item())
                mem_t = self.pixel_queue_t.queue[cls]  # [Q, C]
                fs_cls = fs_fg[label_fg == cls]        # [K, C]
                if fs_cls.size(0) == 0:
                    continue
                # contrastive distillation
                sim_s = F.log_softmax(torch.mm(fs_cls, mem_t.t()) / self.tau, dim=-1)
                sim_t = F.softmax(torch.mm(ft_fg[label_fg == cls], mem_t.t()) / self.tau, dim=-1)
                loss_p2p_mem += F.kl_div(sim_s, sim_t, reduction='batchmean')

            # ======================
            # 6. Region Memory KD (FG only)
            # ======================
            # enqueue region prototypes
            self.region_queue_t.enqueue(ft.unsqueeze(0), label_flat.unsqueeze(0))
            self.region_queue_s.enqueue(fs.unsqueeze(0), label_flat.unsqueeze(0))

            loss_region = 0.
            for cls in torch.unique(label_fg):
                cls = int(cls.item())
                mem_t = self.region_queue_t.queue[cls]  # [R, C]
                mem_s = self.region_queue_s.queue[cls]  # [R, C]

                sim_s = F.log_softmax(torch.mm(mem_s, mem_t.t()) / self.tau, dim=-1)
                sim_t = F.softmax(torch.mm(mem_t, mem_t.t()) / self.tau, dim=-1)

                loss_region += F.kl_div(sim_s, sim_t, reduction='batchmean')

            # ======================
            # 7. Intra/Inter Class Feature Constraints (FG only)
            # ======================
            proto_t = compute_class_prototypes(ft.unsqueeze(0), labels.unsqueeze(0))

            loss_intra = intra_class_loss(fs.unsqueeze(0), proto_t, labels.unsqueeze(0))
            loss_inter = inter_class_loss(proto_t, margin=self.margin_inter)

        # ======================================================
        # Total Loss
        # ======================================================
        loss_total = loss_ce \
                     + self.lambda_btkd * loss_btkd \
                     + self.lambda_batch * loss_p2p_batch \
                     + self.lambda_mem_pixel  * loss_p2p_mem \
                     + self.lambda_mem_region * loss_region \
                     + self.lambda_intra      * loss_intra \
                     + self.lambda_inter      * loss_inter

        return loss_total, {
            "ce": loss_ce.item(),
            "btkd": loss_btkd.item(),
            "p2p_batch": float(loss_p2p_batch),
            "p2p_mem": float(loss_p2p_mem),
            "region": float(loss_region),
            "intra": float(loss_intra),
            "inter": float(loss_inter),
            "gamma": gamma
        }
