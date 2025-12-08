# ------------------------------------------------------------
# DeepLabV3 + SegFormer Teacher/Student Wrapper for BTKD++ + CIRKD
# ------------------------------------------------------------

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from functools import partial

# ===== Import your SegFormer (MiT) code =====
from segformer import get_segformer


# ============================================================
#  Load pretrained weights safely
# ============================================================

def safe_load(model, pretrained_path):
    if pretrained_path is None:
        print("[Warning] pretrained_path = None, skip loading.")
        return model

    if os.path.exists(pretrained_path):
        print(f"[Load] Pretrained weights: {pretrained_path}")
        state = torch.load(pretrained_path, map_location="cpu")
        try:
            model.load_state_dict(state, strict=False)
        except:
            # For SegFormer, use init_weights()
            print("[Info] Attempting SegFormer init_weights loading...")
            model.init_weights(pretrained_path)
    else:
        print(f"[Warning] Pretrained weight NOT found: {pretrained_path}")
    return model



# ============================================================
# DeepLabV3 Wrappers
# ============================================================

class DeepLabV3Wrapper(nn.Module):
    """
    Unified wrapper:
    output:
        logits:     [B, C, H, W]
        embedding:  pixel embedding from backbone last feature (for CIRKD)
    """

    def __init__(self, backbone="resnet101", num_classes=19, pretrained_path=None):
        super().__init__()

        if backbone == "resnet101":
            self.model = torchvision.models.segmentation.deeplabv3_resnet101(
                weights=None, num_classes=num_classes
            )
        elif backbone == "resnet18":
            self.model = torchvision.models.segmentation.deeplabv3_resnet50(
                weights=None, num_classes=num_classes
            )
        elif backbone == "mobilenetv2":
            self.model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_small(
                weights=None, num_classes=num_classes
            )
        else:
            raise ValueError("Unsupported DeepLab backbone")

        # load pretrained weights
        safe_load(self.model, pretrained_path)

        # extract the DeepLab backbone
        self.backbone = self.model.backbone
        self.classifier = self.model.classifier

    def forward(self, x):
        """
        Returns:
            logits: (B, C, H, W)
            embedding: (B, D, H', W')
        """
        features = self.backbone(x)              # dict with 'out'
        embedding = features["out"]              # pixel embedding (ASPP 输入)
        logits = self.classifier(embedding)

        return logits, embedding



# ============================================================
# SegFormer Wrappers (MiT backbone)
# ============================================================

class SegFormerWrapper(nn.Module):
    """
    Wrapper for SegFormer (MiT-B0..B5)
    Uses get_segformer() from segformer.py
    """

    def __init__(self, backbone="MiT_B4", num_classes=19, pretrained_path=None):
        super().__init__()

        self.model = get_segformer(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=pretrained_path
        )

    def forward(self, x):
        logits, embedding = self.model(x)
        return logits, embedding

    def forward(self, x):
        """
        SegFormer forward returns:
            [logits, decoder_feature]
        """
        pred = self.model(x)
        logits = pred[0]          # segmentation logits
        embedding = pred[1]       # pixel embedding (decoder fused features)

        return logits, embedding



# ============================================================
# Factory functions (same style as detection/classification)
# ============================================================

def build_deeplabv3_r101(num_classes=19):
    return DeepLabV3Wrapper(
        backbone="resnet101",
        num_classes=num_classes,
        pretrained_path="./pth/deeplabv3_resnet101.pth"
    )


def build_deeplabv3_r18(num_classes=19):
    return DeepLabV3Wrapper(
        backbone="resnet18",
        num_classes=num_classes,
        pretrained_path="./pth/deeplabv3_resnet18.pth"
    )


def build_deeplabv3_mbv2(num_classes=19):
    return DeepLabV3Wrapper(
        backbone="mobilenetv2",
        num_classes=num_classes,
        pretrained_path="./pth/deeplabv3_mbv2.pth"
    )


def build_segformer_b4(num_classes=19):
    return SegFormerWrapper(
        backbone="MiT_B4",
        num_classes=num_classes,
        pretrained_path="./pth/mit_b4.pth"
    )


def build_segformer_b0(num_classes=19):
    return SegFormerWrapper(
        backbone="MiT_B0",
        num_classes=num_classes,
        pretrained_path="./pth/mit_b0.pth"
    )


# ============================================================
# Quick Test
# ============================================================

if __name__ == "__main__":
    x = torch.randn(1, 3, 769, 769)

    model = build_deeplabv3_r101()
    logits, emb = model(x)
    print("DeepLab logits:", logits.shape, "embedding:", emb.shape)

    x2 = torch.randn(1, 3, 1024, 1024)
    seg_m = build_segformer_b4()
    logits2, emb2 = seg_m(x2)
    print("SegFormer logits:", logits2.shape, "embedding:", emb2.shape)
