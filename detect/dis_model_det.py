# ------------------------------------------------------------
# 使用 resnet_fpn_backbone / mobilenetv2_fpn_backbone
# 定义 FasterRCNN 教师 / 学生模型
# ------------------------------------------------------------

import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

# ============================================================
#  ResNet + FPN Backbones（官方实现）
# ============================================================

def build_resnet_fpn(name="resnet50", pretrained=True, trainable_layers=3):
    """
    name: 'resnet18' / 'resnet50' / 'resnet101'
    trainable_layers: number of trainable layers in backbone
    """
    backbone = resnet_fpn_backbone(
        name,
        pretrained=pretrained,
        trainable_layers=trainable_layers
    )
    return backbone


# ============================================================
#  MobileNetV2 + FPN 学生 backbone
# ============================================================

def build_mobilenetv2_fpn(pretrained=True, trainable_layers=6):
    """
    MobileNetV2 + FPN backbone（官方 torchvision 不直接提供，需要自定义）
    """
    # 加载 MobileNetV2
    mobilenet = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)

    # 将 features 部分取出
    backbone = mobilenet.features
    backbone.out_channels = 1280  # MobileNetV2 的最后一层通道数

    #  构建 FPN
    from torchvision.ops import FeaturePyramidNetwork

    # 输入特征来自 MobileNet 的不同 stage
    return_layers = {
        "2": 0,   # 1/4
        "4": 1,   # 1/8
        "7": 2,   # 1/16
        "14": 3,  # 1/32
    }

    from torchvision.models._utils import IntermediateLayerGetter
    body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    in_channels_list = [24, 32, 96, 1280]  # MobileNetV2 对应通道数
    out_channels = 256

    fpn = FeaturePyramidNetwork(
        in_channels_list=in_channels_list,
        out_channels=out_channels
    )

    # 返回 backbone + fpn
    class BackboneWithFPN(torch.nn.Module):
        def __init__(self, body, fpn):
            super().__init__()
            self.body = body
            self.fpn = fpn
            self.out_channels = out_channels

        def forward(self, x):
            feats = self.body(x)
            fpn_feats = self.fpn(feats)
            return fpn_feats

    return BackboneWithFPN(body, fpn)


# ============================================================
#  教师 / 学生 FasterRCNN 模型构建
# ============================================================

def build_fasterrcnn_resnet101(num_classes=91):
    backbone = build_resnet_fpn("resnet101", pretrained=True)
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model


def build_fasterrcnn_resnet50(num_classes=91):
    backbone = build_resnet_fpn("resnet50", pretrained=True)
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model


def build_fasterrcnn_resnet18(num_classes=91):
    backbone = build_resnet_fpn("resnet18", pretrained=True)
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model


def build_fasterrcnn_mobilenetv2(num_classes=91):
    backbone = build_mobilenetv2_fpn(pretrained=True)
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model


# # ============================================================
# # test
# # ============================================================
#
# if __name__ == "__main__":
#     model_T = build_fasterrcnn_resnet101()
#     model_S = build_fasterrcnn_resnet18()
#
#     x = [torch.randn(3, 224, 224)]
#     print("Teacher forward OK:", model_T(x))
#     print("Student forward OK:", model_S(x))
