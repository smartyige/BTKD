# ------------------------------------------------------------
# MiT Backbones (B0-B5) + SegFormer Decoder
# ------------------------------------------------------------

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model


# ============================================================
# Basic Blocks
# ============================================================

class DWConv(nn.Module):
    """Depthwise Conv for MLP"""
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    """MLP block with DWConv support."""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding (with overlap)."""
    def __init__(self, img_size, patch_size=7, stride=4, in_chans=3, embed_dim=64):
        super().__init__()

        patch_size = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2)
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)   # B, HW, C
        x = self.norm(x)
        return x, H, W


class Attention(nn.Module):
    """Multi-head Self-Attention with Spatial Reduction."""
    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()

        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads

        self.scale = (dim // num_heads) ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # spatial reduction
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)

    def forward(self, x, H, W):
        B, N, C = x.shape

        # Query
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads)
        q = q.permute(0, 2, 1, 3)

        # Key/Value
        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).transpose(1, 2)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2,
                                      self.num_heads, C // self.num_heads)
            kv = kv.permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, N, 2,
                                    self.num_heads, C // self.num_heads)
            kv = kv.permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]  # key, value

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Output projection
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Transformer Block: LN → Attention → LN → MLP"""
    def __init__(self, dim, num_heads, mlp_ratio=4.,
                 qkv_bias=False, drop_path=0., sr_ratio=1):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
            sr_ratio=sr_ratio
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


# ============================================================
# MiT Backbone (B0–B5)
# ============================================================

class MiT_Backbone(nn.Module):
    """Backbone used in SegFormer: MiT (Mix Transformer)."""

    def __init__(self, embed_dims, depths, num_heads, mlp_ratios, sr_ratios,
                 drop_rate=0., drop_path_rate=0.):
        super().__init__()

        # stochastic depth schedule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.patch_embed1 = OverlapPatchEmbed(img_size=(1024, 1024),
                                              patch_size=7, stride=4,
                                              in_chans=3, embed_dim=embed_dims[0])
        self.block1 = nn.ModuleList([
            Block(dim=embed_dims[0], num_heads=num_heads[0],
                  mlp_ratio=mlp_ratios[0], sr_ratio=sr_ratios[0],
                  drop_path=dpr[cur+i])
            for i in range(depths[0])
        ])
        self.norm1 = nn.LayerNorm(embed_dims[0])
        cur += depths[0]

        self.patch_embed2 = OverlapPatchEmbed(img_size=(256, 256),
                                              patch_size=3, stride=2,
                                              in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.block2 = nn.ModuleList([
            Block(dim=embed_dims[1], num_heads=num_heads[1],
                  mlp_ratio=mlp_ratios[1], sr_ratio=sr_ratios[1],
                  drop_path=dpr[cur+i])
            for i in range(depths[1])
        ])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        cur += depths[1]

        self.patch_embed3 = OverlapPatchEmbed(img_size=(128, 128),
                                              patch_size=3, stride=2,
                                              in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.block3 = nn.ModuleList([
            Block(dim=embed_dims[2], num_heads=num_heads[2],
                  mlp_ratio=mlp_ratios[2], sr_ratio=sr_ratios[2],
                  drop_path=dpr[cur+i])
            for i in range(depths[2])
        ])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        cur += depths[2]

        self.patch_embed4 = OverlapPatchEmbed(img_size=(64, 64),
                                              patch_size=3, stride=2,
                                              in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])
        self.block4 = nn.ModuleList([
            Block(dim=embed_dims[3], num_heads=num_heads[3],
                  mlp_ratio=mlp_ratios[3], sr_ratio=sr_ratios[3],
                  drop_path=dpr[cur+i])
            for i in range(depths[3])
        ])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        outs.append(x.transpose(1, 2).reshape(B, -1, H, W))

        # stage 2
        x, H, W = self.patch_embed2(outs[-1])
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        outs.append(x.transpose(1, 2).reshape(B, -1, H, W))

        # stage 3
        x, H, W = self.patch_embed3(outs[-1])
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        outs.append(x.transpose(1, 2).reshape(B, -1, H, W))

        # stage 4
        x, H, W = self.patch_embed4(outs[-1])
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        outs.append(x.transpose(1, 2).reshape(B, -1, H, W))

        return outs  # list: C1, C2, C3, C4


# ============================================================
# SegFormer Decoder
# ============================================================

class SegFormerDecoder(nn.Module):
    def __init__(self, embed_dims, decoder_dim=256, num_classes=19):
        super().__init__()

        self.linear_c4 = nn.Linear(embed_dims[3], decoder_dim)
        self.linear_c3 = nn.Linear(embed_dims[2], decoder_dim)
        self.linear_c2 = nn.Linear(embed_dims[1], decoder_dim)
        self.linear_c1 = nn.Linear(embed_dims[0], decoder_dim)

        self.fuse = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, kernel_size=1),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True)
        )

        self.dropout = nn.Dropout2d(0.1)
        self.predict = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(self, feats):
        """
        feats: list of [C1, C2, C3, C4]
        """
        c1, c2, c3, c4 = feats
        B = c1.shape[0]

        H, W = c1.shape[2], c1.shape[3]

        # flatten + linear
        n4 = c4.flatten(2).transpose(1, 2)
        n4 = self.linear_c4(n4).transpose(1, 2).reshape(B, -1, c4.shape[2], c4.shape[3])
        n4 = F.interpolate(n4, size=(H, W), mode="bilinear", align_corners=False)

        n3 = c3.flatten(2).transpose(1, 2)
        n3 = self.linear_c3(n3).transpose(1, 2).reshape(B, -1, c3.shape[2], c3.shape[3])
        n3 = F.interpolate(n3, size=(H, W), mode="bilinear", align_corners=False)

        n2 = c2.flatten(2).transpose(1, 2)
        n2 = self.linear_c2(n2).transpose(1, 2).reshape(B, -1, c2.shape[2], c2.shape[3])
        n2 = F.interpolate(n2, size=(H, W), mode="bilinear", align_corners=False)

        n1 = c1.flatten(2).transpose(1, 2)
        n1 = self.linear_c1(n1).transpose(1, 2).reshape(B, -1, c1.shape[2], c1.shape[3])

        fused = self.fuse(torch.cat([n4, n3, n2, n1], dim=1))
        fused = self.dropout(fused)

        logits = self.predict(fused)
        return logits, fused  # fused is pixel embedding


# ============================================================
# SegFormer Model
# ============================================================

class SegFormer(nn.Module):
    def __init__(self, backbone_cfg, num_classes=19, pretrained=None):
        super().__init__()

        embed_dims, depths, num_heads, mlp_ratios, sr_ratios, decoder_dim = backbone_cfg

        self.backbone = MiT_Backbone(embed_dims, depths, num_heads,
                                     mlp_ratios, sr_ratios)

        self.decoder = SegFormerDecoder(embed_dims, decoder_dim, num_classes)

        if pretrained is not None and os.path.exists(pretrained):
            print(f"[Load] SegFormer/MiT pretrained: {pretrained}")
            state = torch.load(pretrained, map_location="cpu")
            self.load_state_dict(state, strict=False)

    def forward(self, x):
        feats = self.backbone(x)          # 4 stage features
        logits, embedding = self.decoder(feats)
        return [logits, embedding]


# ============================================================
# Backbones (MiT Configurations)
# ============================================================

MiT_CONFIGS = {
    "MiT_B0": (
        [32, 64, 160, 256],
        [2, 2, 2, 2],
        [1, 2, 5, 8],
        [4, 4, 4, 4],
        [8, 4, 2, 1],
        256
    ),

    "MiT_B1": (
        [64, 128, 320, 512],
        [2, 2, 2, 2],
        [1, 2, 5, 8],
        [4, 4, 4, 4],
        [8, 4, 2, 1],
        256
    ),

    "MiT_B2": (
        [64, 128, 320, 512],
        [3, 4, 6, 3],
        [1, 2, 5, 8],
        [4, 4, 4, 4],
        [8, 4, 2, 1],
        768
    ),

    "MiT_B3": (
        [64, 128, 320, 512],
        [3, 4, 18, 3],
        [1, 2, 5, 8],
        [4, 4, 4, 4],
        [8, 4, 2, 1],
        768
    ),

    "MiT_B4": (
        [64, 128, 320, 512],
        [3, 8, 27, 3],
        [1, 2, 5, 8],
        [4, 4, 4, 4],
        [8, 4, 2, 1],
        768
    )
}


# ============================================================
# Factory
# ============================================================

def get_segformer(backbone="MiT_B1", num_classes=19, pretrained=None):
    if backbone not in MiT_CONFIGS:
        raise ValueError(f"Unknown backbone name: {backbone}")

    cfg = MiT_CONFIGS[backbone]
    return SegFormer(cfg, num_classes=num_classes, pretrained=pretrained)


# # ============================================================
# # Quick Test
# # ============================================================
#
# if __name__ == "__main__":
#     model = get_segformer("MiT_B1", num_classes=19)
#     x = torch.randn(1, 3, 1024, 1024)
#     out = model(x)
#
#     print("Logits:", out[0].shape)
#     print("Embedding:", out[1].shape)
