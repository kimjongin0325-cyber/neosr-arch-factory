import torch
from torch import nn
import torch.nn.functional as F


# ============================================================
# DySample (독립 실행형 최소 버전)
# ============================================================
class DySample(nn.Module):
    def __init__(self, in_ch, out_ch, scale):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * scale * scale, 3, padding=1)
        self.ps = nn.PixelShuffle(scale)

    def forward(self, x):
        return self.ps(self.conv(x))


# ============================================================
# Conv Blocks
# ============================================================
class Conv3XC(nn.Module):
    """3x3 Conv + GELU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GELU()
        )

    def forward(self, x):
        return self.body(x)


def conv1x1(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, 1)


def conv3x3(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, 3, padding=1)


# ============================================================
# SPAB (Spatial Attention Block)
# ============================================================
class SPAB(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.conv_mask = nn.Sequential(
            nn.Conv2d(in_ch, 1, 1),
            nn.Sigmoid(),
        )

        self.conv_out = nn.Conv2d(in_ch, in_ch, 1)

    def forward(self, x):
        attn = self.conv_mask(x)
        return self.conv_out(x * attn + x)


# ============================================================
# SPABS (Stacked SPAB)
# ============================================================
class SPABS(nn.Module):
    def __init__(self, in_ch, num_blocks=3):
        super().__init__()
        self.blocks = nn.Sequential(*[SPAB(in_ch) for _ in range(num_blocks)])

    def forward(self, x):
        return self.blocks(x)


# ============================================================
# Span+ Main Model (독립 실행형)
# ============================================================
class spanplus(nn.Module):
    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        feature_channels=48,
        num_blocks=4,
        upscale=4,
        upsampler="dys",   # dys / ps
    ):
        super().__init__()

        self.upscale = upscale

        # ------------------------
        # 1) shallow feature
        # ------------------------
        self.conv_first = Conv3XC(num_in_ch, feature_channels)

        # ------------------------
        # 2) SPABS blocks
        # ------------------------
        self.body = nn.Sequential(
            *[SPABS(feature_channels, 3) for _ in range(num_blocks)]
        )

        # ------------------------
        # 3) Upsampler 선택
        # ------------------------
        if upsampler == "dys":
            self.upsampler = DySample(feature_channels, num_out_ch, upscale)
        else:
            # fallback: PixelShuffle
            self.upsampler = nn.Sequential(
                nn.Conv2d(feature_channels, num_out_ch * upscale * upscale, 3, padding=1),
                nn.PixelShuffle(upscale)
            )

    def forward(self, x):
        fea = self.conv_first(x)
        fea = self.body(fea)
        out = self.upsampler(fea)
        return out
