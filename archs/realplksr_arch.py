import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from functools import partial

# =======================================
# 📌 DySample (독립형 버전)
# =======================================
class DySample(nn.Module):
    """
    Lightweight Dynamic Upsampling Layer
    (minimal independent version)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        scale,
        groups=4,
        end_convolution=True,
    ):
        super().__init__()
        self.scale = scale
        self.end_convolution = end_convolution

        self.weight_gen = nn.Conv2d(
            in_channels,
            groups * scale * scale,
            1,
            1,
            0
        )

        self.groups = groups
        self.out_channels = out_channels

        if end_convolution:
            self.final = nn.Conv2d(
                in_channels,
                out_channels * (scale ** 2),
                1, 1, 0
            )
        else:
            self.final = None

    def forward(self, x):
        B, C, H, W = x.shape

        # Generate per-pixel weights
        w = self.weight_gen(x)  # (B, groups * s^2, H, W)
        w = w.view(B, self.groups, self.scale * self.scale, H, W)

        # Apply pixel weights
        x = x.view(B, self.groups, C // self.groups, H, W)
        x = x * w.sum(dim=2, keepdim=True)  # simple mixing

        x = x.view(B, C, H, W)

        if self.final is not None:
            x = self.final(x)

        return F.pixel_shuffle(x, self.scale)


# =======================================
# 💠 Blocks
# =======================================
class DCCM(nn.Sequential):
    """Doubled Convolutional Channel Mixer"""
    def __init__(self, dim: int):
        super().__init__(
            nn.Conv2d(dim, dim * 2, 3, 1, 1),
            nn.Mish(),
            nn.Conv2d(dim * 2, dim, 3, 1, 1),
        )
        trunc_normal_(self[-1].weight, std=0.02)


class PLKConv2d(nn.Module):
    """Partial Large Kernel Convolutional Layer"""
    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2)
        trunc_normal_(self.conv.weight, std=0.02)
        self.idx = dim

    def forward(self, x):
        if self.training:
            x1, x2 = torch.split(x, [self.idx, x.size(1) - self.idx], dim=1)
            x1 = self.conv(x1)
            return torch.cat([x1, x2], dim=1)
        x[:, : self.idx] = self.conv(x[:, : self.idx])
        return x


class EA(nn.Module):
    """Element-wise Attention"""
    def __init__(self, dim: int):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.Sigmoid())
        trunc_normal_(self.f[0].weight, std=0.02)

    def forward(self, x):
        return x * self.f(x)


class PLKBlock(nn.Module):
    def __init__(self, dim, kernel_size, split_ratio, norm_groups, use_ea=True):
        super().__init__()
        self.channel_mixer = DCCM(dim)
        pdim = int(dim * split_ratio)
        self.lk = PLKConv2d(pdim, kernel_size)
        self.attn = EA(dim) if use_ea else nn.Identity()
        self.refine = nn.Conv2d(dim, dim, 1, 1, 0)
        self.norm = nn.GroupNorm(norm_groups, dim)

        trunc_normal_(self.refine.weight, std=0.02)
        nn.init.constant_(self.norm.bias, 0)
        nn.init.constant_(self.norm.weight, 1.0)

    def forward(self, x):
        skip = x
        x = self.channel_mixer(x)
        x = self.lk(x)
        x = self.attn(x)
        x = self.refine(x)
        x = self.norm(x)
        return x + skip


# =======================================
# 🔥 realplksr (독립형)
# =======================================
class realplksr(nn.Module):
    def __init__(
        self,
        in_ch=3,
        out_ch=3,
        dim=64,
        n_blocks=28,
        upscaling_factor=4,
        kernel_size=17,
        split_ratio=0.25,
        use_ea=True,
        norm_groups=4,
        dropout=0,
        dysample=False,
        **kwargs,
    ):
        super().__init__()

        self.upscale = upscaling_factor
        self.dysample = dysample

        if not self.training:
            dropout = 0

        self.feats = nn.Sequential(
            nn.Conv2d(in_ch, dim, 3, 1, 1),
            *[
                PLKBlock(dim, kernel_size, split_ratio, norm_groups, use_ea)
                for _ in range(n_blocks)
            ],
            nn.Dropout2d(dropout),
            nn.Conv2d(dim, out_ch * (upscaling_factor ** 2), 3, 1, 1),
        )
        trunc_normal_(self.feats[0].weight, std=0.02)
        trunc_normal_(self.feats[-1].weight, std=0.02)

        self.repeat_op = partial(
            torch.repeat_interleave,
            repeats=upscaling_factor ** 2,
            dim=1,
        )

        if dysample and upscaling_factor != 1:
            groups = out_ch if upscaling_factor % 2 != 0 else 4
            self.to_img = DySample(
                out_ch * (upscaling_factor ** 2),
                out_ch,
                upscaling_factor,
                groups=groups,
                end_convolution=True,
            )
        else:
            self.to_img = nn.PixelShuffle(upscaling_factor)

    def forward(self, x):
        x = self.feats(x) + self.repeat_op(x)
        return self.to_img(x)


def realplksr_s(**kwargs):
    return realplksr(n_blocks=12, kernel_size=13, use_ea=False, **kwargs)

def realplksr_l(**kwargs):
    return realplksr(dim=96, **kwargs)
