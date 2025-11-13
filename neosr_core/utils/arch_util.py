# Standalone arch_util.py (no argparse, ONNX-safe)

from collections.abc import Iterable
from itertools import repeat
import torch
from torch import nn, Tensor
from torch.nn import functional as F

# -----------------------------
# DropPath
# -----------------------------
def drop_path(x, drop_prob=0.0, training=False, scale_by_keep=True):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)

    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if scale_by_keep:
        random_tensor.div_(keep_prob)

    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


# -----------------------------
# DySample
# -----------------------------
class DySample(nn.Module):
    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (
            torch.stack(torch.meshgrid([h, h], indexing="ij"))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def __init__(self, in_channels, out_ch, scale=2, groups=4, end_convolution=True):
        super().__init__()
        assert in_channels % groups == 0, "groups must divide in_channels"

        out_channels = 2 * groups * scale * scale
        self.scale = scale
        self.groups = groups
        self.end_convolution = end_convolution

        if end_convolution:
            self.end_conv = nn.Conv2d(in_channels, out_ch, 1)

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        nn.init.trunc_normal_(self.offset.weight, std=0.02)
        nn.init.constant_(self.scope.weight, 0)

        self.register_buffer("init_pos", self._init_pos())

    def forward(self, x):
        offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)

        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5

        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij"))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .type(x.dtype)
            .to(x.device)
        )

        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(
            1, 2, 1, 1, 1
        )
        coords = 2 * (coords + offset) / normalizer - 1

        coords = (
            F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale)
            .view(B, 2, -1, self.scale * H, self.scale * W)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )

        output = F.grid_sample(
            x.reshape(B * self.groups, -1, H, W),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(B, -1, self.scale * H, self.scale * W)

        if self.end_convolution:
            output = self.end_conv(output)

        return output


# -----------------------------
# tuple helpers
# -----------------------------
def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
