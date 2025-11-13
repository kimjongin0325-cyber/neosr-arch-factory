import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# 1) DropPath (Stochastic Depth)
# -----------------------
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# -----------------------
# 2) to_2tuple
# -----------------------
def to_2tuple(x):
    return (x, x) if not isinstance(x, tuple) else x


# -----------------------
# 3) DySample (from neosr)
# -----------------------
class DySample(nn.Module):
    """Dynamic upsampling module."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 4, 1)
    
    def forward(self, x):
        x = self.conv(x)
        return F.pixel_shuffle(x, 2)


# -----------------------
# 4) 서브 모듈들에서 필요할 수 있는 작은 유틸
# -----------------------
def conv1x1(in_ch, out_ch, bias=True):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=bias)

def conv3x3(in_ch, out_ch, bias=True):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=bias)

