import torch
import torch.nn as nn
import torch.nn.functional as F
from math import prod

# --------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------

def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    output = x / keep_prob * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# --------------------------------------------------------------
# Window operations (PSA-friendly)
# --------------------------------------------------------------

def window_partition(x, window_size):
    """
    x: (B, H, W, C)
    return: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B,
               H // window_size, window_size,
               W // window_size, window_size,
               C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    windows: (num_windows*B, window_size, window_size, C)
    return: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B,
                     H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


# --------------------------------------------------------------
# PSA Attention (Nomos8kSCSRFormer)
# --------------------------------------------------------------

class PSA_Attention(nn.Module):
    def __init__(self, dim, num_heads=6, window_size=21, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = head_dim ** -0.5
        self.window_size = window_size

        # q, kv, proj
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 0.5, bias=qkv_bias)  # weight: [90,180]
        self.proj = nn.Linear(dim, dim)

        # relative position bias table
        # 21x21 = 441
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((window_size * window_size, num_heads))
        )

        # relative index
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size),
            torch.arange(window_size),
            indexing='ij'
        ))  # [2, 21, 21]
        coords_flatten = torch.flatten(coords, 1)  # [2,441]
        relative = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2,441,441]
        relative = relative.permute(1, 2, 0).contiguous()  # [441,441,2]
        relative[:, :, 0] += window_size - 1
        relative[:, :, 1] += window_size - 1
        relative_idx = relative[:, :, 0] * (2 * window_size - 1) + relative[:, :, 1]
        self.register_buffer("relative_position_index", relative_idx)

    def forward(self, x, H, W):
        B, N, C = x.shape  # B, (H*W), C
        x = x.view(B, H, W, C)

        windows = window_partition(x, self.window_size)  # [B*NW, win,win,C]
        Bwin = windows.shape[0]
        win = self.window_size
        xw = windows.view(Bwin, win*win, C)

        q = self.q(xw)
        kv = self.kv(xw)
        k, v = torch.split(kv, kv.shape[-1] // 2, dim=-1)

        q = q.reshape(Bwin, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.reshape(Bwin, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.reshape(Bwin, -1, self.num_heads, C // self.num_heads).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(win * win, win * win, -1)  # [441,441,6]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(0)
        attn = attn + relative_position_bias  # head, win^2, win^2

        attn = F.softmax(attn, dim=-1)

        out = attn @ v  # [Bwin, head, win^2, dim_head]
        out = out.transpose(1, 2).reshape(Bwin, win * win, C)

        out = self.proj(out)
        out = out.view(Bwin, win, win, C)

        x = window_reverse(out, win, H, W)
        x = x.view(B, H * W, C)
        return x


# --------------------------------------------------------------
# MLP (DWConv + Linear)
# --------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=2):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden, hidden, 5, 1, 2, groups=hidden),
            nn.GELU()
        )
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x)
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).view(B, N, -1)
        x = self.fc2(x)
        return x
