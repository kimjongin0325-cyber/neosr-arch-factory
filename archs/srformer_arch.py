import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------
# Utils
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
# Window ops (21×21 windows)
# --------------------------------------------------------------

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(
        B,
        H // window_size, window_size,
        W // window_size, window_size,
        C
    )
    windows = x.permute(0,1,3,2,4,5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / ((H * W) // (window_size * window_size)))
    x = windows.view(
        B,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        -1
    )
    x = x.permute(0,1,3,2,4,5).contiguous()
    x = x.view(B, H, W, -1)
    return x


# --------------------------------------------------------------
# PSA Attention (Nomos8kSCSRFormer)
# embed_dim = 180, num_heads = 6
# --------------------------------------------------------------

class PSA_Attention(nn.Module):
    def __init__(self, dim, num_heads=6, window_size=21, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = head_dim ** -0.5
        self.window_size = window_size

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim // 2, bias=qkv_bias)  # 90 for dim=180
        self.proj = nn.Linear(dim, dim)

        # 21×21 = 441
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((window_size * window_size, num_heads))
        )

        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size),
            torch.arange(window_size),
            indexing="ij"
        ))  # [2,21,21]
        coords_flat = coords.flatten(1)
        relative = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative = relative.permute(1,2,0).contiguous()

        relative[:,:,0] += window_size - 1
        relative[:,:,1] += window_size - 1
        relative_idx = relative[:,:,0]*(2*window_size - 1) + relative[:,:,1]

        self.register_buffer("relative_position_index", relative_idx)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.view(B, H, W, C)

        win = self.window_size
        windows = window_partition(x, win)
        Bwin = windows.shape[0]
        windows = windows.view(Bwin, win*win, C)

        q = self.q(windows)
        kv = self.kv(windows)
        k, v = torch.split(kv, kv.shape[-1] // 2, dim=-1)

        q = q.reshape(Bwin, -1, self.num_heads, C//self.num_heads).transpose(1,2)
        k = k.reshape(Bwin, -1, self.num_heads, C//self.num_heads).transpose(1,2)
        v = v.reshape(Bwin, -1, self.num_heads, C//self.num_heads).transpose(1,2)

        attn = (q @ k.transpose(-2,-1)) * self.scale

        rel_pos = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ]
        rel_pos = rel_pos.view(win*win, win*win, -1)
        rel_pos = rel_pos.permute(2,0,1).unsqueeze(0)
        attn = attn + rel_pos

        attn = F.softmax(attn, dim=-1)

        out = attn @ v
        out = out.transpose(1,2).reshape(Bwin, win*win, C)
        out = self.proj(out)

        out = out.view(Bwin, win, win, C)
        x = window_reverse(out, win, H, W)
        x = x.view(B, H*W, C)
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
            nn.Conv2d(hidden, hidden, kernel_size=5, stride=1, padding=2, groups=hidden),
            nn.GELU()
        )
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x)
        x = x.view(B, H, W, -1).permute(0,3,1,2)
        x = self.dwconv(x)
        x = x.permute(0,2,3,1).view(B, N, -1)
        x = self.fc2(x)
        return x


# --------------------------------------------------------------
# SRFormer Block
# --------------------------------------------------------------

class SRFBlock(nn.Module):
    def __init__(self, dim, num_heads=6, window_size=21, mlp_ratio=2, drop_path_prob=0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = PSA_Attention(dim, num_heads=num_heads, window_size=window_size)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

        self.drop_path = DropPath(drop_path_prob)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


# --------------------------------------------------------------
# Residual Group
# --------------------------------------------------------------

class ResidualGroup(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size):
        super().__init__()
        self.blocks = nn.ModuleList([
            SRFBlock(dim, num_heads=num_heads, window_size=window_size)
            for _ in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, 3,1,1)

    def forward(self, x, H, W):
        identity = x

        for blk in self.blocks:
            x = blk(x, H, W)

        B, N, C = x.shape
        xf = x.view(B, H, W, C).permute(0,3,1,2)
        x = self.conv(xf)
        x = x.permute(0,2,3,1).view(B, N, C)

        return x + identity


# --------------------------------------------------------------
# PatchEmbed / UnEmbed
# --------------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, in_ch, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, 3,1,1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.permute(0,2,3,1).view(B, H*W, C)
        x = self.norm(x)
        return x, H, W


class PatchUnEmbed(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.view(B, H, W, C).permute(0,3,1,2)
        return x


# --------------------------------------------------------------
# Upsample (×2 ×2 = ×4)
# --------------------------------------------------------------

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up1 = nn.Conv2d(dim, 64 * 4, 3,1,1)
        self.up2 = nn.Conv2d(64, 64 * 4, 3,1,1)

        self.pixelshuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.pixelshuffle(self.up1(x))
        x = self.pixelshuffle(self.up2(x))
        return x


# --------------------------------------------------------------
# SRFormer Model (Nomos8k)
# --------------------------------------------------------------

class SRFormer(nn.Module):
    def __init__(self, upscale=4):
        super().__init__()

        self.upscale = upscale
        self.embed_dim = 180
        self.num_heads = 6
        self.window_size = 21
        self.depths = [6,6,6,6,6,6]

        # patch
        self.patch_embed = PatchEmbed(3, self.embed_dim)
        self.unembed = PatchUnEmbed()

        # transformer layers
        self.layers = nn.ModuleList([
            ResidualGroup(self.embed_dim, d, self.num_heads, self.window_size)
            for d in self.depths
        ])

        # after body
        self.norm = nn.LayerNorm(self.embed_dim)
        self.conv_after_body = nn.Conv2d(self.embed_dim, self.embed_dim, 3,1,1)

        # before upsample
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(self.embed_dim, 64, 3,1,1),
            nn.GELU(),
        )

        # first conv
        self.conv_first = nn.Conv2d(3, self.embed_dim, 3,1,1)

        # last conv
        self.conv_last = nn.Conv2d(64, 3, 3,1,1)

        # upsample
        self.upsample = Upsample(self.embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        bic = F.interpolate(x, scale_factor=self.upscale, mode="bicubic", align_corners=False)

        # patch embed
        fea, h, w = self.patch_embed(x)

        # transformer
        for layer in self.layers:
            fea = layer(fea, h, w)

        fea = self.norm(fea)
        fea = self.unembed(fea, h, w)
        fea = self.conv_after_body(fea)

        fea = self.conv_before_upsample(fea)
        fea = self.upsample(fea)
        fea = self.conv_last(fea)

        return fea + bic
