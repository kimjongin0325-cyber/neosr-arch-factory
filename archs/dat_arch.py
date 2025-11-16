# from registry import ARCH_REGISTRY
upscale = 4  # auto-added by arch factory
# from factory.local_utils import DropPath, DySample, to_2tuple, conv1x1, conv3x3
# type: ignore
import math

import numpy as np
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
from torch.utils import checkpoint


########################################
# 기본 유틸
########################################

def img2windows(img, H_sp, W_sp):
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = (
        img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    )
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))
    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


########################################
# Spatial Gate
########################################

class SpatialGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x, H, W):
        x1, x2 = x.chunk(2, dim=-1)
        B, N, C = x.shape
        x2 = (
            self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C // 2, H, W))
            .flatten(2)
            .transpose(-1, -2)
            .contiguous()
        )
        return x1 * x2


########################################
# SGFN
########################################

class SGFN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = SpatialGate(hidden_features // 2)
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.sg(x, H, W)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


########################################
# Dynamic Position Bias
########################################

class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads),
        )

    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases)
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos


########################################
# Spatial Window Attention
########################################

class Spatial_Attention(nn.Module):
    def __init__(
        self,
        dim,
        idx,
        split_size=[8, 8],
        dim_out=None,
        num_heads=6,
        attn_drop=0.0,
        proj_drop=0.0,
        qk_scale=None,
        position_bias=True,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        if idx == 0:
            H_sp, W_sp = self.split_size
        else:
            W_sp, H_sp = self.split_size

        self.H_sp = H_sp
        self.W_sp = W_sp

        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)

            position_bias_h = torch.arange(1 - self.H_sp, self.H_sp)
            position_bias_w = torch.arange(1 - self.W_sp, self.W_sp)
            biases = torch.stack(
                torch.meshgrid([position_bias_h, position_bias_w], indexing="ij")
            )
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()
            self.register_buffer("rpe_biases", biases)

            coords_h = torch.arange(self.H_sp)
            coords_w = torch.arange(self.W_sp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.H_sp - 1
            relative_coords[:, :, 1] += self.W_sp - 1
            relative_coords[:, :, 0] *= 2 * self.W_sp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2win(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = (
            x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        return x

    def forward(self, qkv, H, W, mask=None):
        q, k, v = qkv[0], qkv[1], qkv[2]
        B, L, C = q.shape
        assert L == H * W

        q = self.im2win(q, H, W)
        k = self.im2win(k, H, W)
        v = self.im2win(v, H, W)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if self.position_bias:
            pos = self.pos(self.rpe_biases)
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1
            )
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        N = attn.shape[3]

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(
                0
            )
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)
        x = windows2img(x, self.H_sp, self.W_sp, H, W)
        return x


########################################
# Axial Spatial / Channel Attention
########################################

class Axial_Spatial_Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        reso=64,
        split_size=[8, 8],
        shift_size=[1, 2],
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        rg_idx=0,
        b_idx=0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.shift_size = shift_size
        self.b_idx = b_idx
        self.rg_idx = rg_idx
        self.patches_resolution = reso
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        assert 0 <= self.shift_size[0] < self.split_size[0]
        assert 0 <= self.shift_size[1] < self.split_size[1]

        self.branch_num = 2

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.attns = nn.ModuleList([
            Spatial_Attention(
                dim // 2,
                idx=i,
                split_size=split_size,
                num_heads=num_heads // 2,
                dim_out=dim // 2,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                position_bias=True,
            )
            for i in range(self.branch_num)
        ])

        if (self.rg_idx % 2 == 0 and self.b_idx > 0 and (self.b_idx - 2) % 4 == 0) or (
            self.rg_idx % 2 != 0 and self.b_idx % 4 == 0
        ):
            attn_mask = self.calculate_mask(reso, reso)
            self.register_buffer("attn_mask_0", attn_mask[0])
            self.register_buffer("attn_mask_1", attn_mask[1])
        else:
            self.register_buffer("attn_mask_0", None)
            self.register_buffer("attn_mask_1", None)

        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, 1),
        )

        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, 1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, 1),
        )

    def calculate_mask(self, H, W):
        img_mask_0 = torch.zeros((1, H, W, 1))
        img_mask_1 = torch.zeros((1, H, W, 1))
        h_slices_0 = (
            slice(0, -self.split_size[0]),
            slice(-self.split_size[0], -self.shift_size[0]),
            slice(-self.shift_size[0], None),
        )
        w_slices_0 = (
            slice(0, -self.split_size[1]),
            slice(-self.split_size[1], -self.shift_size[1]),
            slice(-self.shift_size[1], None),
        )

        h_slices_1 = (
            slice(0, -self.split_size[1]),
            slice(-self.split_size[1], -self.shift_size[1]),
            slice(-self.shift_size[1], None),
        )
        w_slices_1 = (
            slice(0, -self.split_size[0]),
            slice(-self.split_size[0], -self.shift_size[0]),
            slice(-self.shift_size[0], None),
        )

        cnt = 0
        for h in h_slices_0:
            for w in w_slices_0:
                img_mask_0[:, h, w, :] = cnt
                cnt += 1
        cnt = 0
        for h in h_slices_1:
            for w in w_slices_1:
                img_mask_1[:, h, w, :] = cnt
                cnt += 1

        img_mask_0 = img_mask_0.view(
            1,
            H // self.split_size[0],
            self.split_size[0],
            W // self.split_size[1],
            self.split_size[1],
            1,
        )
        img_mask_0 = (
            img_mask_0.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, self.split_size[0], self.split_size[1], 1)
        )
        mask_windows_0 = img_mask_0.view(-1, self.split_size[0] * self.split_size[1])
        attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
        attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 != 0, -100.0).masked_fill(
            attn_mask_0 == 0, 0.0
        )

        img_mask_1 = img_mask_1.view(
            1,
            H // self.split_size[1],
            self.split_size[1],
            W // self.split_size[0],
            self.split_size[0],
            1,
        )
        img_mask_1 = (
            img_mask_1.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, self.split_size[1], self.split_size[0], 1)
        )
        mask_windows_1 = img_mask_1.view(-1, self.split_size[1] * self.split_size[0])
        attn_mask_1 = mask_windows_1.unsqueeze(1) - mask_windows_1.unsqueeze(2)
        attn_mask_1 = attn_mask_1.masked_fill(attn_mask_1 != 0, -100.0).masked_fill(
            attn_mask_1 == 0, 0.0
        )

        return attn_mask_0, attn_mask_1

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W

        qkv = self.qkv(x).reshape(B, L, 3, C).permute(2, 0, 1, 3)
        v = qkv[2].transpose(-2, -1).contiguous().view(B, C, H, W)

        max_split = max(self.split_size)
        pad_l = pad_t = 0
        pad_r = (max_split - W % max_split) % max_split
        pad_b = (max_split - H % max_split) % max_split

        qkv = qkv.reshape(3 * B, H, W, C).permute(0, 3, 1, 2)
        qkv = (
            F.pad(qkv, (pad_l, pad_r, pad_t, pad_b))
            .reshape(3, B, C, -1)
            .transpose(-2, -1)
        )
        _H = H + pad_b
        _W = W + pad_r
        _L = _H * _W

        if (self.rg_idx % 2 == 0 and self.b_idx > 0 and (self.b_idx - 2) % 4 == 0) or (
            self.rg_idx % 2 != 0 and self.b_idx % 4 == 0
        ):
            qkv = qkv.view(3, B, _H, _W, C)
            qkv_0 = torch.roll(
                qkv[:, :, :, :, : C // 2],
                shifts=(-self.shift_size[0], -self.shift_size[1]),
                dims=(2, 3),
            ).view(3, B, _L, C // 2)

            qkv_1 = torch.roll(
                qkv[:, :, :, :, C // 2 :],
                shifts=(-self.shift_size[1], -self.shift_size[0]),
                dims=(2, 3),
            ).view(3, B, _L, C // 2)

            if self.patches_resolution != _H or self.patches_resolution != _W:
                mask_tmp = self.calculate_mask(_H, _W)
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=mask_tmp[0].to(x.device))
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=mask_tmp[1].to(x.device))
            else:
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=self.attn_mask_0)
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=self.attn_mask_1)

            x1 = torch.roll(
                x1_shift, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2)
            )
            x2 = torch.roll(
                x2_shift, shifts=(self.shift_size[1], self.shift_size[0]), dims=(1, 2)
            )

            x1 = x1[:, :H, :W, :].reshape(B, L, C // 2)
            x2 = x2[:, :H, :W, :].reshape(B, L, C // 2)
            attened_x = torch.cat([x1, x2], dim=2)

        else:
            x1 = self.attns[0](qkv[:, :, :, : C // 2], _H, _W)[:, :H, :W, :].reshape(
                B, L, C // 2
            )
            x2 = self.attns[1](qkv[:, :, :, C // 2 :], _H, _W)[:, :H, :W, :].reshape(
                B, L, C // 2
            )
            attened_x = torch.cat([x1, x2], dim=2)

        conv_x = self.dwconv(v)

        channel_map = (
            self.channel_interaction(conv_x)
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(B, 1, C)
        )

        attention_reshape = attened_x.transpose(-2, -1).contiguous().view(B, C, H, W)
        spatial_map = self.spatial_interaction(attention_reshape)

        attened_x = attened_x * torch.sigmoid(channel_map)
        conv_x = torch.sigmoid(spatial_map) * conv_x
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, L, C)

        x = attened_x + conv_x
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Axial_Channel_Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, 1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, 1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, 1),
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v_ = v.transpose(-2, -1).reshape(B, C, N).contiguous().view(B, C, H, W)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        attened_x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)

        conv_x = self.dwconv(v_)
        attention_reshape = attened_x.transpose(-2, -1).contiguous().view(B, C, H, W)
        channel_map = self.channel_interaction(attention_reshape)

        spatial_map = (
            self.spatial_interaction(conv_x)
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(B, N, 1)
        )

        attened_x = attened_x * torch.sigmoid(spatial_map)
        conv_x = conv_x * torch.sigmoid(channel_map)
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, N, C)

        x = attened_x + conv_x
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


########################################
# DATB Block
########################################

class DATB(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        reso=64,
        split_size=[2, 4],
        shift_size=[1, 2],
        expansion_factor=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        rg_idx=0,
        b_idx=0,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)

        if b_idx % 2 == 0:
            self.attn = Axial_Spatial_Attention(
                dim,
                num_heads=num_heads,
                reso=reso,
                split_size=split_size,
                shift_size=shift_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                rg_idx=rg_idx,
                b_idx=b_idx,
            )
        else:
            self.attn = Axial_Channel_Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )

        self.drop_path = nn.Identity()

        ffn_hidden_dim = int(dim * expansion_factor)
        self.ffn = SGFN(
            in_features=dim,
            hidden_features=ffn_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
        )
        self.norm2 = norm_layer(dim)

    def forward(self, x, x_size):
        H, W = x_size
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W))
        return x


########################################
# Residual Group
########################################

class ResidualGroup(nn.Module):
    def __init__(
        self,
        dim,
        reso,
        num_heads,
        split_size=[2, 4],
        expansion_factor=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_paths=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        depth=2,
        use_chk=False,
        resi_connection="1conv",
        rg_idx=0,
    ):
        super().__init__()
        self.use_chk = use_chk
        self.reso = reso

        self.blocks = nn.ModuleList([
            DATB(
                dim=dim,
                num_heads=num_heads,
                reso=reso,
                split_size=split_size,
                shift_size=[split_size[0] // 2, split_size[1] // 2],
                expansion_factor=expansion_factor,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_paths[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                rg_idx=rg_idx,
                b_idx=i,
            )
            for i in range(depth)
        ])

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )

    def forward(self, x, x_size):
        H, W = x_size
        res = x
        for blk in self.blocks:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x, x_size, use_reentrant=False)
            else:
                x = blk(x, x_size)

        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        x = self.conv(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = res + x

        return x


########################################
# ❗ 4X UPSAMPLE FIX (완전 정상 복구)
########################################

class Upsample(nn.Sequential):
    """4x Upsample (fixed)."""

    def __init__(self, num_feat):
        m = []
        # PixelShuffle(2) × 2 → 4배 업스케일
        m.append(nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1))
        m.append(nn.PixelShuffle(2))
        m.append(nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1))
        m.append(nn.PixelShuffle(2))
        super().__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """4x UpsampleOneStep"""

    def __init__(self, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        # 4배 업스케일: PixelShuffle(4)
        m.append(nn.Conv2d(num_feat, num_out_ch * 16, 3, 1, 1))
        m.append(nn.PixelShuffle(4))
        super().__init__(*m)

    def flops(self):
        h, w = self.input_resolution
        return h * w * self.num_feat * 3 * 9


########################################
# DAT MAIN
########################################

class dat(nn.Module):
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(
            m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)
        ):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def __init__(
        self,
        img_size=64,
        in_chans=3,
        embed_dim=180,
        split_size=[2, 4],
        depth=[2, 2, 2, 2],
        num_heads=[2, 2, 2, 2],
        expansion_factor=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_chk=False,
        upscale=4,
        img_range=1.0,
        resi_connection="1conv",
        upsampler="pixelshuffle",
        **kwargs,
    ):
        super().__init__()

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range

        if in_chans == 3:
            self.mean = torch.Tensor((0.5, 0.5, 0.5)).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        self.upscale = 4
        self.upsampler = upsampler

        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        self.num_layers = len(depth)
        self.use_chk = use_chk
        self.num_features = embed_dim

        self.before_RG = nn.Sequential(
            Rearrange("b c h w -> b (h w) c"), nn.LayerNorm(embed_dim)
        )

        curr_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]
        heads = num_heads

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = ResidualGroup(
                dim=embed_dim,
                num_heads=heads[i],
                reso=img_size,
                split_size=split_size,
                expansion_factor=expansion_factor,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_paths=dpr[sum(depth[:i]): sum(depth[: i + 1])],
                act_layer=act_layer,
                norm_layer=norm_layer,
                depth=depth[i],
                use_chk=use_chk,
                resi_connection=resi_connection,
                rg_idx=i,
            )
            self.layers.append(layer)

        self.norm = norm_layer(curr_dim)

        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
            )

        if self.upsampler == "pixelshuffle":
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
            )
            self.upsample = Upsample(num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        elif self.upsampler == "pixelshuffledirect":
            self.upsample = UpsampleOneStep(
                embed_dim, num_out_ch, (img_size, img_size)
            )

        self.apply(self._init_weights)

    def forward_features(self, x):
        _, _, H, W = x.shape
        x_size = [H, W]
        x = self.before_RG(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        return x

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == "pixelshuffle":
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == "pixelshuffledirect":
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)

        x = x / self.img_range + self.mean
        return x


########################################
# Factory
########################################

def dat_s(**kwargs):
    return dat(
        in_chans=3,
        img_range=1.0,
        split_size=[8, 16],
        depth=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        expansion_factor=2,
        resi_connection="1conv",
        **kwargs,
    )


def dat_m(**kwargs):
    return dat(
        in_chans=3,
        img_range=1.0,
        split_size=[8, 32],
        depth=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        expansion_factor=4,
        resi_connection="1conv",
        **kwargs,
    )


def dat_2(**kwargs):
    return dat(
        in_chans=3,
        img_range=1.0,
        split_size=[8, 32],
        depth=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        expansion_factor=2,
        resi_connection="1conv",
        **kwargs,
    )
