# Copyright (C) 2024  Hong Cao, Jiahua He, Tao Li, Sheng-You Huang and Huazhong University of Science
# Copyright (C) 2022  Kai Zhang (cskaizhang@gmail.com, https://cszn.github.io/).

# All rights reserved.

# 3D version of Swin-Conv-UNet (https://arxiv.org/pdf/2203.13278.pdf), as used by EMReady (see
# issue #1). Adapted from their implementation: the einops rearranges are written as plain
# reshapes/permutes and timm's DropPath/trunc_normal_ and the FilterResponseNorm3d module are
# replaced by local equivalents, so that this adds no dependencies. SCUNet3D wraps the network
# with the same per-sample input normalization and residual connection as Unet3D.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

from .unet import Unet3D, set_padding_mode


class FilterResponseNorm3d(nn.Module):
    """Filter response normalization with a thresholded linear unit (Singh & Krishnan, 2019)."""

    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        shape = (1, num_features, 1, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.tau = nn.Parameter(torch.zeros(shape))
        self.eps = eps

    def forward(self, x):
        nu2 = x.pow(2).mean(dim=(2, 3, 4), keepdim=True)
        x = x * torch.rsqrt(nu2 + self.eps)
        return torch.max(self.gamma * x + self.beta, self.tau)


class DropPath(nn.Module):
    """Stochastic depth: drops the whole residual branch per sample during training."""

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.dim() - 1)).bernoulli_(keep)
        return x * mask / keep


def _to_windows(x, p):
    """(b, H, W, D, c) -> (b, number of windows, p^3, c)"""
    b, h, w, d, c = x.shape
    x = x.view(b, h // p, p, w // p, p, d // p, p, c).permute(0, 1, 3, 5, 2, 4, 6, 7)
    return x.reshape(b, (h // p) * (w // p) * (d // p), p**3, c)


def _from_windows(x, p, n_windows):
    """Inverse of _to_windows, for a cube of n_windows^3 windows."""
    b, c = x.shape[0], x.shape[-1]
    x = x.view(b, n_windows, n_windows, n_windows, p, p, p, c).permute(0, 1, 4, 2, 5, 3, 6, 7)
    return x.reshape(b, n_windows * p, n_windows * p, n_windows * p, c)


class WMSA(nn.Module):
    """(Shifted) window multi-head self-attention."""

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim**-0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = type
        self.embedding_layer = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)
        n = 2 * window_size - 1
        self.relative_position_params = nn.Parameter(torch.zeros(self.n_heads, n, n, n))
        nn.init.trunc_normal_(self.relative_position_params, std=0.02)
        self.linear = nn.Linear(self.input_dim, self.output_dim)

        p = window_size
        cord = torch.stack(torch.meshgrid(torch.arange(p), torch.arange(p), torch.arange(p), indexing="ij"), dim=-1).reshape(-1, 3)
        self.register_buffer("relation", cord[:, None, :] - cord[None, :, :] + p - 1, persistent=False)

    def generate_mask(self, n_windows, p, shift, device):
        attn_mask = torch.zeros(n_windows, n_windows, n_windows, p, p, p, p, p, p, dtype=torch.bool, device=device)
        if self.type == "W":
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :, :s, :, :, s:, :, :] = True
        attn_mask[-1, :, :, s:, :, :, :s, :, :] = True
        attn_mask[:, -1, :, :, :s, :, :, s:, :] = True
        attn_mask[:, -1, :, :, s:, :, :, :s, :] = True
        attn_mask[:, :, -1, :, :, :s, :, :, s:] = True
        attn_mask[:, :, -1, :, :, s:, :, :, :s] = True
        return attn_mask.view(1, 1, n_windows**3, p**3, p**3)

    def forward(self, x):
        p = self.window_size
        if self.type != "W":
            x = torch.roll(x, shifts=(-(p // 2),) * 3, dims=(1, 2, 3))
        n_windows = x.shape[1] // p
        assert x.shape[1] == x.shape[2] == x.shape[3]
        x = _to_windows(x, p)
        b, nw, n_p, _ = x.shape

        qkv = self.embedding_layer(x).view(b, nw, n_p, 3 * self.n_heads, self.head_dim).permute(3, 0, 1, 2, 4)
        q, k, v = qkv.chunk(3, dim=0)  # each (heads, b, windows, p^3, head_dim)
        sim = torch.einsum("hbwpc,hbwqc->hbwpq", q, k) * self.scale
        rel = self.relation
        sim = sim + self.relative_position_params[:, rel[..., 0], rel[..., 1], rel[..., 2]][:, None, None]

        if self.type != "W":
            sim = sim.masked_fill_(self.generate_mask(n_windows, p, p // 2, x.device), float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum("hbwij,hbwjc->hbwic", probs, v)
        output = output.permute(1, 2, 3, 0, 4).reshape(b, nw, n_p, self.n_heads * self.head_dim)
        output = _from_windows(self.linear(output), p, n_windows)

        if self.type != "W":
            output = torch.roll(output, shifts=(p // 2,) * 3, dims=(1, 2, 3))
        return output


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type="W", input_resolution=None):
        super().__init__()
        assert type in ["W", "SW"]
        self.type = "W" if input_resolution <= window_size else type
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


class ConvTransBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type="W", input_resolution=None):
        super().__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        assert type in ["W", "SW"]
        type = "W" if input_resolution <= window_size else type

        self.trans_block = Block(trans_dim, trans_dim, head_dim, window_size, drop_path, type, input_resolution)
        self.conv1_1 = nn.Conv3d(conv_dim + trans_dim, conv_dim + trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv3d(conv_dim + trans_dim, conv_dim + trans_dim, 1, 1, 0, bias=True)
        self.conv_block = nn.Sequential(
            nn.Conv3d(conv_dim, conv_dim, 3, 1, 1, bias=False),
            FilterResponseNorm3d(conv_dim),
            nn.Conv3d(conv_dim, conv_dim, 3, 1, 1, bias=False),
            FilterResponseNorm3d(conv_dim),
        )

    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = self.trans_block(trans_x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        return x + self.conv1_2(torch.cat((conv_x, trans_x), dim=1))


class SCUNet(nn.Module):
    def __init__(self, in_nc=1, config=(2, 2, 2, 2, 2, 2, 2), dim=32, drop_path_rate=0.2, input_resolution=48, head_dim=16, window_size=3, n_classes=1, resample="original"):
        super().__init__()
        # "original": non-overlapping 2^3 strided convs down and transposed convs up, which
        # build a period-2 grid into the output. "overlapping", as in Unet3D: 3^3 strided convs
        # down, nearest-neighbor upsampling + a 3^3 conv up
        assert resample in ("original", "overlapping")

        def down(cin, cout):
            if resample == "original":
                return nn.Conv3d(cin, cout, 2, 2, 0, bias=False)
            return nn.Conv3d(cin, cout, 3, 2, 1, bias=False)

        def up(cin, cout):
            if resample == "original":
                return nn.ConvTranspose3d(cin, cout, 2, 2, 0, bias=False)
            return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv3d(cin, cout, 3, 1, 1, bias=False))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        def stage(n_blocks, begin, chans, resolution):
            return [
                ConvTransBlock(chans, chans, head_dim, window_size, dpr[i + begin], "W" if not i % 2 else "SW", resolution)
                for i in range(n_blocks)
            ]

        begins = [sum(config[:i]) for i in range(len(config))]
        self.m_head = nn.Sequential(nn.Conv3d(in_nc, dim, 3, 1, 1, bias=False))
        self.m_down1 = nn.Sequential(*stage(config[0], begins[0], dim // 2, input_resolution), down(dim, 2 * dim))
        self.m_down2 = nn.Sequential(*stage(config[1], begins[1], dim, input_resolution // 2), down(2 * dim, 4 * dim))
        self.m_down3 = nn.Sequential(*stage(config[2], begins[2], 2 * dim, input_resolution // 4), down(4 * dim, 8 * dim))
        self.m_body = nn.Sequential(*stage(config[3], begins[3], 4 * dim, input_resolution // 8))
        self.m_up3 = nn.Sequential(up(8 * dim, 4 * dim), *stage(config[4], begins[4], 2 * dim, input_resolution // 4))
        self.m_up2 = nn.Sequential(up(4 * dim, 2 * dim), *stage(config[5], begins[5], dim, input_resolution // 2))
        self.m_up1 = nn.Sequential(up(2 * dim, dim), *stage(config[6], begins[6], dim // 2, input_resolution))
        self.m_tail = nn.Sequential(nn.Conv3d(dim, n_classes, 3, 1, 1, bias=False))

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        return self.m_tail(x + x1)


class SCUNet3D(nn.Module):
    """
    SCUNet with the same per-sample robust input normalization and residual connection as
    Unet3D (see Unet3D.get_loc_and_scale), so it is a drop-in replacement for LitUnet3D.unet.
    The box size must be divisible by 8 * window_size.
    """

    def __init__(self, box_size, dim=32, config=(2, 2, 2, 2, 2, 2, 2), head_dim=16, window_size=3, drop_path_rate=0.0, residual=True, padding_mode="zeros", resample="original"):
        super().__init__()
        self.residual = residual
        self.net = SCUNet(1, config, dim, drop_path_rate, box_size, head_dim, window_size, 1, resample)
        set_padding_mode(self, padding_mode)

    def forward(self, volume):
        loc, scale = Unet3D.get_loc_and_scale(self, volume)  # doesn't use self
        volume = (volume - loc) / scale
        output = self.net(volume)
        if self.residual:
            output = output + volume
        return output * scale + loc
