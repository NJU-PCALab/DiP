import math
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Embed(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer=None,
            bias: bool = True,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[..., None].float() * freqs[None, ...]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        x = self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
        return x


def precompute_freqs_cis_2d(dim: int, height: int, width: int, theta: float = 10000.0, scale=16.0):
    x_pos = torch.linspace(0, scale, width)
    y_pos = torch.linspace(0, scale, height)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    x_freqs = torch.outer(x_pos, freqs).float()
    y_freqs = torch.outer(y_pos, freqs).float()
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1)
    freqs_cis = freqs_cis.reshape(height * width, -1)
    return freqs_cis


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs_cis = freqs_cis[None, :, None, :]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = RMSNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, pos, mask) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rotary_emb(q, k, freqs_cis=pos)
        q = q.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()
        v = v.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()

        x = scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FlattenDiTBlock(nn.Module):
    def __init__(self, hidden_size, groups, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = RAttention(hidden_size, num_heads=groups, qkv_bias=False)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = FeedForward(hidden_size, mlp_hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, pos, mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), pos, mask=mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class LocalDecoder(nn.Module):
    """Per-patch U-Net decoder that maps a patch and its global condition to a denoised patch."""

    def __init__(self, in_channels, cond_hidden_size):
        super().__init__()

        # Encoder: 16x16 -> 8x8 -> 4x4 -> 2x2 -> 1x1
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.pool3 = nn.MaxPool2d(2, stride=2)

        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.pool4 = nn.MaxPool2d(2, stride=2)

        # Bottleneck: inject the global condition feature
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512 + cond_hidden_size, 512, kernel_size=1),
            nn.SiLU(),
        )

        # Decoder: 1x1 -> 2x2 -> 4x4 -> 8x8 -> 16x16
        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(512 + 512, 256, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        self.out_conv = nn.Conv2d(64, in_channels, kernel_size=1)

    def forward(self, x, c):
        # x: [B, C, 16, 16], c: [B, H_cond, 1, 1]
        enc1_out = self.enc1(x)
        p1_out = self.pool1(enc1_out)

        enc2_out = self.enc2(p1_out)
        p2_out = self.pool2(enc2_out)

        enc3_out = self.enc3(p2_out)
        p3_out = self.pool3(enc3_out)

        enc4_out = self.enc4(p3_out)
        p4_out = self.pool4(enc4_out)

        bottleneck_input = torch.cat([p4_out, c], dim=1)
        bottleneck_out = self.bottleneck(bottleneck_input)

        dec4_out = self.up4(bottleneck_out)
        dec4_out = torch.cat([dec4_out, enc4_out], dim=1)
        dec4_out = self.dec4(dec4_out)

        dec3_out = self.up3(dec4_out)
        dec3_out = torch.cat([dec3_out, enc3_out], dim=1)
        dec3_out = self.dec3(dec3_out)

        dec2_out = self.up2(dec3_out)
        dec2_out = torch.cat([dec2_out, enc2_out], dim=1)
        dec2_out = self.dec2(dec2_out)

        dec1_out = self.up1(dec2_out)
        dec1_out = torch.cat([dec1_out, enc1_out], dim=1)
        dec1_out = self.dec1(dec1_out)

        x_out = self.out_conv(dec1_out)
        return x_out


class DiP(nn.Module):
    def __init__(
            self,
            in_channels=4,
            num_groups=12,
            hidden_size=1152,
            hidden_size_x=64,
            nerf_mlpratio=4,
            num_blocks=18,
            num_cond_blocks=4,
            patch_size=2,
            num_classes=1000,
            learn_sigma=True,
            deep_supervision=0,
            weight_path=None,
            load_ema=False,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.num_blocks = num_blocks
        self.num_cond_blocks = num_cond_blocks
        self.patch_size = patch_size

        self.s_embedder = Embed(in_channels * patch_size ** 2, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes + 1, hidden_size)

        self.weight_path = weight_path
        self.load_ema = load_ema

        self.blocks = nn.ModuleList([
            FlattenDiTBlock(self.hidden_size, self.num_groups) for _ in range(self.num_cond_blocks)
        ])

        self.local_decoder = LocalDecoder(
            in_channels=in_channels,
            cond_hidden_size=hidden_size,
        )

        self.initialize_weights()
        self.precompute_pos = dict()

    def fetch_pos(self, height, width, device):
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)].to(device)
        pos = precompute_freqs_cis_2d(self.hidden_size // self.num_groups, height, width).to(device)
        self.precompute_pos[(height, width)] = pos
        return pos

    def initialize_weights(self):
        # Initialize patch embedding like nn.Linear
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)

        # Initialize label embedding table
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def forward(self, x, t, y, s=None, mask=None):
        B, C_orig, H, W = x.shape
        pos = self.fetch_pos(H // self.patch_size, W // self.patch_size, x.device)
        x = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        t = self.t_embedder(t.view(-1)).view(B, -1, self.hidden_size)
        y = self.y_embedder(y).view(B, 1, self.hidden_size)
        c = nn.functional.silu(t + y)

        if s is None:
            s = self.s_embedder(x)
            for i in range(self.num_cond_blocks):
                s = self.blocks[i](s, c, pos, mask)
            s = nn.functional.silu(t + s)

        _, T, D = s.shape
        s_global = s.contiguous().view(B * T, D, 1, 1).repeat(1, 1, self.patch_size // 16, self.patch_size // 16)
        x_raw_local = x.contiguous().view(B * T, -1).view(B * T, self.in_channels, self.patch_size, self.patch_size)

        x_denoised_patches = self.local_decoder(x_raw_local, s_global)

        x_out_flat = x_denoised_patches.view(
            B * T, self.in_channels * self.patch_size * self.patch_size
        ).view(B, T, -1).transpose(1, 2)
        x_final = torch.nn.functional.fold(
            x_out_flat,
            (H, W),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        return x_final
