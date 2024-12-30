from functools import partial

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from ..positional_embeddings import FourierEmbedding, PositionalEmbedding
from .utils import Resample, init_xavier_uniform, init_xavier_uniform_mha


class DiffusionUNet(nn.Module):
    def __init__(
        self,
        img_resolution,
        in_channels,
        out_channels,
        encoder_type="standard",  ## DDPM++ uses "standard", NCSN++ uses "residual"
        decoder_type="standard",  ## DDPM++ uses "standard", NCSN++ uses "residual"
        model_channels=128,
        num_blocks=4,
        attn_resolutions=[16],
        channel_mults=(1, 2, 2, 2),
        channel_mult_emb=4,
        dropout=0.1,
        resample_filter=[1, 1],  # DDPM uses [1, 1] whereas Song et al. use [1, 3, 3, 1]
        noise_embedding_type="positional",
        channel_mult_noise=1,
        label_dim=0,
        label_dropout=0,
    ):
        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = init_xavier_uniform
        init_zeros = partial(init_xavier_uniform, weight_gain=1e-5)
        init_attn = partial(
            init_xavier_uniform_mha, qkv_weight_gain=np.sqrt(0.2), out_weight_gain=1e-5
        )
        resnet_blocks_kwargs = {
            "emb_channels": emb_channels,
            "num_heads": 1,
            "dropout": dropout,
            "skip_scale": np.sqrt(0.5),
            "eps": 1e-6,
            "resample_filter": resample_filter,
            "resample_proj": True,
            "adaptive_scale": False,
            "init": init,
            "init_zeros": init_zeros,
            "init_attn": init_attn,
        }

        self.map_noise = (
            PositionalEmbedding(num_channels=noise_channels, endpoint=True)
            if noise_embedding_type == "positional"
            else FourierEmbedding(num_channels=noise_channels)
        )
        self.map_label = (
            init(nn.Linear(label_dim, noise_channels)) if label_dim > 0 else None
        )
        self.map_layer = nn.Sequential(
            init(nn.Linear(noise_channels, emb_channels)),
            nn.SiLU(),
            init(nn.Linear(emb_channels, emb_channels)),
            nn.SiLU(),
        )

        self.encoder = nn.ModuleDict()
        c_out = in_channels
        c_aux = in_channels

        for level, mult in enumerate(channel_mults):
            res = img_resolution >> level
            if level == 0:
                c_in = c_out
                c_out = model_channels
                self.encoder[f"{res}x{res}_conv"] = init(
                    nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)
                )
            else:
                self.encoder[f"{res}x{res}_down"] = ResNetBlock(
                    in_channels=c_out,
                    out_channels=c_out,
                    down=True,
                    **resnet_blocks_kwargs,
                )
                if encoder_type == "skip":
                    self.encoder[f"{res}x{res}_aux_down"] = Resample(
                        c_aux, down=True, resample_filter=resample_filter
                    )

                    self.encoder[f"{res}x{res}_aux_skip"] = init(
                        nn.Conv2d(c_aux, c_out, kernel_size=1)
                    )
                if encoder_type == "residual":
                    self.encoder[f"{res}x{res}_aux_residual"] = nn.Sequential(
                        Resample(c_aux, down=True, resample_filter=resample_filter),
                        init(nn.Conv2d(c_aux, c_out, kernel_size=3, padding=1)),
                    )
                    c_aux = c_out
            for i in range(num_blocks):
                c_in = c_out
                c_out = model_channels * mult
                attn = res in attn_resolutions
                self.encoder[f"{res}x{res}_block{i}"] = ResNetBlock(
                    c_in, c_out, attention=attn, **resnet_blocks_kwargs
                )
        skips = [
            block.out_channels
            for name, block in self.encoder.items()
            if "aux" not in name
        ]

        self.decoder = nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mults))):
            res = img_resolution >> level
            if level == len(channel_mults) - 1:
                self.decoder[f"{res}x{res}_in_0"] = ResNetBlock(
                    c_out, c_out, attention=True, **resnet_blocks_kwargs
                )
                self.decoder[f"{res}x{res}_in_1"] = ResNetBlock(
                    c_out, c_out, **resnet_blocks_kwargs
                )
            else:
                self.decoder[f"{res}x{res}_up"] = ResNetBlock(
                    c_out, c_out, up=True, **resnet_blocks_kwargs
                )
            for i in range(num_blocks + 1):
                c_in = c_out + skips.pop()
                c_out = model_channels * mult
                attn = res in attn_resolutions and i == num_blocks
                self.decoder[f"{res}x{res}_block{i}"] = ResNetBlock(
                    c_in, c_out, attention=attn, **resnet_blocks_kwargs
                )
            if decoder_type == "skip" or level == 0:
                if decoder_type == "skip" and level < len(channel_mults) - 1:
                    self.decoder[f"{res}x{res}_aux_up"] = Resample(
                        out_channels, up=True, resample_filter=resample_filter
                    )
                self.decoder[f"{res}x{res}_aux_norm"] = nn.GroupNorm(
                    min(32, c_out // 4), c_out, eps=1e-6
                )
                self.decoder[f"{res}x{res}_aux_conv"] = init_zeros(
                    nn.Conv2d(
                        c_out,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                    )
                )

    def forward(self, x, noise_label, conditioning=None, augment_labels=None):
        ### Condnitional mapping of noise and labels
        emb = self.map_noise(noise_label)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        if self.map_label is not None:
            tmp = conditioning
            if self.training and self.label_dropout > 0:
                tmp = tmp * (
                    torch.rand(x.shape[0], 1, device=x.device) > self.label_dropout
                ).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        emb = self.map_layer(emb)

        ### Encoder
        skips = []
        aux = x
        for name, block in self.encoder.items():
            if "aux_down" in name:
                aux = block(aux)
            elif "aux_skip" in name:
                x = skips[-1] = x + block(aux)
            elif "aux_residual" in name:
                x = skips[-1] = (x + block(aux)) / np.sqrt(2)
                aux = x
            else:
                x = block(x, emb) if isinstance(block, ResNetBlock) else block(x)
                skips.append(x)

        ### Decoder
        aux = None
        tmp = None

        for name, block in self.decoder.items():
            if "aux_up" in name:
                aux = block(aux)
            elif "aux_norm" in name:
                tmp = block(x)
            elif "aux_conv" in name:
                tmp = block(nn.functional.silu(tmp))
                aux = tmp if aux is None else aux + tmp
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        return aux


class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        up=False,
        down=False,
        adaptive_scale=True,
        skip_scale=1.0,
        num_groups=32,
        eps=1e-5,
        dropout=0.0,
        attention=False,
        num_heads=None,
        channels_per_head=64,
        resample_filter=[1, 1],
        resample_proj=0,
        init=init_xavier_uniform,
        init_zeros=init_xavier_uniform,
        init_attn=init_xavier_uniform_mha,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_1 = nn.Sequential(
            nn.GroupNorm(min(num_groups, in_channels // 4), in_channels, eps=eps),
            nn.SiLU(),
            Resample(in_channels, up=up, down=down, resample_filter=resample_filter),
            init(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)),
        )
        # Time embedding
        self.timestep_embedder = init(
            nn.Linear(
                emb_channels, 2 * out_channels if adaptive_scale else out_channels
            )
        )
        self.adaptive_scale = adaptive_scale
        self.skip_scale = skip_scale
        self.norm_2 = nn.GroupNorm(
            min(num_groups, out_channels // 4), out_channels, eps=eps
        )
        self.conv_2 = nn.Sequential(
            nn.SiLU(),
            nn.Dropout2d(dropout),
            init_zeros(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)),
        )

        # Skip path
        skip = []
        if up or down:
            skip.append(
                Resample(in_channels, up=up, down=down, resample_filter=resample_filter)
            )
        if out_channels != in_channels or resample_proj:
            skip.append(
                init(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))
            )
        self.skip_path = nn.Sequential(*skip) if len(skip) > 0 else nn.Identity()

        ## Attention
        self.num_heads = int(
            0
            if not attention
            else num_heads
            if num_heads is not None
            else out_channels // channels_per_head
        )
        if self.num_heads:
            self.norm_3 = nn.GroupNorm(
                min(num_groups, out_channels // 4), out_channels, eps=eps
            )
            self.attn = init_attn(nn.MultiheadAttention(out_channels, self.num_heads))

    def forward(self, x, temp):
        inital_x = x
        x = self.conv_1(x)
        t_embed = self.timestep_embedder(temp).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = t_embed.chunk(2, dim=1)
            x = torch.addcmul(shift, self.norm_2(x), scale + 1)
        else:
            x = self.norm_2(x + t_embed)
        x = self.conv_2(x)
        x = x + self.skip_path(inital_x)
        x = x * self.skip_scale
        if self.num_heads:
            x_shape = x.shape
            x_flatten = rearrange(x, "b c h w -> (b h w) c")
            a, _ = self.attn(x_flatten, x_flatten, x_flatten)
            a = rearrange(a, "(b h w) c -> b c h w", h=x_shape[2], w=x_shape[3])
            x = x + a
            x = x * self.skip_scale
        return x
