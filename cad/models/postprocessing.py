import torch
import torch.nn as nn
from consistencydecoder import ConsistencyDecoder

from cad.utils.image_processing import remap_image_torch


class SD1_5VAEPostProcessing(nn.Module):
    def __init__(self, channel_wise_normalisation=False):
        super().__init__()
        if channel_wise_normalisation:
            scale = 0.5 / torch.tensor([4.17, 4.62, 3.71, 3.28])
            bias = -torch.tensor([5.81, 3.25, 0.12, -2.15]) * scale
        else:
            scale = torch.tensor([0.18215, 0.18215, 0.18215, 0.18215])
            bias = torch.tensor([0.0, 0.0, 0.0, 0.0])
        scale = scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        bias = bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.register_buffer("scale", nn.Parameter(scale))
        self.register_buffer("bias", nn.Parameter(bias))

    def forward(self, x):
        x = (x - self.bias) / self.scale
        return x


class SD1_5VAEDecoderPostProcessing(SD1_5VAEPostProcessing):
    def __init__(self, vae, channel_wise_normalisation=False):
        super().__init__(channel_wise_normalisation=channel_wise_normalisation)
        self.vae = vae
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = super().forward(x)
        with torch.no_grad():
            return remap_image_torch(self.vae.decode(x).sample.detach())


class SD1_5VAEConsistencyProcessing(SD1_5VAEPostProcessing):
    def __init__(self, channel_wise_normalisation=False):
        super().__init__(channel_wise_normalisation=channel_wise_normalisation)
        self.consistency = ConsistencyDecoder()

    def forward(self, x):
        x = super().forward(x)
        with torch.no_grad():
            return remap_image_torch(self.consistency(x).detach())
