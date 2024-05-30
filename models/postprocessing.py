from utils.image_processing import remap_image_torch
from diffusers import AutoencoderKL
import torch.nn as nn
import torch


class SD1_5VAEPostProcessing(nn.Module):
    def __init__(self):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5", device="cuda:0", subfolder="vae"
        )
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            return remap_image_torch(
                self.vae.decode(x * 1 / self.vae.config.scaling_factor).sample.detach()
            )
