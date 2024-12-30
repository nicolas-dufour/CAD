import os
import sys

root_path = os.path.abspath("..")

if not root_path in sys.path:
    sys.path.append(root_path)
from copy import deepcopy
from pathlib import Path

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from hydra.core.global_hydra import GlobalHydra
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from callbacks.log_images import TextCondPromptBed
from data.datamodule import dict_collate_and_pad
from cad.models.diffusion import DiffusionModule
from cad.utils.image_processing import remap_image_torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg):
    model = DiffusionModule.load_from_checkpoint(
        Path(cfg.checkpoints.dirpath) / Path("last.ckpt"),
        cfg=cfg.model,
        strict=False,
    )
    model = model.to(device)
    model.eval()

    generator = torch.Generator(device="cpu").manual_seed(3407)
    dataloader = DataLoader(
        TextCondPromptBed(
            Path(cfg.data_dir) / "text_prompt_testbed",
            4,
            "flan_t5_xl",
            generator=generator,
            shape=(4, 32, 32),
        ),
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=dict_collate_and_pad(["flan_t5_xl"], max_length=128),
    )

    for i, batch in enumerate(dataloader):
        x_k = batch["x_N"].to(device)
        cond = {
            "flan_t5_xl_embeddings": batch["flan_t5_xl_embeddings"].to(device),
            "flan_t5_xl_mask": batch["flan_t5_xl_mask"].to(device),
        }
        with torch.no_grad():
            images = model.sample(
                x_N=x_k,
                cond=cond,
                cfg=7,
                stage="val",
            )
        images = rearrange(images, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=2)
        images = images.cpu().numpy()
        Image.fromarray(images).save(f"sample_{i}.png")


if __name__ == "__main__":
    main()
