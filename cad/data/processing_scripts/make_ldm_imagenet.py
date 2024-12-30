import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from cad.utils.image_processing import CenterCrop

vae_image_transforms_256 = transforms.Compose(
    [
        CenterCrop(ratio="1:1"),
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
    ]
)


class ImageDatasetWithPath(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.paths = list(root.glob("*.JPEG"))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.paths[idx].stem


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae_model = AutoencoderKL.from_pretrained(
    "stabilityai/sdxl-vae",
    device="cuda:0",
).to(device)
vae_model.eval()


def create_ldm_embedding(src, dest, batch_size=32):
    dataset = ImageDatasetWithPath(src, transform=vae_image_transforms_256)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )
    for batch in tqdm(dataloader):
        img, paths = batch
        img = img.to(device)
        with torch.no_grad():
            z = (
                vae_model.encode(img).latent_dist.mean.cpu().numpy()
                * vae_model.config.scaling_factor
            )
        for file, sample in zip(paths, z):
            np.save(dest / f"{file}.npy", sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="path to source files")
    parser.add_argument("--dest", help="path to destination files")
    args = parser.parse_args()

    src = Path(args.src)
    list_of_classes = os.listdir(src)
    for folder in tqdm(list_of_classes):
        src_shard = src / folder
        dest_shard = Path(args.dest) / folder
        dest_shard.mkdir(parents=True, exist_ok=True)
        create_ldm_embedding(src_shard, dest_shard)
