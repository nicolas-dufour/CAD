import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
from pathlib import Path

import torch
import webdataset as wds
from diffusers import AutoencoderKL
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer, T5EncoderModel

from cad.utils.image_processing import CenterCrop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_transforms = transforms.Compose(
    [
        CenterCrop(ratio="1:1"),
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
    ]
)


def add_embeddings(src, dest, model, batch_size=512, has_t5=True):
    if has_t5:
        keys_wd = [
            "__key__",
            "jpg",
            "jpg",
            "txt",
            "json",
            "flan_t5_xl_embeddings.npy",
        ]
        functions = [
            lambda x: x,
            lambda x: x,
            image_transforms,
            lambda x: x,
            lambda x: x,
            lambda x: x,
        ]
    else:
        keys_wd = ["__key__", "jpg", "jpg", "txt", "json"]
        functions = [
            lambda x: x,
            lambda x: x,
            image_transforms,
            lambda x: x,
            lambda x: x,
        ]
    dataset = wds.DataPipeline(
        wds.SimpleShardList(str(src)),
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.decode("pilrgb"),
        wds.to_tuple(
            *keys_wd,
        ),
        wds.map_tuple(
            *functions,
        ),
        wds.batched(batch_size),
    )
    loader = wds.WebLoader(dataset, num_workers=32, batch_size=None)
    with wds.TarWriter(str(dest)) as sink:
        for batch in tqdm(loader, total=10000 // batch_size):
            if has_t5:
                keys, images_orig, images, text, json, t5_embeddings = batch
            else:
                keys, images_orig, images, text, json = batch
            with torch.no_grad():
                embeddings = (
                    model.encode(images.to(device)).latent_dist.mean.cpu().numpy()
                )
            for i in range(len(keys)):
                sample = {
                    "__key__": keys[i],
                    "jpg": images_orig[i],
                    "txt": text[i],
                    "json": json[i],
                    "vae_embeddings.npy": embeddings[i],
                }
                if has_t5:
                    sample["flan_t5_xl_embeddings.npy"] = t5_embeddings[i]
                sink.write(sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="path to source files")
    parser.add_argument("--dest", help="path to destination files")
    parser.add_argument("--shard_id", help="shard id")
    args = parser.parse_args()

    src = Path(args.src)
    list_of_shards = list(src.glob("*.tar"))
    list_of_shards.sort()
    shard = str(list_of_shards[int(args.shard_id)]).split("/")[-1]
    dest = Path(args.dest)
    dest.mkdir(exist_ok=True, parents=True)

    model = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", device="cuda:0").to(
        device
    )
    model.eval()

    batch_size = 32

    print(f"Loading {shard}")

    tar_name = shard.split(".")[0]

    src_shard = src / shard  # f"{{{tar_name}...{tar_name}}}.tar"

    print(f"Processing {src_shard} to {dest / shard}")
    add_embeddings(src_shard, dest / shard, model, batch_size, has_t5=False)
