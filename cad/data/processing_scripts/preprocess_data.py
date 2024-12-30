import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import argparse
import json
from collections import UserDict
from pathlib import Path

import numpy as np
import torch
import webdataset as wds
from diffusers import AutoencoderKL
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPModel, T5EncoderModel
from webdataset.autodecode import ImageHandler

from cad.utils.image_processing import CenterCrop

vae_image_transforms_512 = transforms.Compose(
    [
        CenterCrop(ratio="1:1"),
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
    ]
)

vae_image_transforms_256 = transforms.Compose(
    [
        CenterCrop(ratio="1:1"),
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
    ]
)


def dict_collate(batch):
    output_dict = {}
    if isinstance(batch[0], dict):
        for key in batch[0].keys():
            list_key = [d[key] for d in batch]
            if key != "json":
                output_dict[key] = dict_collate(list_key)
            else:
                output_dict[key] = list_key
        return output_dict
    elif isinstance(batch[0], Image.Image):
        return [img for img in batch]
    else:
        return torch.utils.data.dataloader.default_collate(batch)


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    # logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_tokenizer = AutoTokenizer.from_pretrained("facebook/metaclip-h14-fullcc2.5b")
clip_image_transform = CLIPImageProcessor.from_pretrained(
    "facebook/metaclip-h14-fullcc2.5b"
)
clip_model = CLIPModel.from_pretrained("facebook/metaclip-h14-fullcc2.5b")
clip_model = clip_model.to(device)

t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
t5_model = T5EncoderModel.from_pretrained("google/flan-t5-xl")
t5_model = t5_model.to(device)
t5_model.eval()

vae_model = AutoencoderKL.from_pretrained(
    "stabilityai/sdxl-vae",
    device="cuda:0",
).to(device)
vae_model.eval()


def add_clip_scores_and_embeddings(src, dest, batch_size=512):
    dataset = wds.DataPipeline(
        wds.SimpleShardList(str(src)),
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.rename(
            __key__="__key__",
            clip_image="jpg",
            vae_image_256="jpg",
            vae_image_512="jpg",
            image="jpg",
            txt="txt",
            json="json",
        ),
        wds.decode(
            ImageHandler("pilrgb", ["clip_image", "vae_image_256", "vae_image_512"])
        ),  # avoid encoding decoding jpeg for true
        wds.map_dict(
            clip_image=lambda x: x,
            vae_image_256=lambda x: vae_image_transforms_256(x),
            vae_image_512=lambda x: vae_image_transforms_512(x),
            image=lambda x: x,
            txt=lambda x: x,
            json=lambda x: x,
        ),
        wds.to_tuple(
            "__key__",
            "clip_image",
            "vae_image_256",
            "vae_image_512",
            "image",
            "txt",
            "json",
        ),
        wds.batched(batch_size),
    )
    loader = wds.WebLoader(dataset, num_workers=32, batch_size=None)
    with wds.TarWriter(str(dest)) as sink:
        for batch in tqdm(loader, total=10000 // batch_size):
            (
                keys,
                clip_image,
                vae_image_256,
                vae_image_512,
                orig_images,
                orig_text,
                json,
            ) = batch
            orig_text = [
                t.replace("<PERSON>", "") for t in orig_text
            ]  # remove <PERSON> tokens in captions
            clip_image = clip_image_transform(clip_image, return_tensors="pt")
            clip_text = clip_tokenizer(
                orig_text,
                truncation=True,
                padding="max_length",
                max_length=77,
                return_tensors="pt",
            )
            t5_text = t5_tokenizer(
                orig_text,
                return_tensors="pt",
                padding="longest",
                max_length=512,
                truncation=True,
            )
            with torch.no_grad():
                clip_image = {k: v.to(device) for k, v in clip_image.items()}
                clip_text = {k: v.to(device) for k, v in clip_text.items()}

                t5_text = {k: v.to(device) for k, v in t5_text.items()}
                vae_image_256 = vae_image_256.to(device)
                vae_image_512 = vae_image_512.to(device)
                clip_output = clip_model(
                    **clip_image, **clip_text, output_hidden_states=True
                )
                clip_scores = (
                    torch.diag(clip_output.logits_per_image).detach().cpu().numpy()
                )
                clip_text_embeddings = (
                    clip_output.text_model_output.hidden_states[-2]
                    .detach()
                    .cpu()
                    .numpy()
                )
                t5_embeddings = (
                    t5_model(**t5_text).last_hidden_state.detach().cpu().numpy()
                )
                vae_embeddings_256 = (
                    vae_model.encode(vae_image_256).latent_dist.mean.cpu().numpy()
                ) * vae_model.config.scaling_factor
                vae_embeddings_512 = (
                    vae_model.encode(vae_image_512).latent_dist.mean.cpu().numpy()
                ) * vae_model.config.scaling_factor
            for i in range(len(keys)):
                t5_embeddings_length = t5_text["attention_mask"][i].sum().item()
                json[i]["clip_score"] = clip_scores[i].item()
                sample = {
                    "__key__": keys[i],
                    "jpg": orig_images[i],
                    "txt": orig_text[i],
                    "json": json[i],
                    # "metaclip_h14_fullcc2.5b_embeddings.npy": clip_text_embeddings[i],
                    "flan_t5_xl_embeddings.npy": t5_embeddings[i][
                        :t5_embeddings_length
                    ],
                    "vae_embeddings_256.npy": vae_embeddings_256[i],
                    "vae_embeddings_512.npy": vae_embeddings_512[i],
                }
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
    batch_size = 16

    print(f"Loading {shard}")

    tar_name = shard.split(".")[0]

    src_shard = src / shard  # f"{{{tar_name}...{tar_name}}}.tar"

    print(f"Processing {src_shard} to {dest / shard}")
    add_clip_scores_and_embeddings(src_shard, dest / shard, batch_size)
