import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import argparse
from pathlib import Path

import torch
import webdataset as wds
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from webdataset.autodecode import ImageHandler

from cad.utils.aesthetic_scorer import AestheticScorer


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
aesthetic_scorer = AestheticScorer().to(device)


def add_aesthetics(src, dest, batch_size=512):
    dataset = wds.DataPipeline(
        wds.SimpleShardList(str(src)),
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.rename(
            __key__="__key__",
            aesthetic_img="jpg",
            vae_embeddings_256="vae_embeddings_256.npy",
            vae_embeddings_512="vae_embeddings_512.npy",
            flan_t5_xl_embeddings="flan_t5_xl_embeddings.npy",
            image="jpg",
            txt="txt",
            json="json",
        ),
        wds.decode(
            ImageHandler("pilrgb", ["aesthetic_img"])
        ),  # avoid encoding decoding jpeg for true
        wds.to_tuple(
            "__key__",
            "aesthetic_img",
            "vae_embeddings_256",
            "vae_embeddings_512",
            "flan_t5_xl_embeddings",
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
                aesthetic_img,
                vae_embeddings_256,
                vae_embeddings_512,
                flan_t5_xl_embeddings,
                orig_images,
                orig_text,
                json,
            ) = batch
            with torch.no_grad():
                aesthetic_scores = aesthetic_scorer(aesthetic_img)
                for i in range(len(keys)):
                    json[i]["aesthetic_score"] = aesthetic_scores[i].item()
                    sample = {
                        "__key__": keys[i],
                        "jpg": orig_images[i],
                        "txt": orig_text[i],
                        "json": json[i],
                        # "metaclip_h14_fullcc2.5b_embeddings.npy": clip_text_embeddings[i],
                        "flan_t5_xl_embeddings.npy": flan_t5_xl_embeddings[i],
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
    print(list_of_shards)
    shard = str(list_of_shards[int(args.shard_id)]).split("/")[-1]
    dest = Path(args.dest)
    dest.mkdir(exist_ok=True, parents=True)
    batch_size = 16

    print(f"Loading {shard}")

    tar_name = shard.split(".")[0]

    src_shard = src / shard  # f"{{{tar_name}...{tar_name}}}.tar"

    print(f"Processing {src_shard} to {dest / shard}")
    add_aesthetics(src_shard, dest / shard, batch_size)
