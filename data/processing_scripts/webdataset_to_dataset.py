import webdataset as wds
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json


def main(src, dest):
    all_shard = [str(tar) for tar in list(Path(src).glob("*.tar"))]
    print(all_shard)
    webdataset = wds.WebDataset(all_shard).to_tuple(
        "__key__",
        "jpg",
        "txt",
        "json",
        "flan_t5_xl_embeddings.npy",
        "vae_embeddings_256.npy",
        "vae_embeddings_512.npy",
    )
    dest_dir = Path(dest)
    dest_dir.mkdir(exist_ok=True, parents=True)
    images_dir = dest_dir / "images"
    images_dir.mkdir(exist_ok=True, parents=True)

    flan_t5_xl_embeddings_dir = dest_dir / "flan_t5_xl_embeddings"
    flan_t5_xl_embeddings_dir.mkdir(exist_ok=True, parents=True)

    vae_embeddings_256_dir = dest_dir / "vae_embeddings_256"
    vae_embeddings_256_dir.mkdir(exist_ok=True, parents=True)

    vae_embeddings_512_dir = dest_dir / "vae_embeddings_512"
    vae_embeddings_512_dir.mkdir(exist_ok=True, parents=True)

    global_metadata = []
    for (
        key,
        image,
        text,
        metadata,
        flan_t5_xl_embeddings,
        vae_embeddings_256,
        vae_embeddings_512,
    ) in tqdm(webdataset):
        # decode metadata
        metadata = json.loads(metadata)
        with open(images_dir / f"{key}.jpg", "wb") as f:
            f.write(image)
        with open(flan_t5_xl_embeddings_dir / f"{key}.npy", "wb") as f:
            f.write(flan_t5_xl_embeddings)
        with open(vae_embeddings_256_dir / f"{key}.npy", "wb") as f:
            f.write(vae_embeddings_256)
        with open(vae_embeddings_512_dir / f"{key}.npy", "wb") as f:
            f.write(vae_embeddings_512)
        # Append to dataframe
        global_metadata.append(
            {
                "key": str(key),
                "caption": text,
                "url": metadata["url"],
                "key": metadata["key"],
                "width": metadata["width"],
                "height": metadata["height"],
                "original_width": metadata["original_width"],
                "original_height": metadata["original_height"],
                "clip_score": metadata["clip_score"],
            }
        )
    global_metadata = pd.DataFrame(global_metadata)
    global_metadata["key"] = global_metadata["key"].astype(str)
    global_metadata = global_metadata.reset_index(drop=True)
    global_metadata.to_csv(dest_dir / "global_metadata.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="path to source files")
    parser.add_argument("--dest", help="path to destination files")
    args = parser.parse_args()

    main(args.src, args.dest)
