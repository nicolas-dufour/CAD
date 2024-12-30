import json
from pathlib import Path

import pandas as pd
import webdataset as wds
from tqdm import tqdm


def main(src, dest):
    all_shard = [str(tar) for tar in list(Path(src).glob("*.tar"))]
    webdataset = wds.WebDataset(all_shard).to_tuple(
        "__key__",
        "jpg",
        "txt",
        "label.json",
        "vae_embeddings_mean_256.npy",
        "vae_embeddings_std_256.npy",
        "flan_t5_xl_embeddings.npy",
    )
    dest_dir = Path(dest)
    dest_dir.mkdir(exist_ok=True, parents=True)
    images_dir = dest_dir / "images"
    images_dir.mkdir(exist_ok=True, parents=True)

    vae_embeddings_256_mean_dir = dest_dir / "vae_embeddings_mean_256"
    vae_embeddings_256_mean_dir.mkdir(exist_ok=True, parents=True)

    vae_embeddings_256_std_dir = dest_dir / "vae_embeddings_std_256"
    vae_embeddings_256_std_dir.mkdir(exist_ok=True, parents=True)

    flan_t5_dir = dest_dir / "flan_t5_xl_embeddings"
    flan_t5_dir.mkdir(exist_ok=True, parents=True)

    global_metadata = []
    for (
        key,
        image,
        text,
        metadata,
        vae_embeddings_mean_256,
        vae_embeddings_std_256,
        flan_t5_xl_embeddings,
    ) in tqdm(webdataset):
        # decode metadata
        metadata = json.loads(metadata)
        with open(images_dir / f"{key}.jpg", "wb") as f:
            f.write(image)
        with open(vae_embeddings_256_mean_dir / f"{key}.npy", "wb") as f:
            f.write(vae_embeddings_mean_256)
        with open(vae_embeddings_256_std_dir / f"{key}.npy", "wb") as f:
            f.write(vae_embeddings_std_256)
        with open(flan_t5_dir / f"{key}.npy", "wb") as f:
            f.write(flan_t5_xl_embeddings)

        global_metadata.append(
            {
                "key": str(key),
                "caption": text,
                "label": metadata["id"],
                "class_name": metadata["class_name"],
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
