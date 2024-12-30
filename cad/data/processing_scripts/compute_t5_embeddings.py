import argparse
from pathlib import Path

import torch
import webdataset as wds
from tqdm import tqdm
from transformers import AutoTokenizer, T5EncoderModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_embeddings(src, dest, tokenizer, model, batch_size=512):
    dataset = wds.DataPipeline(
        wds.SimpleShardList(str(src)),
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.decode("pilrgb"),
        wds.to_tuple("__key__", "jpg", "txt", "json"),
        wds.batched(batch_size),
    )
    loader = wds.WebLoader(dataset, num_workers=32, batch_size=None)
    with wds.TarWriter(str(dest)) as sink:
        for keys, images, text, json in tqdm(loader, total=10000 // batch_size):
            tokens = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=77,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                embeddings = model(**tokens).last_hidden_state.detach().cpu().numpy()
            for i in range(len(keys)):
                sample = {
                    "__key__": keys[i],
                    "jpg": images[i],
                    "txt": text[i],
                    "json": json[i],
                    "flan_t5_xl_embeddings.npy": embeddings[i],
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

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    model = T5EncoderModel.from_pretrained("google/flan-t5-xl")
    model = model.to(device)
    model.eval()

    batch_size = 1024

    print(f"Loading {shard}")

    tar_name = shard.split(".")[0]

    src_shard = src / shard  # f"{{{tar_name}...{tar_name}}}.tar"

    print(f"Processing {src_shard} to {dest / shard}")
    add_embeddings(src_shard, dest / shard, tokenizer, model, batch_size)
