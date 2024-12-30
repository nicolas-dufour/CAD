import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5EncoderModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_embeddings(src, dest, tokenizer, model):
    with open(src, "r") as f:
        texts = f.readlines()
    metadata = []
    for i, text in enumerate(tqdm(texts)):
        text = text.strip()
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            embeddings = model(**tokens).last_hidden_state.detach().cpu().numpy()
            if embeddings.shape[0] == 1:
                embeddings = embeddings[0]
        file_name = f"{str(i).zfill(4)}.npy"
        np.save(dest / "flan_t5_xl_embeddings" / file_name, embeddings)
        metadata.append(
            {
                "file_name": f"{file_name}",
                "text": text,
            }
        )
    metadata = pd.DataFrame(metadata)
    metadata.to_csv(dest / "metadata.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="path to source files")
    parser.add_argument("--dest", help="path to destination files")
    args = parser.parse_args()

    src = Path(args.src)

    dest = Path(args.dest)
    dest.mkdir(exist_ok=True, parents=True)

    embeddings_path = dest / "flan_t5_xl_embeddings"
    embeddings_path.mkdir(exist_ok=True, parents=True)

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    model = T5EncoderModel.from_pretrained("google/flan-t5-xl")
    model = model.to(device)
    model.eval()

    add_embeddings(src, dest, tokenizer, model)
