import argparse
from pathlib import Path
import csv

import open_clip
import torch
import webdataset as wds
from tqdm import tqdm
from PIL import Image
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_csv_to_dict(file_path):
    metadata_dict = {}
    with open(file_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            key = row["key"]
            caption = row["caption"]
            metadata_dict[key] = caption
    return metadata_dict


# Load the CSV file
csv_file_path = "cad/datasets/coco_10k/global_metadata.csv"
prompts = load_csv_to_dict(csv_file_path)

with wds.ShardWriter(
    "cad/datasets/coco_10k/webdataset/%03d.tar", maxcount=30000, maxsize=15000000000
) as sink:
    for key, caption in tqdm(prompts.items()):
        img_path = Path("cad/datasets/coco_10k/images/") / f"{key}.jpg"
        try:
            image = Image.open(img_path).convert("RGB")
            sink.write(
                {
                    "__key__": key,
                    "txt": caption,
                    "jpg": image,
                    "json": json.dumps(
                        {
                            "caption": caption,
                            "height": image.height,
                            "width": image.width,
                            "key": key,
                        }
                    ),
                }
            )
        except FileNotFoundError:
            print(f"Image not found: {img_path}")
        except Exception as e:
            print(f"Error processing {key}: {str(e)}")

print(f"Processed {len(prompts)} items.")
