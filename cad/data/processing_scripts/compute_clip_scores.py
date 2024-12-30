import argparse
from pathlib import Path

import open_clip
import torch
import webdataset as wds
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_clip_scores_and_embeddings(
    src, dest, clip_processor, clip_model, batch_size=512
):
    dataset = wds.DataPipeline(
        wds.SimpleShardList(str(src)),
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.decode("pilrgb"),
        wds.to_tuple("__key__", "jpg", "txt", "jpg", "txt", "json"),
        wds.map_tuple(
            lambda x: x,
            clip_processor,
            open_clip.tokenize,
            lambda x: x,
            lambda x: x,
            lambda x: x,
        ),
        wds.batched(batch_size),
    )
    loader = wds.WebLoader(dataset, num_workers=32, batch_size=None)
    with wds.TarWriter(str(dest)) as sink:
        for keys, images, text, orig_images, orig_text, json in tqdm(
            loader, total=10000 // batch_size
        ):
            images = images.to(device)
            text = text.to(device).squeeze(1)
            with torch.no_grad():
                image_features = clip_model.visual(images)
                cast_dtype = clip_model.transformer.get_cast_dtype()
                text_features = clip_model.token_embedding(text).to(
                    cast_dtype
                )  # [batch_size, n_ctx, d_model]

                text_features = text_features + clip_model.positional_embedding.to(
                    cast_dtype
                )
                text_features = text_features.permute(1, 0, 2)  # NLD -> LND
                text_tokens = clip_model.transformer(
                    text_features, attn_mask=clip_model.attn_mask
                )
                text_tokens = text_tokens.permute(1, 0, 2)  # LND -> NLD
                text_features = clip_model.ln_final(
                    text_tokens
                )  # [batch_size, n_ctx, transformer.width]
                # take features from the eot embedding (eot_token is the highest number in each sequence)
                text_features = (
                    text_features[
                        torch.arange(text_features.shape[0]), text.argmax(dim=-1)
                    ]
                    @ clip_model.text_projection
                )
                clip_scores = torch.nn.functional.cosine_similarity(
                    image_features, text_features, dim=-1
                )
            clip_scores = clip_scores.detach().cpu().numpy()
            for i in range(len(keys)):
                json[i]["clip_score"] = clip_scores[i].item()
                sample = {
                    "__key__": keys[i],
                    "jpg": orig_images[i],
                    "txt": orig_text[i],
                    "json": json[i],
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

    clip_model, _, clip_processor = open_clip.create_model_and_transforms(
        "ViT-H-14-quickgelu", pretrained="metaclip_fullcc"
    )
    clip_model = clip_model.eval().to(device)
    batch_size = 512

    print(f"Loading {shard}")

    tar_name = shard.split(".")[0]

    src_shard = src / shard  # f"{{{tar_name}...{tar_name}}}.tar"

    print(f"Processing {src_shard} to {dest / shard}")
    add_clip_scores_and_embeddings(
        src_shard, dest / shard, clip_processor, clip_model, batch_size
    )


# TODO
# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
# model = T5EncoderModel.from_pretrained("google/flan-t5-xl")
# model = model.to(device)
# model.eval()
# tokens = tokenizer(
#     "ah this is nice",
#     truncation=True,
#     padding="max_length",
#     max_length=tokenizer.model_max_length,
#     return_tensors="pt",
# ).to(device)
# model(**tokens).last_hidden_state
