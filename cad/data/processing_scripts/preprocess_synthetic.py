import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import argparse
from pathlib import Path

import torch
import webdataset as wds
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    CLIPImageProcessor,
    CLIPModel,
    LlavaForConditionalGeneration,
    T5EncoderModel,
)
from webdataset.autodecode import ImageHandler

from cad.utils.image_processing import CenterCrop


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

llava_model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/bakLlava-v1-hf",
    low_cpu_mem_usage=True,
    load_in_4bit=True,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ),
)
llava_processor = AutoProcessor.from_pretrained("llava-hf/bakLlava-v1-hf")


def caption_image(prompt, image, device=torch.device("cuda:0")):
    prompt = f"""SYSTEM:
    - You are LLaVA, a large language and vision assistant trained by UW Madison WAIV Lab.
    - You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.
    - You should follow the instructions carefully and explain your answers in detail.
    USER: <image>\n{prompt}\nASSISTANT:"""
    prompts = [prompt] * len(image)
    with torch.no_grad():
        inputs = llava_processor(text=prompts, images=image, return_tensors="pt").to(
            device, torch.float16
        )
        # Generate
        generate_ids = llava_model.generate(**inputs, max_length=512)
        results = llava_processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    return [
        result.replace(prompt.replace("<image>", " "), "").removeprefix(
            " The image features "
        )
        for result in results
    ]


def add_clip_scores_and_embeddings(src, dest, batch_size=512):
    dataset = wds.DataPipeline(
        wds.SimpleShardList(str(src)),
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.rename(
            __key__="__key__",
            clip_image="jpg",
            image="jpg",
            txt="txt",
            json="json",
            flan_t5_xl_embeddings="flan_t5_xl_embeddings.npy",
            vae_embeddings_256="vae_embeddings_256.npy",
            vae_embeddings_512="vae_embeddings_512.npy",
        ),
        wds.decode(
            ImageHandler("pilrgb", ["clip_image"])
        ),  # avoid encoding decoding jpeg for true
        wds.to_tuple(
            "__key__",
            "clip_image",
            "image",
            "txt",
            "json",
            "flan_t5_xl_embeddings",
            "vae_embeddings_256",
            "vae_embeddings_512",
        ),
        wds.batched(batch_size),
    )
    loader = wds.WebLoader(dataset, num_workers=32, batch_size=None)
    with wds.TarWriter(str(dest)) as sink:
        for batch in tqdm(loader, total=10000 // batch_size):
            (
                keys,
                clip_image,
                orig_images,
                orig_text,
                json,
                flan_t5_xl_embeddings,
                vae_embeddings_256,
                vae_embeddings_512,
            ) = batch

            prompt = "Describe this image and its style in a very detailed manner"
            synthetic_captions = caption_image(prompt, clip_image)
            clip_image = clip_image_transform(clip_image, return_tensors="pt")
            synthetic_clip_text = clip_tokenizer(
                synthetic_captions,
                truncation=True,
                padding="max_length",
                max_length=77,
                return_tensors="pt",
            )
            synthetic_t5_text = t5_tokenizer(
                synthetic_captions,
                return_tensors="pt",
                padding="longest",
                max_length=512,
                truncation=True,
            )
            with torch.no_grad():
                clip_image = {k: v.to(device) for k, v in clip_image.items()}
                synthetic_clip_text = {
                    k: v.to(device) for k, v in synthetic_clip_text.items()
                }

                synthetic_t5_text = {
                    k: v.to(device) for k, v in synthetic_t5_text.items()
                }
                clip_output = clip_model(
                    **clip_image, **synthetic_clip_text, output_hidden_states=True
                )
                synthetic_clip_scores = (
                    torch.diag(clip_output.logits_per_image).detach().cpu().numpy()
                )
                clip_text_embeddings = (
                    clip_output.text_model_output.hidden_states[-2]
                    .detach()
                    .cpu()
                    .numpy()
                )
                t5_embeddings = (
                    t5_model(**synthetic_t5_text)
                    .last_hidden_state.detach()
                    .cpu()
                    .numpy()
                )

            for i in range(len(keys)):
                t5_embeddings_length = (
                    synthetic_t5_text["attention_mask"][i].sum().item()
                )
                json[i]["synthetic_clip_score"] = synthetic_clip_scores[i].item()
                sample = {
                    "__key__": keys[i],
                    "jpg": orig_images[i],
                    "txt": orig_text[i],
                    "json": json[i],
                    "flan_t5_xl_embeddings.npy": flan_t5_xl_embeddings[i],
                    "synthetic_flan_t5_xl_embeddings.npy": t5_embeddings[i][
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
    batch_size = 32

    print(f"Loading {shard}")

    tar_name = shard.split(".")[0]

    src_shard = src / shard  # f"{{{tar_name}...{tar_name}}}.tar"

    print(f"Processing {src_shard} to {dest / shard}")
    add_clip_scores_and_embeddings(src_shard, dest / shard, batch_size)
