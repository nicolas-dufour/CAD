from __future__ import annotations

import os
import random
import time

import gradio as gr
import numpy as np
import PIL.Image
import torch

import torch

import os
import torch
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor
import uuid

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hydra
from hydra.core.global_hydra import GlobalHydra

from models.diffusion import DiffusionModule
from PIL import Image
from omegaconf import OmegaConf

DESCRIPTION = """# CAD demo
"""
if torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CUDA 😀</p>"
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    DESCRIPTION += "\n<p>Running on XPU 🤓</p>"
else:
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES") == "1"
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "768"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE") == "1"


"""
   Operation System Options:
      If you are using MacOS, please set the following (device="mps") ;
      If you are using Linux & Windows with Nvidia GPU, please set the device="cuda";
      If you are using Linux & Windows with Intel Arc GPU, please set the device="xpu";
"""
# device = "mps"    # MacOS
# device = "xpu"    # Intel Arc GPU
device = "cuda"  # Linux & Windows


"""
   DTYPE Options:
      To reduce GPU memory you can set "DTYPE=torch.float16",
      but image quality might be compromised
"""
DTYPE = (
    torch.float16
)  # torch.float16 works as well, but pictures seem to be a bit worse

models_overrides = {
    # "CC12m_LAION_Aesthetics_6_RIN_ldm_256_small": "cc12m_256_rin_small_ldm",
    "CC12m_LAION_Aesthetics_6_RIN_ldm_256_small_cad": {
        "overrides": "cc12m_256_rin_small_ldm_cad"
    },
    "CC12m_LAION_Aesthetics_6_RIN_ldm_256_small_cad_legacy": {
        "overrides": "cc12m_256_rin_small_ldm_cad",
        "model.network.latents_dim": 768,
    },
    "CC12m_LAION_Aesthetics_6_RIN_ldm_256_small_cad_text_swap": {
        "overrides": "cc12m_256_rin_small_ldm_cad"
    },
    # "CC12m_LAION_Aesthetics_6_RIN_ldm_512_small_cad": "cc12m_512_rin_small_ldm_cad",
    "CC12m_LAION_Aesthetics_6_RIN_ldm_256_small_filtered": {
        "overrides": "cc12m_256_rin_small_ldm_filtered"
    },
    # "CC12m_LAION_Aesthetics_6_RIN_ldm_256_small_weighted": "cc12m_256_rin_small_ldm_weighted",
}

MODELS = {}


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def save_image(img, profile: gr.OAuthProfile | None, metadata: dict, root_path="./"):
    unique_name = str(uuid.uuid4()) + ".png"
    unique_name = os.path.join(root_path, unique_name)
    img.save(unique_name)
    # gr_user_history.save_image(label=metadata["prompt"], image=img, profile=profile, metadata=metadata)
    return unique_name


def save_images(image_array, profile: gr.OAuthProfile | None, metadata: dict):
    paths = []
    root_path = "./demos/images/"
    os.makedirs(root_path, exist_ok=True)
    with ThreadPoolExecutor() as executor:
        paths = list(
            executor.map(
                save_image,
                image_array,
                [profile] * len(image_array),
                [metadata] * len(image_array),
                [root_path] * len(image_array),
            )
        )
    return paths


def load_model(model_name: str) -> DiffusionModule:
    if model_name not in MODELS:
        ckpt_name = model_name
        override = models_overrides[model_name]
        GlobalHydra.instance().clear()
        hydra.initialize(version_base=None, config_path=f"../configs")
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                "root_dir=..",
                f"experiment_name={ckpt_name}",
                "model/precomputed_text_embeddings='no'",
                *[f"{k}={v}" for k, v in override.items()],
            ],
        )
        model = DiffusionModule.load_from_checkpoint(
            f"checkpoints/{ckpt_name}/last.ckpt",
            strict=False,
            cfg=cfg.model,
        ).to(device)

        model.eval()
        MODELS[model_name] = model
    return model_name


def generate(
    model_name: str,
    prompt: str,
    seed: int = 0,
    guidance_scale: float = 8.0,
    num_inference_steps: int = 4,
    num_images: int = 4,
    randomize_seed: bool = False,
    confidence_value: float = 1.0,
    unconfidence_value: float = 0.0,
    uncond_prompt: str = "",
    progress=gr.Progress(track_tqdm=True),
    profile: gr.OAuthProfile | None = None,
) -> PIL.Image.Image:
    uncond_prompt = None if uncond_prompt == "" else uncond_prompt
    model = MODELS[model_name]
    seed = randomize_seed_fn(seed, randomize_seed)
    torch.manual_seed(seed)
    images = model.sample(
        batch_size=num_images,
        shape=(4, 32, 32) if "256" in model_name else (4, 64, 64),
        cond=prompt,
        cfg=guidance_scale,
        num_steps=num_inference_steps,
        confidence_value=confidence_value,
        unconfidence_value=unconfidence_value,
        unconfident_prompt=uncond_prompt,
        stage="val",
    )
    images = images.cpu().numpy().transpose(0, 2, 3, 1)

    results = [Image.fromarray(image) for image in images]

    paths = save_images(
        results,
        profile,
        metadata={
            "prompt": prompt,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
        },
    )
    return paths, seed


def generate_comparaison(
    prompt: str,
    seed: int = 0,
    guidance_scale: float = 8.0,
    num_inference_steps: int = 4,
    num_images: int = 4,
    randomize_seed: bool = False,
    confidence_value: float = 1.0,
    unconfidence_value: float = 0.0,
    uncond_prompt_others: str = "",
    uncond_prompt_cad: str = "",
    progress=gr.Progress(track_tqdm=True),
    profile: gr.OAuthProfile | None = None,
):
    results = []
    models_keys = [
        "CC12m_LAION_Aesthetics_6_RIN_ldm_256_small_cad_legacy",
        "CC12m_LAION_Aesthetics_6_RIN_ldm_256_small_filtered",
        "CC12m_LAION_Aesthetics_6_RIN_ldm_256_small_cad",
        "CC12m_LAION_Aesthetics_6_RIN_ldm_256_small_cad_text_swap",
    ]
    seed = randomize_seed_fn(seed, randomize_seed)
    torch.manual_seed(seed)
    x_N = torch.randn((num_images, 4, 32, 32), device=device, dtype=DTYPE)
    for model_name in models_keys:
        if "cad" in model_name:
            uncond_prompt = uncond_prompt_cad
        else:
            uncond_prompt = uncond_prompt_others
        model = MODELS[model_name]
        images = model.sample(
            x_N=x_N,
            cond=prompt,
            cfg=guidance_scale,
            num_steps=num_inference_steps,
            confidence_value=confidence_value,
            unconfidence_value=unconfidence_value,
            unconfident_prompt=uncond_prompt,
            stage="val",
        )
        images = images.cpu().numpy().transpose(0, 2, 3, 1)

        images = [Image.fromarray(image) for image in images]

        paths = save_images(
            images,
            profile,
            metadata={
                "prompt": prompt,
                "seed": seed,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
            },
        )
        results.append(paths)
    return *results, seed


def generate_confidence(
    model_name: str,
    prompt: str,
    seed: int = 0,
    guidance_scale: float = 8.0,
    num_inference_steps: int = 4,
    num_images: int = 4,
    randomize_seed: bool = False,
    unconfidence_value: float = 0.0,
    uncond_prompt: str = "",
    progress=gr.Progress(track_tqdm=True),
    profile: gr.OAuthProfile | None = None,
):
    results = []
    seed = randomize_seed_fn(seed, randomize_seed)
    torch.manual_seed(seed)
    x_N = torch.randn((num_images, 4, 32, 32), device=device, dtype=DTYPE)
    for confidence_value in [0.0, 2 / 7, 4 / 7, 6 / 7, 1.0]:
        images = MODELS[model_name].sample(
            x_N=x_N,
            cond=prompt,
            cfg=guidance_scale,
            num_steps=num_inference_steps,
            confidence_value=confidence_value,
            unconfidence_value=unconfidence_value,
            unconfident_prompt=uncond_prompt,
            stage="val",
        )
        images = images.cpu().numpy().transpose(0, 2, 3, 1)
        images = [Image.fromarray(image) for image in images]
        paths = save_images(
            images,
            profile,
            metadata={
                "prompt": prompt,
                "seed": seed,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "confidence_value": confidence_value,
            },
        )
        results.append(paths)
    flatten_results = [
        item for sublist in list(map(list, zip(*results))) for item in sublist
    ]
    return flatten_results, seed


for model_name in models_overrides.keys():
    load_model(model_name)

examples = []
examples_path = "./datasets/text_prompt_testbed/prompts.txt"
with open(examples_path, "r") as f:
    example = f.readlines()
    for line in example:
        examples.append(line.strip())

with gr.Blocks(css="demos/style.css") as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
    )
    ## Single model
    with gr.Group():
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            run_button = gr.Button("Run", scale=0)
        result = gr.Gallery(
            label="Generated images",
            show_label=False,
            elem_id="gallery",
            columns=4,
        )
    with gr.Accordion("Advanced options", open=False):
        seed = gr.Slider(
            label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0, randomize=True
        )
        randomize_seed = gr.Checkbox(label="Randomize seed across runs", value=True)
        with gr.Row():
            model_name = gr.Dropdown(
                list(models_overrides.keys()),
                label="Model to sample from",
                value=list(models_overrides.keys())[0],
                interactive=True,
            )

            guidance_scale = gr.Slider(
                label="Guidance scale for base",
                minimum=0.0,
                maximum=100.0,
                step=0.1,
                value=4.0,
            )
            confidence_value = gr.Slider(
                label="Confidence value for when using CAD",
                minimum=0.0,
                maximum=1.0,
                step=1 / 8,
                value=1.0,
            )
            unconfidence_value = gr.Slider(
                label="Unconfident value for when using CAD",
                minimum=0.0,
                maximum=1.0,
                step=1 / 7,
                value=0.0,
            )
        with gr.Row():
            uncond_prompt = gr.Text(
                label="Negative Prompt",
                max_lines=1,
                placeholder="",
            )
            num_images = gr.Slider(
                label="Number of images",
                minimum=1,
                maximum=16,
                step=1,
                value=4,
                visible=True,
            )
            num_inference_steps = gr.Slider(
                label="Number of inference steps for base",
                minimum=1,
                maximum=1000,
                step=1,
                value=250,
            )
    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=result,
        fn=generate,
        cache_examples=CACHE_EXAMPLES,
    )

    with gr.Group():
        with gr.Row():
            prompt_comparaison = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            run_button_comparaison = gr.Button("Run", scale=0)
        with gr.Row():
            result_baseline = gr.Gallery(
                label="CAD legacy",
                show_label=True,
                elem_id="gallery",
                columns=1,
                rows=4,
                height=700,
                allow_preview=False,
            )
            result_filtered = gr.Gallery(
                label="Filtered",
                show_label=True,
                elem_id="gallery",
                columns=1,
                rows=4,
                height=700,
                allow_preview=False,
            )
            result_weighted = gr.Gallery(
                label="CAD",
                show_label=True,
                elem_id="gallery",
                columns=1,
                rows=4,
                height=700,
                allow_preview=False,
            )
            result_cad = gr.Gallery(
                label="CAD text swap",
                show_label=True,
                elem_id="gallery",
                columns=1,
                rows=4,
                height=700,
                allow_preview=False,
            )
    with gr.Accordion("Advanced options", open=False):
        seed_comparaison = gr.Slider(
            label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0, randomize=True
        )
        randomize_seed_comparaison = gr.Checkbox(
            label="Randomize seed across runs", value=True
        )
        with gr.Row():
            guidance_scale_comparaison = gr.Slider(
                label="Guidance scale for base",
                minimum=0.0,
                maximum=100.0,
                step=0.1,
                value=4.0,
            )
            confidence_value_comparaison = gr.Slider(
                label="Confidence value for when using CAD",
                minimum=0.0,
                maximum=1.0,
                step=1 / 8,
                value=1.0,
            )
            unconfidence_value_comparaison = gr.Slider(
                label="Unconfident value for when using CAD",
                minimum=0.0,
                maximum=1.0,
                step=1 / 7,
                value=0.0,
            )
        with gr.Row():
            uncond_prompt_others_comparaison = gr.Text(
                label="Negative Prompt for others than CAD",
                max_lines=1,
                value="",
            )
            uncond_prompt_cad_comparaison = gr.Text(
                label="Negative Prompt for CAD",
                max_lines=1,
                value="A random prompt",
            )
            num_images_comparaison = gr.Slider(
                label="Number of images",
                minimum=1,
                maximum=16,
                step=1,
                value=4,
                visible=True,
            )
            num_inference_steps_comparaison = gr.Slider(
                label="Number of inference steps for base",
                minimum=1,
                maximum=1000,
                step=1,
                value=250,
            )
    # with gr.Accordion("Past generations", open=False):
    #     gr_user_history.render()
    gr.Examples(
        examples=examples,
        inputs=prompt_comparaison,
        outputs=[result_baseline, result_filtered, result_weighted, result_cad],
        fn=generate_comparaison,
        cache_examples=CACHE_EXAMPLES,
    )

    with gr.Group():
        with gr.Row():
            prompt_confidence = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            run_button_confidence = gr.Button("Run", scale=0)
        with gr.Row():
            result_confidence = gr.Gallery(
                label="Confidence 0 to 1",
                show_label=True,
                elem_id="gallery",
                columns=5,
                height=700,
                allow_preview=True,
            )
    with gr.Accordion("Advanced options", open=False):
        seed_confidence = gr.Slider(
            label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0, randomize=True
        )
        randomize_seed_confidence = gr.Checkbox(
            label="Randomize seed across runs", value=True
        )
        with gr.Row():
            model_name_confidence = gr.Dropdown(
                [x for x in list(models_overrides.keys()) if "cad" in x],
                label="Model to sample from",
                value=[x for x in list(models_overrides.keys()) if "cad" in x][0],
                interactive=True,
            )
            guidance_scale_confidence = gr.Slider(
                label="Guidance scale for base",
                minimum=0.0,
                maximum=100.0,
                step=0.1,
                value=4.0,
            )
            unconfidence_value_confidence = gr.Slider(
                label="Unconfident value for when using CAD",
                minimum=0.0,
                maximum=1.0,
                step=1 / 7,
                value=0.0,
            )
        with gr.Row():
            uncond_prompt_confidence = gr.Text(
                label="Negative Prompt for CAD",
                max_lines=1,
                value="A random prompt",
            )
            num_images_confidence = gr.Slider(
                label="Number of images",
                minimum=1,
                maximum=16,
                step=1,
                value=4,
                visible=True,
            )
            num_inference_steps_confidence = gr.Slider(
                label="Number of inference steps for base",
                minimum=1,
                maximum=1000,
                step=1,
                value=250,
            )
    # with gr.Accordion("Past generations", open=False):
    #     gr_user_history.render()
    gr.Examples(
        examples=examples,
        inputs=prompt_confidence,
        outputs=result_confidence,
        fn=generate_confidence,
        cache_examples=CACHE_EXAMPLES,
    )

    gr.on(
        triggers=[
            prompt.submit,
            run_button.click,
        ],
        fn=generate,
        inputs=[
            model_name,
            prompt,
            seed,
            guidance_scale,
            num_inference_steps,
            num_images,
            randomize_seed,
            confidence_value,
            unconfidence_value,
            uncond_prompt,
        ],
        outputs=[result, seed],
        api_name="run",
    )
    gr.on(
        triggers=[
            prompt_comparaison.submit,
            run_button_comparaison.click,
        ],
        fn=generate_comparaison,
        inputs=[
            prompt_comparaison,
            seed_comparaison,
            guidance_scale_comparaison,
            num_inference_steps_comparaison,
            num_images_comparaison,
            randomize_seed_comparaison,
            confidence_value_comparaison,
            unconfidence_value_comparaison,
            uncond_prompt_others_comparaison,
            uncond_prompt_cad_comparaison,
        ],
        outputs=[
            result_baseline,
            result_filtered,
            result_weighted,
            result_cad,
            seed_comparaison,
        ],
        api_name="run",
    )
    gr.on(
        triggers=[
            prompt_confidence.submit,
            run_button_confidence.click,
        ],
        fn=generate_confidence,
        inputs=[
            model_name_confidence,
            prompt_confidence,
            seed_confidence,
            guidance_scale_confidence,
            num_inference_steps_confidence,
            num_images_confidence,
            randomize_seed_confidence,
            unconfidence_value_confidence,
            uncond_prompt_confidence,
        ],
        outputs=[result_confidence, seed_confidence],
        api_name="run",
    )

    gr.on(
        triggers=[model_name.change],
        fn=load_model,
        inputs=[model_name],
        outputs=[],
        api_name="load_model",
        queue=True,
        # cancels=[prompt.submit, run_button.click],
    )

if __name__ == "__main__":
    demo.queue(api_open=False)
    # demo.queue(max_size=20).launch()
    demo.launch(share=True)
