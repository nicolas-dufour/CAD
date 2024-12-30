from __future__ import annotations

import os
import random
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import gradio as gr
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from PIL import Image

from cad.cad.models.diffusion import DiffusionModule

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
    "Imagenet_256_LDM_Large": {
        "overrides": "imagenet_256_rin_ldm_large",
        "model.channel_wise_normalisation": True,
        "model.data_preprocessing.vae_sample": True,
    },
    # "Imagenet_256_LDM_channel_wise_vae": {
    #     "overrides": "imagenet_256_rin_ldm",
    #     "model.channel_wise_normalisation": True,
    #     "model.data_preprocessing.vae_sample": True,
    # },
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
    id: int,
    seed: int = 0,
    guidance_scale: float = 8.0,
    num_inference_steps: int = 4,
    num_images: int = 4,
    randomize_seed: bool = False,
    clamp_value: float = 1.0,
    sampler="val",
    progress=gr.Progress(track_tqdm=True),
    profile: gr.OAuthProfile | None = None,
) -> PIL.Image.Image:
    uncond_prompt = None if uncond_prompt == "" else uncond_prompt
    model = MODELS[model_name]
    seed = randomize_seed_fn(seed, randomize_seed)
    torch.manual_seed(seed)
    images = model.sample(
        batch_size=num_images,
        shape=(4, 32, 32),
        cond=id,
        cfg=guidance_scale,
        num_steps=num_inference_steps,
        stage=sampler,
        clamp_value=clamp_value,
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


for model_name in models_overrides.keys():
    load_model(model_name)


with gr.Blocks(css="demos/style.css") as demo:
    gr.Markdown(
        """
        ## CAD Imagenet Demo
        """
    )
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
    )
    ## Single model
    with gr.Group():
        with gr.Row():
            prompt = gr.Slider(
                label="Class ID", minimum=0, maximum=999, step=1, value=0
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
                value="Imagenet_256_LDM_Large",
                interactive=True,
            )

            guidance_scale = gr.Slider(
                label="Guidance scale", minimum=0.0, maximum=7.0, step=0.1, value=0.5
            )
        with gr.Row():
            num_images = gr.Slider(
                label="Number of images",
                minimum=1,
                maximum=4,
                step=1,
                value=4,
                visible=True,
            )
            num_inference_steps = gr.Slider(
                label="Number of inference steps",
                minimum=1,
                maximum=1000,
                step=1,
                value=500,
            )
            clamp_value = gr.Slider(
                label="Clamp value",
                minimum=1.0,
                maximum=2.0,
                step=0.1,
                value=1.2,
            )

    gr.on(
        triggers=[
            # prompt.submit,
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
            clamp_value,
        ],
        outputs=[result, seed],
        api_name="run",
    )
    gr.on(
        triggers=[model_name.change],
        fn=load_model,
        inputs=[model_name],
        outputs=[],
        api_name="load_model",
        queue=True,
    )

if __name__ == "__main__":
    demo.queue(api_open=False)
    demo.launch(share=True)
