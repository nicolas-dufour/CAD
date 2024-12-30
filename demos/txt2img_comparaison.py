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

from cad.models.diffusion import DiffusionModule
from cad.models.samplers import ddim_sampler, ddpm_sampler, dpm_sampler
from cad.models.schedulers import (
    CosineScheduler,
    CosineSchedulerSimple,
    LinearScheduler,
    SigmoidScheduler,
)

DESCRIPTION = """# CAD Model Comparison Demo
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
    "CC12M_LAION_Aesthetics6_512_LDM_RIN_grafted_latents": {
        "overrides": "cad_512",
        "model.network.num_blocks": 4,
    },
    "CC12M_LAION_Aesthetics6_512_LDM_RIN_grafted_latents_cooldown_aes_6": {
        "overrides": "cad_512",
        "model.network.num_blocks": 4,
    },
    "CC12M_LAION_Aesthetics6_512_LDM_RIN_grafted_latents_cooldown_aes_6_25": {
        "overrides": "cad_512",
        "model.network.num_blocks": 4,
    },
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


def generate_comparaison(
    prompt: str,
    seed: int = 0,
    guidance_scale: float = 8.0,
    guidance_type: str = "constant",
    guidance_start_step: int = 0,
    latents_cfg_rate: float = 0,
    num_inference_steps: int = 4,
    num_images: int = 4,
    randomize_seed: bool = False,
    coherence_value: float = 1.0,
    uncoherence_value: float = 0.0,
    uncond_prompt: str = "",
    thresholding_type: str = "clamp",
    clamp_value: float = 1.0,
    thresholding_percentile: float = 0.995,
    scheduler_type: str = "sigmoid",
    scheduler_start: float = -3,
    scheduler_end: float = 3,
    scheduler_tau: float = 1.1,
    sampler: str = "dpm",
    progress=gr.Progress(track_tqdm=True),
    profile: gr.OAuthProfile | None = None,
):
    results = []
    models_keys = [
        "CC12M_LAION_Aesthetics6_512_LDM_RIN_grafted_latents",
        "CC12M_LAION_Aesthetics6_512_LDM_RIN_grafted_latents_cooldown_aes_6",
        "CC12M_LAION_Aesthetics6_512_LDM_RIN_grafted_latents_cooldown_aes_6_25",
    ]
    seed = randomize_seed_fn(seed, randomize_seed)
    torch.manual_seed(seed)
    scheduler = scheduler_fn(
        scheduler_type, scheduler_start, scheduler_end, scheduler_tau
    )
    sampler = SAMPLERS[sampler]
    for model_name in models_keys:
        x_N = (
            torch.randn(
                (num_images, 4, 32, 32),
                device=device,
                dtype=DTYPE,
                generator=torch.Generator(device=device).manual_seed(seed),
            )
            if "256" in model_name
            else torch.randn(
                (num_images, 4, 64, 64),
                device=device,
                dtype=DTYPE,
                generator=torch.Generator(device=device).manual_seed(seed),
            )
        )
        model = MODELS[model_name]
        images = model.sample(
            sampler=sampler,
            scheduler=scheduler,
            x_N=x_N,
            cond=prompt,
            cfg=guidance_scale,
            guidance_type=guidance_type,
            guidance_start_step=guidance_start_step,
            latents_cfg_rate=latents_cfg_rate,
            num_steps=num_inference_steps,
            coherence_value=coherence_value,
            uncoherence_value=uncoherence_value,
            unconfident_prompt=uncond_prompt,
            thresholding_type=thresholding_type,
            clamp_value=clamp_value,
            thresholding_percentile=thresholding_percentile,
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


for model_name in models_overrides.keys():
    load_model(model_name)

examples = []
examples_path = "cad/datasets/text_prompt_testbed/prompts.txt"
with open(examples_path, "r") as f:
    example = f.readlines()
    for line in example:
        examples.append(line.strip())

SAMPLERS = {
    "ddim": ddim_sampler,
    "ddpm": ddpm_sampler,
    "dpm": dpm_sampler,
}


def scheduler_fn(
    scheduler_type: str, start: float, end: float, tau: float, clip_min: float = 1e-9
):
    if scheduler_type == "sigmoid":
        return SigmoidScheduler(start, end, tau, clip_min)
    elif scheduler_type == "cosine":
        return CosineScheduler(start, end, tau, clip_min)
    elif scheduler_type == "linear":
        return LinearScheduler()
    elif scheduler_type == "cosine_simple":
        return CosineSchedulerSimple(0.002, 0.0025)
    else:
        raise ValueError(f"Scheduler type {scheduler_type} not supported")


with gr.Blocks(css="demos/style.css") as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
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
                label="Base 180K",
                show_label=True,
                elem_id="gallery",
                columns=1,
                rows=4,
                height=700,
                allow_preview=False,
            )
            result_filtered = gr.Gallery(
                label="Aes 6 cooldown",
                show_label=True,
                elem_id="gallery",
                columns=1,
                rows=4,
                height=700,
                allow_preview=False,
            )
            result_filtered_25 = gr.Gallery(
                label="Aes 6.25 cooldown",
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
                label="Guidance scale",
                minimum=0.0,
                maximum=50.0,
                step=0.1,
                value=13.0,
            )
            coherence_value_comparaison = gr.Slider(
                label="Coherence value",
                minimum=0.0,
                maximum=1.0,
                step=1 / 7,
                value=1.0,
            )
            uncoherence_value_comparaison = gr.Slider(
                label="Uncoherent value",
                minimum=0.0,
                maximum=1.0,
                step=1 / 7,
                value=0.0,
            )
        with gr.Row():
            uncond_prompt = gr.Text(
                label="Negative Prompt",
                max_lines=1,
                value="A random prompt",
            )
            num_images_comparaison = gr.Slider(
                label="Number of images",
                minimum=1,
                maximum=4,
                step=1,
                value=4,
                visible=True,
            )
            sampler_comparaison = gr.Dropdown(
                list(SAMPLERS.keys()),
                label="Sampler",
                value="ddim",
                interactive=True,
            )
            num_inference_steps_comparaison = gr.Slider(
                label="Number of inference steps",
                minimum=1,
                maximum=1000,
                step=1,
                value=150,
            )
        with gr.Row():
            guidance_type_comparaison = gr.Dropdown(
                ["constant", "linear"],
                label="Guidance Scheduler",
                value="constant",
                interactive=True,
            )
            guidance_start_step_comparaison = gr.Slider(
                label="First step of guidance",
                minimum=0,
                maximum=1000,
                step=1,
                value=15,
            )
            latents_cfg_rate_comparaison = gr.Slider(
                label="Guidance scale for latents",
                minimum=0.0,
                maximum=50.0,
                step=0.1,
                value=0.0,
            )
        with gr.Row():
            thresholding_type_comparaison = gr.Dropdown(
                ["clamp", "dynamic_thresholding", "per_channel_dynamic_thresholding"],
                label="Clamping Strategy",
                value="clamp",
                interactive=True,
            )
            clamp_value_comparaison = gr.Slider(
                label="Clamp value",
                minimum=1.0,
                maximum=2.0,
                step=0.01,
                value=1.2,
            )
            thresholding_percentile_comparaison = gr.Slider(
                label="Dynamic Threholding quantile",
                minimum=0,
                maximum=1,
                step=0.001,
                value=0.65,
            )
        with gr.Row():
            time_scheduler_comparaison = gr.Dropdown(
                ["sigmoid", "cosine", "cosine_simple", "linear"],
                label="Time Scheduler",
                value="linear",
                interactive=True,
            )
            time_scheduler_start_comparaison = gr.Slider(
                label="Time Scheduler Start",
                minimum=-10,
                maximum=0,
                step=0.01,
                value=-3,
            )
            time_scheduler_end_comparaison = gr.Slider(
                label="Time Scheduler End",
                minimum=0,
                maximum=10,
                step=0.01,
                value=3,
            )
            tau_scheduler_comparaison = gr.Slider(
                label="Time Scheduler Tau",
                minimum=0,
                maximum=2,
                step=0.01,
                value=1.1,
            )
    gr.Examples(
        examples=examples,
        inputs=prompt_comparaison,
        outputs=[result_baseline, result_filtered],
        fn=generate_comparaison,
        cache_examples=CACHE_EXAMPLES,
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
            guidance_type_comparaison,
            guidance_start_step_comparaison,
            latents_cfg_rate_comparaison,
            num_inference_steps_comparaison,
            num_images_comparaison,
            randomize_seed_comparaison,
            coherence_value_comparaison,
            uncoherence_value_comparaison,
            uncond_prompt,
            thresholding_type_comparaison,
            clamp_value_comparaison,
            thresholding_percentile_comparaison,
            time_scheduler_comparaison,
            time_scheduler_start_comparaison,
            time_scheduler_end_comparaison,
            tau_scheduler_comparaison,
            sampler_comparaison,
        ],
        outputs=[
            result_baseline,
            result_filtered,
            result_filtered_25,
            seed_comparaison,
        ],
        api_name="run",
    )

if __name__ == "__main__":
    demo.queue(api_open=False)
    demo.launch(share=True)
