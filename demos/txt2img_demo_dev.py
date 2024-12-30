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

from functools import partial

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from PIL import Image

from cad.models.diffusion import DiffusionModule
from cad.models.samplers import (
    ddim_sampler,
    ddpm_sampler,
    dpm_sampler,
    dpm_euler_sampler,
    dpm_plusplus_2S_sampler,
    dpm_plusplus_2M_sampler,
)
from cad.models.schedulers import (
    CosineScheduler,
    CosineSchedulerSimple,
    LinearScheduler,
    SigmoidScheduler,
)

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

aesthetic_negative_prompt = "out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"
models_overrides = {
    # "CC12m_LAION_Aesthetics_6_RIN_ldm_256_small": "cc12m_256_rin_small_ldm",
    "CC12m_LAION_Aesthetics_6_RIN_ldm_256_small_cad": {
        "overrides": "cad_256",
        # "model/postprocessing": "sd_1_5_consistency",
        "model/val_sampler": "dpm",
        "model/inference_noise_scheduler": "cosine_simple",
        "model.inference_noise_scheduler.ns": 0.002,
        "model.inference_noise_scheduler.ds": 0.0025,
    },
    "CC12M_LAION_Aesthetics6_512_LDM_RIN_grafted_latents_cooldown_aes_6_25": {
        "overrides": "cad_512",
        "model.network.num_blocks": 4,
        # "model/postprocessing": "sd_1_5_consistency",
    },
    # "CC12m_LAION_Aesthetics_6_RIN_ldm_256_small_cad_legacy": {
    #     "overrides": "cad_256",
    #     "model.network.latents_dim": 768,
    # },
    # "CC12m_LAION_Aesthetics_6_RIN_ldm_256_small_cad_text_swap": {
    #     "overrides": "cad_256"
    # },
    # # "CC12m_LAION_Aesthetics_6_RIN_ldm_512_small_cad": "cc12m_512_rin_small_ldm_cad",
    # "CC12m_LAION_Aesthetics_6_RIN_ldm_256_small_filtered": {
    #     "overrides": "cc12m_256_rin_small_ldm_filtered"
    # },
    # "CC12m_LAION_Aesthetics_6_RIN_ldm_256_small_weighted": "cc12m_256_rin_small_ldm_weighted",
}

MODELS = {}

SAMPLERS = {
    "ddim": ddim_sampler,
    "ddpm": ddpm_sampler,
    "dpm": dpm_sampler,
    "dpm_plusplus_2S": dpm_plusplus_2S_sampler,
    "dpm_plusplus_2M": dpm_plusplus_2M_sampler,
    "dpm_euler": dpm_euler_sampler,
}


def scheduler_fn(
    scheduler_type: str, start: float, end: float, tau: float, clip_min: float = 1e-9
):
    if scheduler_type == "sigmoid":
        return SigmoidScheduler(start, end, tau, clip_min)
    elif scheduler_type == "cosine":
        return CosineScheduler(start, end, tau, clip_min)
    elif scheduler_type == "linear":
        return LinearScheduler(clip_min)
    elif scheduler_type == "cosine_simple":
        return CosineSchedulerSimple(0.002, 0.0025)
    else:
        raise ValueError(f"Scheduler type {scheduler_type} not supported")


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
        hydra.initialize(version_base=None, config_path=f"../cad/configs")
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                "root_dir=../cad",
                f"experiment_name={ckpt_name}",
                "model/precomputed_text_embeddings='no'",
                *[f"{k}={v}" for k, v in override.items()],
            ],
        )
        model = DiffusionModule.load_from_checkpoint(
            f"cad/checkpoints/{ckpt_name}/last.ckpt",
            strict=False,
            cfg=cfg.model,
        ).to(device)

        model.eval()
        MODELS[model_name] = model
    return model_name


def generate(
    model_name: str,
    prompt: str,
    sampler: str,
    seed: int = 0,
    guidance_scale: float = 8.0,
    guidance_type="constant",
    guidance_start_step=0,
    latents_cfg_rate=0,
    num_inference_steps: int = 4,
    num_images: int = 4,
    randomize_seed: bool = False,
    coherence_value: float = 1.0,
    uncoherence_value: float = 0.0,
    uncond_prompt: str = "",
    thresholding_type="clamp",
    clamp_value=1.0,
    thresholding_percentile=0.995,
    progress=gr.Progress(track_tqdm=True),
    scheduler_type: str = "sigmoid",
    scheduler_start: float = -3,
    scheduler_end: float = 3,
    scheduler_tau: float = 1.1,
    profile: gr.OAuthProfile | None = None,
) -> PIL.Image.Image:
    uncond_prompt = None if uncond_prompt == "" else uncond_prompt
    model = MODELS[model_name]
    scheduler = scheduler_fn(
        scheduler_type, scheduler_start, scheduler_end, scheduler_tau
    )
    sampler = SAMPLERS[sampler]
    seed = randomize_seed_fn(seed, randomize_seed)
    torch.manual_seed(seed)
    images = model.sample(
        sampler=sampler,
        scheduler=scheduler,
        batch_size=num_images,
        shape=(4, 32, 32) if "256" in model_name else (4, 64, 64),
        cond=prompt,
        cfg=guidance_scale,
        guidance_type=guidance_type,
        guidance_start_step=guidance_start_step,
        latents_cfg_rate=latents_cfg_rate,
        num_steps=num_inference_steps,
        coherence_value=coherence_value,
        uncoherence_value=uncoherence_value,
        unconfident_prompt=uncond_prompt,
        stage="val",
        thresholding_type=thresholding_type,
        clamp_value=clamp_value,
        thresholding_percentile=thresholding_percentile,
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


def generate_coherence(
    model_name: str,
    prompt: str,
    sampler: str,
    seed: int = 0,
    guidance_scale: float = 8.0,
    guidance_type="constant",
    guidance_start_step=0,
    latents_cfg_rate=0,
    num_inference_steps: int = 4,
    num_images: int = 4,
    randomize_seed: bool = False,
    uncoherence_value: float = 0.0,
    uncond_prompt: str = "",
    thresholding_type="clamp",
    clamp_value=1.0,
    thresholding_percentile=0.995,
    scheduler_type: str = "sigmoid",
    scheduler_start: float = -3,
    scheduler_end: float = 3,
    scheduler_tau: float = 1.1,
    progress=gr.Progress(track_tqdm=True),
    profile: gr.OAuthProfile | None = None,
):
    results = []
    seed = randomize_seed_fn(seed, randomize_seed)
    scheduler = scheduler_fn(
        scheduler_type, scheduler_start, scheduler_end, scheduler_tau
    )
    sampler = SAMPLERS[sampler]
    torch.manual_seed(seed)
    x_N = (
        torch.randn((num_images, 4, 32, 32), device=device, dtype=DTYPE)
        if "256" in model_name
        else torch.randn((num_images, 4, 64, 64), device=device, dtype=DTYPE)
    )
    for coherence_value in [0.0, 2 / 7, 4 / 7, 6 / 7, 1.0]:
        images = MODELS[model_name].sample(
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
                "coherence_value": coherence_value,
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
examples_path = "cad/datasets/text_prompt_testbed/prompts.txt"
with open(examples_path, "r") as f:
    example = f.readlines()
    for line in example:
        examples.append(line.strip())

with gr.Blocks(css="demos/style.css") as demo:
    gr.Markdown(
        """
        ## CAD Text-to-Image Demo
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
                value="CC12M_LAION_Aesthetics6_512_LDM_RIN_grafted_latents_cooldown_aes_6_25",
                interactive=True,
            )

            guidance_scale = gr.Slider(
                label="Guidance scale",
                minimum=0.0,
                maximum=100.0,
                step=0.1,
                value=30.0,
            )
            coherence_value = gr.Slider(
                label="Coherence value",
                minimum=0.0,
                maximum=1.0,
                step=1 / 7,
                value=1.0,
            )
            uncoherence_value = gr.Slider(
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
                value="ewrfokwe rweo rfowe fjewo fwe",
            )
            num_images = gr.Slider(
                label="Number of images",
                minimum=1,
                maximum=4,
                step=1,
                value=4,
                visible=True,
            )
            sampler = gr.Dropdown(
                SAMPLERS.keys(),
                label="Sampler",
                value="dpm_plusplus_2M",
                interactive=True,
            )
            num_inference_steps = gr.Slider(
                label="Number of inference steps",
                minimum=1,
                maximum=1000,
                step=1,
                value=250,
            )
        with gr.Row():
            guidance_type = gr.Dropdown(
                ["constant", "linear"],
                label="Guidance Scheduler",
                value="constant",
                interactive=True,
            )
            guidance_start_step = gr.Slider(
                label="First step of guidance",
                minimum=0,
                maximum=1000,
                step=1,
                value=10,
            )
            latents_cfg_rate = gr.Slider(
                label="Guidance scale for latents",
                minimum=0.0,
                maximum=50.0,
                step=0.1,
                value=0.0,
            )
        with gr.Row():
            thresholding_type = gr.Dropdown(
                ["clamp", "dynamic_thresholding", "per_channel_dynamic_thresholding"],
                label="Clamping Strategy",
                value="per_channel_dynamic_thresholding",
                interactive=True,
            )
            clamp_value = gr.Slider(
                label="Clamp value",
                minimum=1.0,
                maximum=2.0,
                step=0.01,
                value=1.0,
            )
            thresholding_percentile = gr.Slider(
                label="Dynamic Threholding quantile",
                minimum=0,
                maximum=1,
                step=0.001,
                value=0.005,
            )
        with gr.Row():
            time_scheduler = gr.Dropdown(
                ["sigmoid", "cosine", "cosine_simple", "linear"],
                label="Time Scheduler",
                value="sigmoid",
                interactive=True,
            )
            time_scheduler_start = gr.Slider(
                label="Time Scheduler Start",
                minimum=-10,
                maximum=0,
                step=0.01,
                value=-3,
            )
            time_scheduler_end = gr.Slider(
                label="Time Scheduler End",
                minimum=0,
                maximum=10,
                step=0.01,
                value=3,
            )
            tau_scheduler = gr.Slider(
                label="Time Scheduler Tau",
                minimum=0,
                maximum=2,
                step=0.01,
                value=1.1,
            )
    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=result,
        fn=generate,
        cache_examples=CACHE_EXAMPLES,
    )
    with gr.Group():
        gr.Markdown(
            """
        ## Varying the coherence
        """
        )
        with gr.Row():
            prompt_coherence = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            run_button_coherence = gr.Button("Run", scale=0)
        with gr.Row():
            result_coherence = gr.Gallery(
                label="Coherence 0 to 1",
                show_label=True,
                elem_id="gallery",
                columns=5,
                height=700,
                allow_preview=True,
            )
    with gr.Accordion("Advanced options", open=False):
        seed_coherence = gr.Slider(
            label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0, randomize=True
        )
        randomize_seed_coherence = gr.Checkbox(
            label="Randomize seed across runs", value=True
        )
        with gr.Row():
            model_name_coherence = gr.Dropdown(
                models_overrides.keys(),
                label="Model to sample from",
                value="CC12M_LAION_Aesthetics6_512_LDM_RIN_grafted_latents_cooldown_aes_6_25",
                interactive=True,
            )
            guidance_scale_coherence = gr.Slider(
                label="Guidance scale",
                minimum=0.0,
                maximum=100.0,
                step=0.1,
                value=30.0,
            )
            uncoherence_value_coherence = gr.Slider(
                label="Uncoherent value",
                minimum=0.0,
                maximum=1.0,
                step=1 / 7,
                value=0.0,
            )
        with gr.Row():
            uncond_prompt_coherence = gr.Text(
                label="Negative Prompt",
                max_lines=1,
                value="ewrfokwe rweo rfowe fjewo fwe",
            )
            num_images_coherence = gr.Slider(
                label="Number of images",
                minimum=1,
                maximum=4,
                step=1,
                value=4,
                visible=True,
            )
            sampler_coherence = gr.Dropdown(
                SAMPLERS.keys(),
                label="Sampler",
                value="dpm_plusplus_2M",
                interactive=True,
            )
            num_inference_steps_coherence = gr.Slider(
                label="Number of inference steps",
                minimum=1,
                maximum=1000,
                step=1,
                value=250,
            )
        with gr.Row():
            guidance_type_coherence = gr.Dropdown(
                ["constant", "linear"],
                label="Guidance Scheduler",
                value="constant",
                interactive=True,
            )
            guidance_start_step_coherence = gr.Slider(
                label="First step of guidance",
                minimum=0,
                maximum=1000,
                step=1,
                value=10,
            )
            latents_cfg_rate_coherence = gr.Slider(
                label="Guidance scale for latents",
                minimum=0.0,
                maximum=50.0,
                step=0.1,
                value=0.0,
            )
        with gr.Row():
            thresholding_type_coherence = gr.Dropdown(
                ["clamp", "dynamic_thresholding", "per_channel_dynamic_thresholding"],
                label="Clamping Strategy",
                value="per_channel_dynamic_thresholding",
                interactive=True,
            )
            clamp_value_coherence = gr.Slider(
                label="Clamp value",
                minimum=1.0,
                maximum=2.0,
                step=0.01,
                value=1.0,
            )
            thresholding_percentile_coherence = gr.Slider(
                label="Dynamic Threholding quantile",
                minimum=0,
                maximum=1,
                step=0.001,
                value=0.005,
            )
        with gr.Row():
            time_scheduler_coherence = gr.Dropdown(
                ["sigmoid", "cosine", "cosine_simple", "linear"],
                label="Time Scheduler",
                value="sigmoid",
                interactive=True,
            )
            time_scheduler_start_coherence = gr.Slider(
                label="Time Scheduler Start",
                minimum=-10,
                maximum=0,
                step=0.01,
                value=-3,
            )
            time_scheduler_end_coherence = gr.Slider(
                label="Time Scheduler End",
                minimum=0,
                maximum=10,
                step=0.01,
                value=3,
            )
            tau_scheduler_coherence = gr.Slider(
                label="Time Scheduler Tau",
                minimum=0,
                maximum=2,
                step=0.01,
                value=1.1,
            )
    gr.Examples(
        examples=examples,
        inputs=prompt_coherence,
        outputs=result_coherence,
        fn=generate_coherence,
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
            sampler,
            seed,
            guidance_scale,
            guidance_type,
            guidance_start_step,
            latents_cfg_rate,
            num_inference_steps,
            num_images,
            randomize_seed,
            coherence_value,
            uncoherence_value,
            uncond_prompt,
            thresholding_type,
            clamp_value,
            thresholding_percentile,
            time_scheduler,
            time_scheduler_start,
            time_scheduler_end,
            tau_scheduler,
        ],
        outputs=[result, seed],
        api_name="run",
    )
    gr.on(
        triggers=[
            prompt_coherence.submit,
            run_button_coherence.click,
        ],
        fn=generate_coherence,
        inputs=[
            model_name_coherence,
            prompt_coherence,
            sampler_coherence,
            seed_coherence,
            guidance_scale_coherence,
            guidance_type_coherence,
            guidance_start_step_coherence,
            latents_cfg_rate_coherence,
            num_inference_steps_coherence,
            num_images_coherence,
            randomize_seed_coherence,
            uncoherence_value_coherence,
            uncond_prompt_coherence,
            thresholding_type_coherence,
            clamp_value_coherence,
            thresholding_percentile_coherence,
            time_scheduler_coherence,
            time_scheduler_start_coherence,
            time_scheduler_end_coherence,
            tau_scheduler_coherence,
        ],
        outputs=[result_coherence, seed_coherence],
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
