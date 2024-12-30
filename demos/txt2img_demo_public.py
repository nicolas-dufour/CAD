from __future__ import annotations

import os
import random
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor

import gradio as gr
import numpy as np
import PIL.Image
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
from cad.pipe import CADT2IPipeline

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
    "CAD_256": "nicolas-dufour/CAD_256",
    "CAD_512": "nicolas-dufour/CAD_512",
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
        return LinearScheduler(clip_min=clip_min)
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
        MODELS[model_name] = CADT2IPipeline(
            models_overrides[model_name],
        ).to(device)
    return model_name


def generate(
    model_name: str,
    prompt: str,
    sampler: str,
    seed: int = 0,
    guidance_scale: float = 8.0,
    guidance_type="constant",
    guidance_start_step=0,
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
    profile: gr.OAuthProfile | None = None,
) -> PIL.Image.Image:
    uncond_prompt = None if uncond_prompt == "" else uncond_prompt
    model = MODELS[model_name]
    scheduler = scheduler_fn("linear", 1, 0, 1)
    sampler = SAMPLERS[sampler]
    seed = randomize_seed_fn(seed, randomize_seed)
    torch.manual_seed(seed)
    images = model(
        prompt,
        sampler=sampler,
        scheduler=scheduler,
        num_samples=num_images,
        cfg=guidance_scale,
        guidance_type=guidance_type,
        guidance_start_step=guidance_start_step,
        num_steps=num_inference_steps,
        coherence_value=coherence_value,
        uncoherence_value=uncoherence_value,
        unconfident_prompt=uncond_prompt,
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


for model_name in models_overrides.keys():
    load_model(model_name)

examples = [
    "Photography closeup portrait of an adorable rusty broken­down steampunk robot covered in budding vegetation, surrounded by tall grass, misty futuristic sci­fi forest environment.",
    "A cute little matte low poly isometric Zelda Breath of the wild forest island, waterfalls, soft shadows, trending on Artstation, 3d render, monument valley, fez video game.",
    "A wizard by Q Hayashida in the style of Dorohedoro for Elden Ring, with biggest most intricate sword, on sunlit battlefield, breath of the wild, striking illustration.",
    "Chinese illustration, oriental landscape painting, above super wide angle, magical, romantic, detailed, colorful, multi-dimensional paper kirigami craft.",
    "A shanty version of Tokyo, new rustic style, bold colors with all colors palette, video game, genshin, tribe, fantasy, overwatch.",
    "Papillon dog puppy in style of sumi ink painting, fantasy art, enigmatic, mysterious.",
    "Pirate ship trapped in a cosmic maelstrom nebula, rendered in cosmic beach whirlpool engine, volumetric lighting, spectacular, ambient lights, light pollution, cinematic atmosphere, art nouveau style, illustration art artwork by SenseiJaye, intricate detail.",
    "portrait photo of a asia old warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes, 50mm portrait photography, hard rim lighting photography–beta",
    "An old-world galleon navigating through turbulent ocean waves under a stormy sky, lit by flashes of lightning",
]

with gr.Blocks(css="demos/style.css") as demo:
    gr.Markdown(
        """
        ## Cézanne Text-to-Image Demo
        This demo is powered by [Cézanne](https://huggingface.co/nicolas-dufour/CAD_512), a state-of-the-art text-to-image model.
        The associated paper can be found [here](https://arxiv.org/abs/2405.20324).
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
                value="CAD_512",
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
                value="ddpm",
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
    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=result,
        fn=generate,
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
            num_inference_steps,
            num_images,
            randomize_seed,
            coherence_value,
            uncoherence_value,
            uncond_prompt,
            thresholding_type,
            clamp_value,
            thresholding_percentile,
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
