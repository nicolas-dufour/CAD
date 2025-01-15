import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="cad-diffusion",
    version="1.2",
    description="Package for the CAD diffusion model",
    author="Nicolas Dufour",
    python_requires=">=3.11",
    install_requires=[
        "torch",
        "torchvision",
        "einops",
        "transformers",
        "diffusers",
        "huggingface_hub>=0.23",
    ],
    packages=["cad"],
    package_data={"cad": ["**/*"]},
    include_package_data=True,
    extras_require={
        "train": [
            "hydra-core",
            "lightning",
            "torch-fidelity",
            "pandas",
            "wandb",
            "timm",
            "scikit-learn",
            "webdataset==0.2.57",
            "lovely_tensors",
        ],
        "demo": ["gradio"],
    },
)
