<div align="center">

# Don’t drop your samples! Coherence-aware training benefits Conditional diffusion

<a href="https://nicolas-dufour.github.io/" >Nicolas Dufour</a>, <a href="https://scholar.google.com/citations?user=n_C2h-QAAAAJ&hl=fr&oi=ao" >Victor Besnier</a>, <a href="https://vicky.kalogeiton.info/" >Vicky Kalogeiton</a>, <a href="https://davidpicard.github.io/" >David Picard</a>
</div>

![CAD](/assets/varying_coherence.png)

This repo has the code for the paper "Dont Drop Your samples: Coherence-aware training benefits Condition Diffusion" accepted at CVPR 2024 as a Highlight.

The core idea is that diffusion model is usually trained on noisy data. The usual solution is to filter massive datapools. We propose a new training method that leverages the coherence of the data to improve the training of diffusion models. We show that this method improves the quality of the generated samples on several datasets.

Project website: [https://nicolas-dufour.github.io/cad](https://nicolas-dufour.github.io/cad)

## Install

To install, first create a conda env with python 3.9/3.10

```bash
conda create -n cad python=3.10
```
Activate the env

```bash
conda activate cad
```

For inference only,

```bash
pip install cad-diffusion
```

If you want to use the training pipeline:

```bash
pip install cad-diffusion[training]
```

Depending of your CUDA version be careful installing torch.

This repo is based around Hydra and requires to specify an override such as:
```bash
python cad/train.py overrides=imagenet_64_rin **Other hydra args**
```
You can use the default or create your own override to train the desired model.

## Pretrained models

To use the pretrained model do the following:
```python
from cad import CADT2IPipeline

pipe = CADT2IPipeline("nicolas-dufour/CAD_512").to("cuda")

prompt = "An avocado armchair"

image = pipe(prompt, cfg=15)
```

If you just want to download the models, not the sampling pipeline, you can do:

```python
from cad import CAD

model = CAD.from_pretrained("nicolas-dufour/CAD_512")
```

Models are hosted in the hugging face hub. The previous scripts download them automatically, but weights can be found at:

[https://huggingface.co/nicolas-dufour/CAD_256](https://huggingface.co/nicolas-dufour/CAD_256)

[https://huggingface.co/nicolas-dufour/CAD_512](https://huggingface.co/nicolas-dufour/CAD_512)

## Using the Pipeline

The `CADT2IPipeline` class provides a comprehensive interface for generating images from text prompts. Here's a detailed guide on how to use it:

### Basic Usage

```python
from cad import CADT2IPipeline

# Initialize the pipeline
pipe = CADT2IPipeline("nicolas-dufour/CAD_512").to("cuda")

# Generate an image from a prompt
prompt = "An avocado armchair"
image = pipe(prompt, cfg=15)
```

### Advanced Configuration

The pipeline can be initialized with several customization options:

```python
pipe = CADT2IPipeline(
    model_path="nicolas-dufour/CAD_512",
    sampler="ddim",                    # Options: "ddim", "ddpm", "dpm", "dpm_2S", "dpm_2M"
    scheduler="sigmoid",               # Options: "sigmoid", "cosine", "linear"
    postprocessing="sd_1_5_vae",      # Options: "consistency-decoder", "sd_1_5_vae"
    scheduler_start=-3,
    scheduler_end=3,
    scheduler_tau=1.1,
    device="cuda"
)
```

### Generation Parameters

The pipeline's `__call__` method accepts various parameters to control the generation process:

```python
image = pipe(
    cond="A beautiful landscape",          # Text prompt or list of prompts
    num_samples=4,                         # Number of images to generate
    cfg=15,                               # Classifier-free guidance scale
    guidance_type="constant",             # Type of guidance: "constant", "linear"
    guidance_start_step=0,                # Step to start guidance
    coherence_value=1.0,                  # Coherence value for sampling
    uncoherence_value=0.0,                # Uncoherence value for sampling
    thresholding_type="clamp",           # Type of thresholding: "clamp", "dynamic_thresholding", "per_channel_dynamic_thresholding"
    clamp_value=1.0,                      # Clamp value for thresholding
    thresholding_percentile=0.995         # Percentile for thresholding
)
```

#### Guidance Types
- `constant`: Applies uniform guidance throughout the sampling process
- `linear`: Linearly increases guidance strength from start to end
- `exponential`: Exponentially increases guidance strength from start to end

#### Thresholding Types
- `clamp`: Clamps values to a fixed range using `clamp_value`
- `dynamic`: Dynamically adjusts thresholds based on the batch statistics
- `percentile`: Uses percentile-based thresholding with `thresholding_percentile`

### Advanced Parameters

For more control over the generation process, you can also specify:

- `x_N`: Initial noise tensor
- `latents`: Previous latents for continuation
- `num_steps`: Custom number of sampling steps
- `sampler`: Custom sampler function
- `scheduler`: Custom scheduler function
- `guidance_start_step`: Step to start guidance
- `generator`: Random number generator for reproducibility
- `unconfident_prompt`: Custom unconfident prompt text

## Training
### Datasets
Downlowad the datasets and add them in ```/datasets```. A few presets are already defined in the ```configs/data``` folder (Imagenet, CIFAR-10, LAION Aesthetic 6+ and CC12M)

To add a custom dataset, create a new config file in ```configs/data``` and add the dataset to the ```datasets``` folder.

This repo supports both Pytorch Datasets and Webdatasets.

### LAION and CC12m preprocessing
To preprocess the LAION Aesthetic 6+ and CC12M datasets, you can use the following command:
```bash
python cad/data/processing_scripts/preprocess_data.py --src path_to_src_wds --dest path_to_dst_wds --shard_id number_of_the_shard
```
This is better used with a cluster to preprocess the data in parallel with job array.

### Training class-conditional models

To train CAD on Imagenet you can use the following command:
```bash
python cad/train.py overrides=imagenet_64_rin_cad
```
For CIFAR-10:
```bash
python cad/train.py overrides=cifar10_rin_cad
```

### Training text-conditional models
As a side contribution, we also provide a new text-to-image model called TextRIN. This model is based on RIN and is conditioned on text.
![TextRIN](/assets/text_rin_white.png)

To train TextRIN with CAD on LAION Aesthetic 6+ and CC12M you can use the following command:
```bash
python cad/train.py overrides=cc12m_256_rin_tiny_ldm_cad
```

### Training text-conditional models

To train TextRIN without CAD on LAION Aesthetic 6+ and CC12M you can use the following command:

```bash
python cad/train.py overrides=cc12m_256_rin_tiny_ldm
```

### Reproduction of RIN
This repo also features a reproduction of RIN for Imagenet-64 and CIFAR-10. To train RIN on Imagenet-64 you can use the following command:

```bash
python cad/train.py overrides=imagenet_64_rin
```

For CIFAR-10:
```bash
python cad/train.py overrides=cifar10_rin
```

## Citation
If you happen to use this repo in your experiments, you can acknowledge us by citing the following paper:

```bibtex
@article{dufour2024dont,
  title={Don’t drop your samples! Coherence-aware training benefits Conditional diffusion},
  author={Nicolas Dufour and Victor Besnier and Vicky Kalogeiton and David Picard},
  journal={CVPR}
  year={2024}
}
```
