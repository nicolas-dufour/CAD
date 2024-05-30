<div align="center">

# Dont Drop Your samples: Coherence-aware training benefits Condition Diffusion

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
Then run
```bash
pip install -r requirements.txt
```

Depending of your CUDA version be careful installing torch.

This repo is based around Hydra and requires to specify an override such as:
```bash
python train.py overrides=imagenet_64_rin **Other hydra args**
```
You can use the default or create your own override to train the desired model.

## Datasets
Downlowad the datasets and add them in ```/datasets```. A few presets are already defined in the ```configs/data``` folder (Imagenet, CIFAR-10, LAION Aesthetic 6+ and CC12M)

To add a custom dataset, create a new config file in ```configs/data``` and add the dataset to the ```datasets``` folder.

This repo supports both Pytorch Datasets and Webdatasets.

### LAION and CC12m preprocessing
To preprocess the LAION Aesthetic 6+ and CC12M datasets, you can use the following command:
```bash
python data/processing_scripts/preprocess_data.py --src path_to_src_wds --dest path_to_dst_wds --shard_id number_of_the_shard 
```
This is better used with a cluster to preprocess the data in parallel with job array.

## Training class-conditional models

To train CAD on Imagenet you can use the following command:
```bash
python train.py overrides=imagenet_64_rin_cad
```
For CIFAR-10:
```bash
python train.py overrides=cifar10_rin_cad
```

## Training text-conditional models
As a side contribution, we also provide a new text-to-image model called TextRIN. This model is based on RIN and is conditioned on text.
![TextRIN](/assets/text_rin_white.png)

To train TextRIN with CAD on LAION Aesthetic 6+ and CC12M you can use the following command:
```bash
python train.py overrides=cc12m_256_rin_tiny_ldm_cad.yaml
```

## TextRIN without CAD

To train TextRIN without CAD on LAION Aesthetic 6+ and CC12M you can use the following command:
```bash
python train.py overrides=cc12m_256_rin_tiny_ldm.yaml
```

## Reproduction of RIN
This repo also features a reproduction of RIN for Imagenet-64 and CIFAR-10. To train RIN on Imagenet-64 you can use the following command:

```bash
python train.py overrides=imagenet_64_rin
```

For CIFAR-10:
```bash
python train.py overrides=cifar10_rin
```

## Pretrained models
Coming soon

## Citation
If you happen to use this repo in your experiments, you can acknowledge us by citing the following paper:
```bibtex
@article{dufour2024cad,
  title={Dont Drop Your samples: Coherence-aware training benefits Condition Diffusion},
  author={Nicolas Dufour and Victor Besnier and Vicky Kalogeiton and David Picard},
  journal={CVPR}
  year={2024}
}
```
