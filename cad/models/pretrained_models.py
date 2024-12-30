import sys
import os

from cad.models.networks.rin import CADRINTextCond
from huggingface_hub import PyTorchModelHubMixin
import torch
import argparse

models_overrides = {
    "CC12m_LAION_Aesthetics_6_RIN_ldm_256_small_cad": "cad_256",
    "CC12M_LAION_Aesthetics6_512_LDM_RIN_grafted_latents_cooldown_aes_6_25": "cad_512",
}


class CAD(
    CADRINTextCond,
    PyTorchModelHubMixin,
    repo_url="https://github.com/nicolas-dufour/CAD",
    pipeline_tag="text-to-image",
    tags=["cad", "diffusion", "arxiv:2405.20324"],
    license="mit",
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def upload_model(checkpoint_dir, repo_name):
    import hydra
    from omegaconf import OmegaConf

    hydra.initialize(version_base=None, config_path=f"../configs")
    cfg = hydra.compose(
        config_name="config",
        overrides=[
            f"overrides={models_overrides[checkpoint_dir]}",
            "model/precomputed_text_embeddings='no'",
        ],
    )
    network_config = cfg.model.network
    serialized_network_config = OmegaConf.to_container(network_config, resolve=True)
    print(serialized_network_config)
    del serialized_network_config["_target_"]
    model = CAD(**serialized_network_config)
    ckpt = torch.load(f"cad/checkpoints/{checkpoint_dir}/last_coherence.ckpt")
    ckpt_state_dict = ckpt["state_dict"]
    ckpt_state_dict = {k: v for k, v in ckpt_state_dict.items() if "ema_network" in k}
    ckpt_state_dict = {
        k.replace("ema_network.", ""): v for k, v in ckpt_state_dict.items()
    }
    model.load_state_dict(ckpt_state_dict)
    model.push_to_hub(repo_name, commit_message="Fixed ckpt keys")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--repo_name", type=str, required=True)
    args = parser.parse_args()
    upload_model(args.checkpoint_dir, args.repo_name)
