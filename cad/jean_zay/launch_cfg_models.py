import argparse
import os
from pathlib import Path

from launch import JeanZayExperiment


def parse_mode():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()

    return args


cmd_modifiers = []
exps = []

models = {
    "CC12m_LAION_Aesthetics_6_RIN_ldm_256_small_cad": "cad_256",
    "CC12M_LAION_Aesthetics6_512_LDM_RIN_grafted_latents_cooldown_aes_6_25": "cad_512",
}
cfgs = [1.0, 3.0, 5.0, 10.0, 15.0, 30.0, 50.0, 70.0, 100.0]
for model_name, model_override in models.items():
    for cfg_rate in cfgs:
        exp_name = model_name
        job_name = f"cfg"
        jz_exp = JeanZayExperiment(exp_name, job_name)
        jz_exp.nodes = 1
        jz_exp.num_gpus_per_node = 1
        jz_exp.qos = "t3"
        jz_exp.account = "syq"
        jz_exp.gpu_type = "a100"
        jz_exp.time = "20:00:00"
        jz_exp.cmd_path = "test.py"

        exps.append(jz_exp)

        trainer_modifiers = {
            "experiment_name": exp_name,
            "computer": "a100.yaml",
            "computer.devices": jz_exp.num_gpus_per_node,
            "computer.progress_bar_refresh_rate": 10,
            "computer.num_nodes": jz_exp.nodes,
            "data_dir": Path(os.environ["NEWSCRATCH"]),
        }

        exp_modifier = {
            "+checkpoint_name": "last.ckpt",
            "overrides": model_override,
            "computer.precision": "16-mixed",
            "data.val_instance.return_image": True,
            "data.val_instance.return_text": True,
            "data.full_batch_size": 32,
            "data.val_instance.root": Path(os.environ["NEWSCRATCH"]) / Path("coco_10k"),
            "model/precomputed_text_embeddings": "'no'",
            "data.val_instance.text_embedding_name": "null",
            "data.val_instance.vae_embedding_name_mean": "null",
            "model.cfg_rate": cfg_rate,
        }

        cmd_modifiers.append(dict(trainer_modifiers, **exp_modifier))


if __name__ == "__main__":
    for exp, cmd_modifier in zip(exps, cmd_modifiers):
        args = parse_mode()
        exp.build_cmd(cmd_modifier)
        if args.launch == True:
            exp.launch()
