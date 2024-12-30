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
    "Imagenet_256_LDM_Large": "imagenet_256_rin_ldm_large",
    "Imagenet_256_LDM_channel_wise_vae": "imagenet_256_rin_ldm" "",
}
cfgs = [0.25, 0.5, 0.75, 1.25, 1.5, 1.75]
for model_name, model_override in models.items():
    for cfg_rate in cfgs:
        exp_name = model_name
        job_name = f"small_cfg"
        jz_exp = JeanZayExperiment(exp_name, job_name)
        jz_exp.nodes = 1
        jz_exp.num_gpus_per_node = 1
        jz_exp.qos = "t3"
        jz_exp.account = "syq"
        jz_exp.gpu_type = "a100"
        jz_exp.time = "14:30:00"
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
            "model.channel_wise_normalisation": True,
            "model.data_preprocessing.vae_sample": True,
            "model.test_sampler.num_steps": 250,
            "data.full_batch_size": 32,
            # "data.val_instance.vae_embedding_name": "null",
            "model.cfg_rate": cfg_rate,
        }

        cmd_modifiers.append(dict(trainer_modifiers, **exp_modifier))


if __name__ == "__main__":
    for exp, cmd_modifier in zip(exps, cmd_modifiers):
        args = parse_mode()
        exp.build_cmd(cmd_modifier)
        if args.launch == True:
            exp.launch()
