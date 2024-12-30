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
    "cad_256": "CC12m_LAION_Aesthetics_6_RIN_ldm_256_small_cad",
}
coherences = [i / 7 for i in range(8)]
for model_override, model_name in models.items():
    for coherence in coherences:
        exp_name = model_name
        job_name = f"conf"
        jz_exp = JeanZayExperiment(exp_name, job_name)
        jz_exp.nodes = 1
        jz_exp.num_gpus_per_node = 1
        jz_exp.qos = "t3"
        jz_exp.account = "syq"
        jz_exp.gpu_type = "a100"
        jz_exp.time = "02:30:00"
        jz_exp.cmd_path = "validate.py"

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
            "overrides": model_override,
            "computer.precision": "16-mixed",
            "model.start_ema_step": 250000,
            "data.val_instance.return_image": True,
            "data.val_instance.return_text": True,
            "data.full_batch_size": 64,
            "model/precomputed_text_embeddings": "'no'",
            "data.val_instance.text_embedding_name": "null",
            "data.val_instance.vae_embedding_name": "null",
            "model.cfg_rate": 5.0,
            "model.coherence_value": coherence,
        }

        cmd_modifiers.append(dict(trainer_modifiers, **exp_modifier))


if __name__ == "__main__":
    for exp, cmd_modifier in zip(exps, cmd_modifiers):
        args = parse_mode()
        exp.build_cmd(cmd_modifier)
        if args.launch == True:
            exp.launch()
