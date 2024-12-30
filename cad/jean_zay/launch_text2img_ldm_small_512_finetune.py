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

# exp_name = f"CC12m_LAION_Aesthetics_6_RIN_ldm_512_small_cad"
# job_name = f"512_finetune"
# jz_exp = JeanZayExperiment(exp_name, job_name)
# jz_exp.nodes = 1
# jz_exp.num_gpus_per_node = 8
# jz_exp.qos = "t3"
# jz_exp.account = "syq"
# jz_exp.gpu_type = "a100"
# jz_exp.time = "20:00:00"

# exps.append(jz_exp)

# trainer_modifiers = {
#     "computer": "cluster-node-a100.yaml",
#     "computer.devices": jz_exp.num_gpus_per_node,
#     "computer.progress_bar_refresh_rate": 10,
#     "computer.num_nodes": jz_exp.nodes,
#     "data_dir": Path(os.environ["NEWSCRATCH"]),
# }

# exp_modifier = {
#     "overrides": "cad_256",
#     "computer.precision": "16-mixed",
#     "model.start_ema_step": 250000,
#     "data.img_resolution": 512,
#     "data.data_resolution": 64,
#     "experiment_name": "CC12m_LAION_Aesthetics_6_RIN_ldm_512_small_cad",
#     "trainer.max_steps": 500000,
#     "model.optimizer.optim.lr": 0.0002,
# }

# cmd_modifiers.append(dict(trainer_modifiers, **exp_modifier))

# exp_name = f"CC12M_LAION_Aesthetics6_512_LDM_RIN_grafted_latents_and_blocks"
# job_name = f"l+b_512_finetune"
# jz_exp = JeanZayExperiment(exp_name, job_name)
# jz_exp.nodes = 2
# jz_exp.num_gpus_per_node = 8
# jz_exp.qos = "t3"
# jz_exp.account = "syq"
# jz_exp.gpu_type = "a100"
# jz_exp.time = "20:00:00"

# exps.append(jz_exp)

# trainer_modifiers = {
#     "computer": "cluster-node-a100.yaml",
#     "computer.devices": jz_exp.num_gpus_per_node,
#     "computer.progress_bar_refresh_rate": 10,
#     "computer.num_nodes": jz_exp.nodes,
#     "data_dir": Path(os.environ["NEWSCRATCH"]),
# }

# exp_modifier = {
#     "overrides": "cad_512",
#     "computer.precision": "16-mixed",
#     "experiment_name": "CC12M_LAION_Aesthetics6_512_LDM_RIN_grafted_latents_and_blocks",
#     "trainer.max_steps": 500000,
#     "model.optimizer.optim.lr": 0.0001,
# }

# cmd_modifiers.append(dict(trainer_modifiers, **exp_modifier))

# exp_name = f"CC12M_LAION_Aesthetics6_512_LDM_RIN_grafted_latents"
# job_name = f"l_512_finetune"
# jz_exp = JeanZayExperiment(exp_name, job_name)
# jz_exp.nodes = 2
# jz_exp.num_gpus_per_node = 8
# jz_exp.qos = "t3"
# jz_exp.account = "syq"
# jz_exp.gpu_type = "a100"
# jz_exp.time = "20:00:00"

# exps.append(jz_exp)

# trainer_modifiers = {
#     "computer": "cluster-node-a100.yaml",
#     "computer.devices": jz_exp.num_gpus_per_node,
#     "computer.progress_bar_refresh_rate": 10,
#     "computer.num_nodes": jz_exp.nodes,
#     "data_dir": Path(os.environ["NEWSCRATCH"]),
# }

# exp_modifier = {
#     "overrides": "cad_512",
#     "model/lr_scheduler": "warmup_sqrt_decay",
#     "model.network.num_blocks": 4,
#     "experiment_name": exp_name,
#     "trainer.max_steps": 200000,
#     "model.optimizer.optim.lr": 8.686031504834843e-05
# }

# cmd_modifiers.append(dict(trainer_modifiers, **exp_modifier))

exp_name = f"CC12M_LAION_Aesthetics6_512_LDM_RIN_grafted_latents_cooldown_aes_6"
job_name = f"l_512_finetune"
jz_exp = JeanZayExperiment(exp_name, job_name)
jz_exp.nodes = 2
jz_exp.num_gpus_per_node = 8
jz_exp.qos = "t3"
jz_exp.account = "syq"
jz_exp.gpu_type = "a100"
jz_exp.time = "20:00:00"

exps.append(jz_exp)

trainer_modifiers = {
    "computer": "cluster-node-a100.yaml",
    "computer.devices": jz_exp.num_gpus_per_node,
    "computer.progress_bar_refresh_rate": 10,
    "computer.num_nodes": jz_exp.nodes,
    "data_dir": Path(os.environ["NEWSCRATCH"]),
}

exp_modifier = {
    "overrides": "cad_512",
    "model/lr_scheduler": "warmup_sqrt_decay",
    "model.network.num_blocks": 4,
    "experiment_name": exp_name,
    "trainer.precision": 32,
    "trainer.max_steps": 200000,
    "model.optimizer.optim.lr": 8.686031504834843e-05,
    "data.train_instance.aesthetic_threshold": 6,
}

cmd_modifiers.append(dict(trainer_modifiers, **exp_modifier))

exp_name = f"CC12M_LAION_Aesthetics6_512_LDM_RIN_grafted_latents_cooldown_aes_6_25"
job_name = f"l_512_finetune"
jz_exp = JeanZayExperiment(exp_name, job_name)
jz_exp.nodes = 2
jz_exp.num_gpus_per_node = 8
jz_exp.qos = "t3"
jz_exp.account = "syq"
jz_exp.gpu_type = "a100"
jz_exp.time = "20:00:00"

exps.append(jz_exp)

trainer_modifiers = {
    "computer": "cluster-node-a100.yaml",
    "computer.devices": jz_exp.num_gpus_per_node,
    "computer.progress_bar_refresh_rate": 10,
    "computer.num_nodes": jz_exp.nodes,
    "data_dir": Path(os.environ["NEWSCRATCH"]),
}

exp_modifier = {
    "overrides": "cad_512",
    "model/lr_scheduler": "warmup_sqrt_decay",
    "model.network.num_blocks": 4,
    "experiment_name": exp_name,
    "trainer.precision": 32,
    "trainer.max_steps": 200000,
    "model.optimizer.optim.lr": 8.686031504834843e-05,
    "data.train_instance.aesthetic_threshold": 6.25,
}

cmd_modifiers.append(dict(trainer_modifiers, **exp_modifier))

# exp_name = f"CC12M_LAION_Aesthetics6_512_LDM_RIN_grafted_blocks"
# job_name = f"b_512_finetune"
# jz_exp = JeanZayExperiment(exp_name, job_name)
# jz_exp.nodes = 2
# jz_exp.num_gpus_per_node = 8
# jz_exp.qos = "t3"
# jz_exp.account = "syq"
# jz_exp.gpu_type = "a100"
# jz_exp.time = "20:00:00"

# exps.append(jz_exp)

# trainer_modifiers = {
#     "computer": "cluster-node-a100.yaml",
#     "computer.devices": jz_exp.num_gpus_per_node,
#     "computer.progress_bar_refresh_rate": 10,
#     "computer.num_nodes": jz_exp.nodes,
#     "data_dir": Path(os.environ["NEWSCRATCH"]),
# }

# exp_modifier = {
#     "overrides": "cad_512",
#     "model.network.num_latents": 256,
#     "computer.precision": "16-mixed",
#     "experiment_name": "CC12M_LAION_Aesthetics6_512_LDM_RIN_grafted_blocks",
#     "trainer.max_steps": 500000,
#     "model.optimizer.optim.lr": 0.0001,
# }

# cmd_modifiers.append(dict(trainer_modifiers, **exp_modifier))

if __name__ == "__main__":
    for exp, cmd_modifier in zip(exps, cmd_modifiers):
        args = parse_mode()
        exp.build_cmd(cmd_modifier)
        if args.launch == True:
            exp.launch()
