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
# exp_name = f"CC12m_LAION_Aesthetics_6_RIN_ldm_256_tiny"
# job_name = f"ldm_tiny"
# jz_exp = JeanZayExperiment(exp_name, job_name)
# jz_exp.nodes = 1
# jz_exp.num_gpus_per_node = 8
# jz_exp.qos = "t3"
# jz_exp.account = "syq"
# jz_exp.gpu_type = "a100"
# jz_exp.time = "20:00:00"

# exps.append(jz_exp)

# trainer_modifiers = {
#     "experiment_name": exp_name,
#     "computer": "cluster-node-a100.yaml",
#     "computer.devices": jz_exp.num_gpus_per_node,
#     "computer.progress_bar_refresh_rate": 10,
#     "computer.num_nodes": jz_exp.nodes,
#     "data_dir": Path(os.environ["NEWSCRATCH"]),
# }

# exp_modifier = {
#     "overrides": "cc12m_256_rin_tiny_ldm",
#     "computer.precision": "16-mixed",
# }

# cmd_modifiers.append(dict(trainer_modifiers, **exp_modifier))

exp_name = f"CC12m_LAION_Aesthetics_6_RIN_ldm_256_tiny_cad"
job_name = f"cad_ldm_tiny"
jz_exp = JeanZayExperiment(exp_name, job_name)
jz_exp.nodes = 1
jz_exp.num_gpus_per_node = 8
jz_exp.qos = "t3"
jz_exp.account = "syq"
jz_exp.gpu_type = "a100"
jz_exp.time = "20:00:00"
jz_exp.min_time = "10:00:00"

exps.append(jz_exp)

trainer_modifiers = {
    "experiment_name": exp_name,
    "computer": "cluster-node-a100.yaml",
    "computer.devices": jz_exp.num_gpus_per_node,
    "computer.progress_bar_refresh_rate": 10,
    "computer.num_nodes": jz_exp.nodes,
    "data_dir": Path(os.environ["NEWSCRATCH"]),
}

exp_modifier = {
    "overrides": "cc12m_256_rin_tiny_ldm_cad",
    "trainer.max_steps": 1000000,
    "computer.precision": "16-mixed",
}

cmd_modifiers.append(dict(trainer_modifiers, **exp_modifier))


# exp_name = f"CC12m_LAION_Aesthetics_6_RIN_ldm_256_tiny_filtered"
# job_name = f"filtered_tiny"
# jz_exp = JeanZayExperiment(exp_name, job_name)
# jz_exp.nodes = 1
# jz_exp.num_gpus_per_node = 8
# jz_exp.qos = "t3"
# jz_exp.account = "syq"
# jz_exp.gpu_type = "a100"
# jz_exp.time = "20:00:00"

# exps.append(jz_exp)

# trainer_modifiers = {
#     "experiment_name": exp_name,
#     "computer": "cluster-node-a100.yaml",
#     "computer.devices": jz_exp.num_gpus_per_node,
#     "computer.progress_bar_refresh_rate": 10,
#     "computer.num_nodes": jz_exp.nodes,
#     "data_dir": Path(os.environ["NEWSCRATCH"]),
# }

# exp_modifier = {
#     "overrides": "cc12m_256_rin_tiny_ldm_filtered",
#     "computer.precision": "16-mixed",
# }

# cmd_modifiers.append(dict(trainer_modifiers, **exp_modifier))

# exp_name = f"CC12m_LAION_Aesthetics_6_RIN_ldm_256_tiny_weighted"
# job_name = f"ldm_tiny"
# jz_exp = JeanZayExperiment(exp_name, job_name)
# jz_exp.nodes = 1
# jz_exp.num_gpus_per_node = 8
# jz_exp.qos = "t3"
# jz_exp.account = "syq"
# jz_exp.gpu_type = "a100"
# jz_exp.time = "20:00:00"

# exps.append(jz_exp)

# trainer_modifiers = {
#     "experiment_name": exp_name,
#     "computer": "cluster-node-a100.yaml",
#     "computer.devices": jz_exp.num_gpus_per_node,
#     "computer.progress_bar_refresh_rate": 10,
#     "computer.num_nodes": jz_exp.nodes,
#     "data_dir": Path(os.environ["NEWSCRATCH"]),
# }

# exp_modifier = {
#     "overrides": "cc12m_256_rin_tiny_ldm_weighted",
#     "computer.precision": "16-mixed",
# }

# cmd_modifiers.append(dict(trainer_modifiers, **exp_modifier))


if __name__ == "__main__":
    for exp, cmd_modifier in zip(exps, cmd_modifiers):
        args = parse_mode()
        exp.build_cmd(cmd_modifier)
        if args.launch == True:
            exp.launch()
