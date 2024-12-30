import os
from pathlib import Path


class JeanZayExperiment:
    def __init__(
        self,
        exp_name,
        job_name,
        slurm_array_nb_jobs=None,
        num_nodes=1,
        num_gpus_per_node=1,
        qos="t3",
        account="syq",
        gpu_type="v100",
        cmd_path="train.py",
        time=None,
        launch_from_compute_node=False,
        min_time=None,
    ):
        self.expname = exp_name
        self.job_name = job_name
        self.nodes = num_nodes
        self.num_gpus_per_node = num_gpus_per_node
        self.qos = qos
        self.account = account
        self.gpu_type = gpu_type
        self.slurm_array_nb_jobs = slurm_array_nb_jobs
        self.cmd_path = cmd_path
        self.time = time
        self.min_time = min_time
        self.launch_from_compute_node = launch_from_compute_node

    def build_cmd(self, hydra_args):
        hydra_modifiers = []

        for hydra_arg, value in hydra_args.items():
            if hydra_arg.startswith("--"):
                hydra_modifiers.append(f" {hydra_arg} {value}")
            else:
                hydra_modifiers.append(f" {hydra_arg}={value}")
        self.cmd = f"python {self.cmd_path} {''.join(hydra_modifiers)}"
        print(f"srun {self.cmd}")

    def launch(self, debug=False):
        if debug:
            self.qos = "dev"
            self.time = "01:00:00"
            self.min_time = None
        if not hasattr(self, "cmd"):
            raise ValueError("Run build_cmd first")
        if self.qos == "t4":
            self.qos_name = "qos_gpu-t4"
            self.time = "99:59:59" if self.time is None else self.time
        elif self.qos == "t3":
            self.qos_name = "qos_gpu-t3"
            self.time = "19:59:59" if self.time is None else self.time
        elif self.qos == "dev":
            self.qos_name = "qos_gpu-dev"
            self.time = "01:59:59" if self.time is None else self.time

        else:
            raise ValueError("Not a valid QoS")

        if self.gpu_type == "a100":
            self.gpu_slurm_directive = "#SBATCH -C a100"
            self.cpus_per_task = 8
            self.qos_name = self.qos_name.replace("gpu_", "gpu_a100")
        elif self.gpu_type == "h100":
            self.gpu_slurm_directive = "#SBATCH -C h100"
            self.cpus_per_task = 24
            self.qos_name = self.qos_name.replace("gpu_", "gpu_h100")
        elif self.gpu_type == "v100":
            self.gpu_slurm_directive = "#SBATCH -C v100-32g"
            self.cpus_per_task = 10
        else:
            raise ValueError("Not a valid GPU type")

        local_slurmfolder = Path("checkpoints") / Path(self.expname) / Path("slurm")
        local_slurmfolder.mkdir(parents=True, exist_ok=True)
        slurm_path = local_slurmfolder / ("job_file" + ".slurm")
        if type(self.slurm_array_nb_jobs) is int:
            sbatch_array = f"#SBATCH --array=0-{self.slurm_array_nb_jobs-1}"
        elif type(self.slurm_array_nb_jobs) is list:
            sbatch_array = f"#SBATCH --array={','.join([str(i) for i in self.slurm_array_nb_jobs])}"
        elif self.slurm_array_nb_jobs is None:
            sbatch_array = ""
        else:
            raise ValueError("Not a valid type for slurm_array_nb_jobs")
        slurm = f"""#!/bin/bash
#SBATCH --job-name={self.job_name}
{sbatch_array}
#SBATCH --nodes={self.nodes}	# number of nodes
#SBATCH --account={self.account}@{self.gpu_type}
#SBATCH --ntasks-per-node={self.num_gpus_per_node}
#SBATCH --gres=gpu:{self.num_gpus_per_node}
#SBATCH --qos={self.qos_name}
{self.gpu_slurm_directive}
#SBATCH --cpus-per-task={self.cpus_per_task}
#SBATCH --hint=nomultithread
#SBATCH --time={self.time}
{"#SBATCH --time-min=" + self.min_time if self.min_time is not None else ""}
#SBATCH --output=/gpfswork/rech/syq/uey53ph/diffusion/{local_slurmfolder}/job_%j.out
#SBATCH --error=/gpfswork/rech/syq/uey53ph/diffusion/{local_slurmfolder}/job_%j.err
#SBATCH --signal=SIGUSR1@90
module purge
{"module load arch/{self.gpu_type}" if self.gpu_type in ["a100", "h100"] else ""}
module load pytorch-gpu/py3/2.2.0
source /linkhome/rech/genlgm01/uey53ph/.venvs/diffusion/bin/activate

export PYTHONPATH=/linkhome/rech/genlgm01/uey53ph/.venvs/diffusion/bin/python
export TRANSFORMERS_OFFLINE=1 # to avoid downloading
export HYDRA_FULL_ERROR=1 # to have the full traceback
export WANDB_CACHE_DIR=$NEWSCRATCH/wandb_cache
export TMPDIR=$JOBSCRATCH
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export IS_CLUSTER=True
set -x
srun {self.cmd}
        """
        with open(slurm_path, "w") as slurm_file:
            slurm_file.write(slurm)
        if self.launch_from_compute_node:
            os.system('unset $(env | egrep "SLURM_|SBATCH_"| cut -d= -f1)')
        os.system(f"sbatch {slurm_path}")
