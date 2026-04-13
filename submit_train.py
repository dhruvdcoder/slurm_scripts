#!/usr/bin/env python3
# fmt: off
import os
from pathlib import Path

import dotenv

# Load from project root (parent of xlm-core) so it works regardless of cwd
_project_root = Path(__file__).resolve().parent.parent
dotenv.load_dotenv(_project_root / ".env", override=True)
found_secrets = dotenv.load_dotenv(_project_root / ".secrets.env", override=True)
if not found_secrets:
    print("Warning: .secrets.env not found")
# fmt: on

import shlex
import sys
import re
from typing import Dict, cast
import hydra
from omegaconf import DictConfig

# simple_slurm parses SQUEUE_FORMAT at import time; Alliance/Killarney sets it
# to a non-CSV format that simple_slurm can't handle, so we clear it first.
os.environ.pop("SQUEUE_FORMAT", None)

from simple_slurm import Slurm
import omegaconf
from common import get_experiment_string
from hydra.core.plugins import Plugins


# steal the raw args before Hydra’s decorator runs
RAW_ARGS = sys.argv[1:]
if "---" in RAW_ARGS:
    sep = RAW_ARGS.index("---")
    INNER_ARGS = RAW_ARGS[sep + 1 :]
    OUTER_ARGS = RAW_ARGS[:sep]
else:
    INNER_ARGS = []
    OUTER_ARGS = RAW_ARGS

# rewrite sys.argv so Hydra only sees the outer bits
sys.argv = [sys.argv[0]] + OUTER_ARGS

# Hydra configuration parameters
_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": str(Path(__file__).parent / "slurm"),
    "config_name": "train_sbatch",
}
from searchpath_plugin import HydraCommonSearchPathPlugin
Plugins.instance().register(HydraCommonSearchPathPlugin)

# resolvers
def _parse_gpu_count(gres: str) -> int:
    # Handles gpu:4, gpu:h100:4, gpu:a100:4, gpu:l40s:4, etc.
    match = re.search(r"gpu:(?:[^:]+:)?(\d+)", gres)
    if match:
        return int(match.group(1))
    raise ValueError(f"Invalid gres: {gres}")


def _determine_trainer_strategy(
    ntasks_per_node: int, nodes: int, hooks: bool = True
) -> str:
    if ntasks_per_node == 1 and nodes == 1:
        return "single_device"
    if ntasks_per_node >= 1 and nodes > 1 and hooks:
        return "ddp_multinode"
    if ntasks_per_node >= 1 and nodes > 1 and not hooks:
        return "ddp_multinode_no_hooks"
    if ntasks_per_node >= 1 and nodes == 1:
        return "ddp"
    return "single_device"


omegaconf.OmegaConf.register_new_resolver("parse_gpu_count", _parse_gpu_count)
omegaconf.OmegaConf.register_new_resolver(
    "determine_trainer_strategy", _determine_trainer_strategy
)


def validate_config(cfg: DictConfig) -> None:
    if cfg.train.debug is not None:
        # check the trainer_strategy, devices, num_nodes, precision, compile
        if cfg.train.trainer_strategy in ["ddp_multinode", "ddp"]:
            print("[Warning] Using debug mode with multi-node training")
        if cfg.train.devices > 1:
            print("[Warning] Using debug mode with multi-GPU training")
        if cfg.train.num_nodes > 1:
            print("[Warning] Using debug mode with multi-node training")


@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> None:
    """Main function to configure and submit SLURM job."""
    # Collect overrides for the inner script
    validate_config(cfg)
    # determine the logs folder
    logs_dir = Path(cfg.paths.log_dir) / cfg.job_name
    run_dir = Path(cfg.paths.run_dir)
    # slurm_output_file = logs_dir / "%x.out"
    slurm_output_file = run_dir / "%x.out"
    slurm_config = cast(
        Dict, omegaconf.OmegaConf.to_container(cfg.slurm, resolve=True)
    )
    # simple_slurm emits "None" as a literal string rather than omitting the
    # flag, so filter out any null/None values before building the Slurm object.
    slurm_config = {k: v for k, v in slurm_config.items() if v is not None}
    slurm_config["output"] = str(slurm_output_file)
    # Configure SLURM settings from config
    slurm = Slurm(**slurm_config)
    # add job_name

    # Emit pre-commands (module loads, venv activation, etc.) before env exports
    for cmd in cfg.get("pre_cmd", []):
        slurm.add_cmd(cmd)

    # Set environment variables using slurm.add_cmd
    for key, value in cfg.env.items():
        slurm.add_cmd(f"export {key}={value}")
    # Pass wandb vars to the job (from .env / .secrets.env)
    for key in ("WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT"):
        val = os.environ.get(key)
        if val:
            slurm.add_cmd(f"export {key}={shlex.quote(val)}")

    # Get wandb job ID from command line args or use SLURM_JOB_NAME
    job_name = cfg.job_name

    # Print GPU info
    # slurm.add_cmd(
    #    "python -c 'import torch; print(\"num_gpus: \", torch.cuda.device_count())'"
    # )

    # Main training command with srun
    experiment = get_experiment_string(cfg.train)
    cmd = [
        "xlm",
        f"job_name={job_name}",
        f"job_type={cfg.train.job_type}",
        f"experiment={experiment}",
        f"loggers={cfg.train.loggers}",
    ]

    debug = cfg.train.get("debug")
    if debug is not None:
        cmd += [f"debug={debug}"]
    cmd += [
        f"per_device_batch_size={cfg.train.batch_size}",
        f"trainer_strategy={cfg.train.trainer_strategy}",
        f"trainer.devices={cfg.train.devices}",
        f"trainer.num_nodes={cfg.train.num_nodes}",
        f"++trainer.precision={cfg.train.precision}",
        f"compile={cfg.train.compile}",
        "+loggers.wandb.resume=allow",
    ]
    wandb_id = False
    for inner_arg in INNER_ARGS:
        if "wandb.id" in inner_arg:
            wandb_id = True
            break
    if not wandb_id:
        cmd += [
            f"+loggers.wandb.id={job_name if cfg.get('use_job_name_as_id', True) else 'null'}",
        ]

    if INNER_ARGS:
        cmd += INNER_ARGS

    # Add srun command with the training command
    quoted = [shlex.quote(arg) for arg in cmd]
    slurm.add_cmd("srun " + " ".join(quoted))
    script = slurm.script()
    # Print the generated bash script
    print("Generated SLURM script:")
    # always print the script to the console
    print(script)

    # Save the generated bash script to a file
    # Submit the job
    if cfg.do == "submit":
        script_file = run_dir / "sbatch.sh"
        script_file.parent.mkdir(parents=True, exist_ok=True)
        with open(script_file, "w") as f:
            f.write(script)
        print(f"\nSLURM script saved to: {script_file}")

        job_id = slurm.sbatch()
        print(f"Submitted job with ID: {job_id}")


if __name__ == "__main__":
    main()
