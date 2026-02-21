#!/usr/bin/env python3
# fmt: off
import dotenv
# read env variables before anything else is imported
dotenv.load_dotenv(
    override=True
)  # set env variables from .env file, override=True is important
found_secrets = dotenv.load_dotenv(".secrets.env", override=True)
if not found_secrets:
    print("Warning: .secrets.env not found")
# fmt: on

import shlex
import sys
from typing import Dict
import re
from pathlib import Path
from typing import cast
import hydra
from omegaconf import DictConfig
from simple_slurm import Slurm
import omegaconf

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
    "config_path": str(Path("../configs") / "slurm"),
    "config_name": "nll_sbatch.yaml",
}


# resolvers


def validate_config(cfg: DictConfig) -> None:
    pass


def construct_job_name(cfg: DictConfig) -> str:
    base_job_name = cfg.job_name
    # add generative perplexity
    generative_perplexity = (
        str(cfg.eval.generative_perplexity).split("_")[0]
        if cfg.eval.get("generative_perplexity", None) is not None
        else "null"
    )
    # checkpoint
    checkpoint_path = Path(cfg.eval.checkpoint_path).stem
    return f"{base_job_name}_{generative_perplexity}_{checkpoint_path}"


@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> None:
    """Main function to configure and submit SLURM job."""
    # Collect overrides for the inner script
    validate_config(cfg)
    # determine the logs folder

    job_name = construct_job_name(cfg)
    cfg.job_name = job_name
    logs_dir = Path(cfg.paths.log_dir) / job_name
    run_dir = Path(cfg.paths.run_dir)
    # slurm_output_file = logs_dir / "%x.out"
    slurm_output_file = run_dir / "%x.out"
    slurm_config = cast(
        Dict, omegaconf.OmegaConf.to_container(cfg.slurm, resolve=True)
    )
    slurm_config["output"] = str(slurm_output_file)
    # Configure SLURM settings from config
    slurm = Slurm(**slurm_config)
    # add job_name

    # Set environment variables using slurm.add_cmd
    for key, value in cfg.env.items():
        slurm.add_cmd(f"export {key}={value}")

    if cfg.eval.get("generative_perplexity", None) is not None:
        experiment = (
            "["
            + str(cfg.eval.experiment)
            + ","
            + str(cfg.eval.generative_perplexity)
            + "]"
        )
    else:
        experiment = str(cfg.eval.experiment)
    checkpoint_path = str(cfg.eval.checkpoint_path)
    try:
        epoch_step = Path(checkpoint_path).stem.split("-")
        if len(epoch_step) == 2:
            checkpoint_epoch = int(epoch_step[0])
            checkpoint_step = int(epoch_step[1])
        else:
            checkpoint_epoch = None
            checkpoint_step = None
    except ValueError:
        checkpoint_epoch = None
        checkpoint_step = None
    # Main training command with srun
    if cfg.eval.debug is None:
        if "infill" in str(cfg.eval.experiment):
            cmd = [
                "python",
                "-O",
                "src/xlm/commands/lightning_main.py",
                f"job_name={job_name}",
                f"job_type={cfg.eval.job_type}",
                f"experiment={experiment}",
                f"++eval.checkpoint_path={checkpoint_path}",
                f"per_device_batch_size={cfg.eval.batch_size}",
                f"per_device_val_batch_size={cfg.eval.batch_size}",
                f"global_batch_size={cfg.eval.batch_size}",
                "trainer_strategy=single_device",
                f"++trainer.precision={cfg.eval.precision}",  # sample in 32-bit precision
                "compile=false",
                "+loggers.wandb.resume=allow",
                f"+loggers.wandb.id={job_name if cfg.get('use_job_name_as_id', True) else 'null'}",
                f"+tags.eval_type={cfg.eval.eval_type}",
                f"+tags.generative_perplexity={str(cfg.eval.generative_perplexity).split('_')[0] if cfg.eval.get('generative_perplexity', None) is not None else 'null'}",
                f"+tags.checkpoint={Path(checkpoint_path).stem}",
                # "~datamodule.dataset_managers.val.lm",
                "~datamodule.dataset_managers.val.unconditional_prediction=null",
                # f"datamodule.dataset_managers.test.unconditional_prediction.num_examples={cfg.eval.num_examples}",
            ]
        else:
            cmd = [
                "python",
                "-O",
                "src/xlm/commands/lightning_main.py",
                f"job_name={job_name}",
                f"job_type={cfg.eval.job_type}",
                f"experiment={experiment}",
                f"++eval.checkpoint_path={checkpoint_path}",
                f"per_device_batch_size={cfg.eval.batch_size}",
                f"per_device_val_batch_size={cfg.eval.batch_size}",
                f"global_batch_size={cfg.eval.batch_size}",
                "trainer_strategy=single_device",
                f"++trainer.precision={cfg.eval.precision}",  # sample in 32-bit precision
                "compile=false",
                "+loggers.wandb.resume=allow",
                f"+loggers.wandb.id={job_name if cfg.get('use_job_name_as_id', True) else 'null'}",
                f"+tags.eval_type={cfg.eval.eval_type}",
                f"+tags.generative_perplexity={str(cfg.eval.generative_perplexity).split('_')[0] if cfg.eval.get('generative_perplexity', None) is not None else 'null'}",
                f"+tags.checkpoint={Path(checkpoint_path).stem}",
                f"datamodule.dataset_managers.val.unconditional_prediction.num_examples={cfg.eval.num_examples}",
                # "~datamodule.dataset_managers.val.unconditional_prediction=null",
                "~datamodule.dataset_managers.val.lm",
                # f"datamodule.dataset_managers.test.unconditional_prediction.num_examples={cfg.eval.num_examples}",
            ]
        if checkpoint_epoch is not None and checkpoint_step is not None:
            cmd += [
                f"++checkpoint_epoch={checkpoint_epoch}",
                f"++checkpoint_step={checkpoint_step}",
            ]
    else:
        raise ValueError("debug mode is not supported for evaluation")

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
