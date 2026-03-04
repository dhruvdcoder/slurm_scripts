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
from typing import Dict
from pathlib import Path
from typing import cast
import hydra
from omegaconf import DictConfig
from simple_slurm import Slurm
import omegaconf
from common import steal_args, remove_dms, get_experiment_string
from hydra.core.plugins import Plugins
from searchpath_plugin import HydraCommonSearchPathPlugin

# steal the raw args before Hydra’s decorator runs
INNER_ARGS, OUTER_ARGS = steal_args()

# Hydra configuration parameters
_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": str(Path(__file__).parent / "slurm"),
    "config_name": "eval_sbatch.yaml",
}
Plugins.instance().register(HydraCommonSearchPathPlugin)



def construct_job_name(cfg: DictConfig) -> str:
    base_job_name = cfg.job_name
    return base_job_name


@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> None:
    """Main function to configure and submit SLURM job."""
    # Collect overrides for the inner script

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
    experiment = get_experiment_string(cfg)
    checkpoint_path = str(cfg.eval.checkpoint_path)
    # Main training command with srun
    cmd = [
        "xlm",
        f"job_name={job_name}",
        f"job_type={cfg.eval.job_type}",
        f"experiment={experiment}",
        f"++eval.checkpoint_path={checkpoint_path}",
        f"++eval.split={cfg.eval.get('split', 'validation')}",
        "trainer_strategy=single_device",
        f"++trainer.precision={cfg.eval.precision}",  # sample in 32-bit precision
        "compile=false",
        "+loggers.wandb.resume=allow",
        f"+loggers.wandb.id={job_name if cfg.get('use_job_name_as_id', True) else 'null'}",
    ]
    _remove_dms = remove_dms(cfg)
    # add things like `~datamodule.dataset_managers.val.lm` to the command
    if _remove_dms:
        cmd += _remove_dms


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

    # Save  generated bash script to a file
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
