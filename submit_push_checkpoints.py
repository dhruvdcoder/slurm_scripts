#!/usr/bin/env python3
# fmt: off
import os
from pathlib import Path

import dotenv

_project_root = Path(__file__).resolve().parent.parent.parent
dotenv.load_dotenv(_project_root / ".env", override=True)
found_secrets = dotenv.load_dotenv(_project_root / ".secrets.env", override=True)
if not found_secrets:
    print("Warning: .secrets.env not found")
# fmt: on

import re
import shlex
from typing import Dict, List, Tuple, cast

import hydra
import omegaconf
from omegaconf import DictConfig
from simple_slurm import Slurm

from common import get_experiment_string, steal_args
from hydra.core.plugins import Plugins
from searchpath_plugin import HydraCommonSearchPathPlugin

INNER_ARGS, OUTER_ARGS = steal_args()

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": str(Path(__file__).parent / "slurm"),
    "config_name": "push_checkpoints_sbatch",
}
Plugins.instance().register(HydraCommonSearchPathPlugin)


def _collect_checkpoint_jobs(cfg: DictConfig) -> List[Tuple[Path, str]]:
    """Return sorted list of (absolute checkpoint path, hub branch name)."""
    pc = cfg.push_checkpoints
    resolve_at = str(pc.get("resolve_checkpoints_at", "submit"))
    if resolve_at != "submit":
        raise NotImplementedError(
            "push_checkpoints.resolve_checkpoints_at!=submit is not implemented; "
            "glob checkpoints at submit time or extend submit_push_checkpoints.py."
        )

    ckpt_root = Path(pc.checkpoints_dir).expanduser().resolve()
    if not ckpt_root.is_dir():
        raise ValueError(f"checkpoints_dir is not a directory: {ckpt_root}")

    glob_pat = str(pc.get("checkpoint_glob", "*.ckpt"))
    pattern_str = str(pc.get("filename_pattern", r"^(\d+)-(\d+)\.ckpt$"))
    try:
        cre = re.compile(pattern_str)
    except re.error as e:
        raise ValueError(f"Invalid push_checkpoints.filename_pattern: {e}") from e

    branch_prefix = str(pc.get("branch_prefix", "step"))
    include_special = bool(pc.get("include_special_named", False))

    matches: List[Tuple[Path, str]] = []
    for path in sorted(ckpt_root.glob(glob_pat)):
        if not path.is_file():
            continue
        name = path.name
        m = cre.match(name)
        if m:
            step = m.group(2)
            branch = f"{branch_prefix}-{step}"
            matches.append((path.resolve(), branch))
            continue
        if include_special and name in ("best.ckpt", "last.ckpt"):
            stem = path.stem
            matches.append((path.resolve(), f"{branch_prefix}-{stem}"))

    return matches


def _build_one_push_cmd(
    cfg: DictConfig,
    job_name: str,
    experiment: str,
    ckpt_path: Path,
    branch: str,
    hub_repo_id: str,
) -> List[str]:
    pc = cfg.push_checkpoints
    weight_source = str(pc.get("weight_source", "lightning_ckpt"))

    commit_message = f"Push {ckpt_path.name} on branch {branch}"

    cmd: List[str] = [
        "xlm-push-to-hub",
        f"job_name={job_name}",
        f"experiment={experiment}",
        "hub=default",
        f"++hub.repo_id={hub_repo_id}",
        f"++hub.branch={branch}",
        f"++hub.commit_message={commit_message}",
    ]

    ckpt_str = str(ckpt_path)
    if weight_source == "lightning_ckpt":
        cmd.append(f"++hub_checkpoint_path={ckpt_str}")
    elif weight_source == "model_only":
        cmd.append("+skip_init_weights=True")
        cmd.append(f"++model_only_checkpoint_path={ckpt_str}")
    else:
        raise ValueError(
            f"Unknown push_checkpoints.weight_source={weight_source!r} "
            "(use lightning_ckpt or model_only)"
        )

    if INNER_ARGS:
        cmd.extend(INNER_ARGS)

    return cmd


@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> None:
    jobs = _collect_checkpoint_jobs(cfg)
    if not jobs:
        raise ValueError(
            "No checkpoints matched push_checkpoints glob/pattern under "
            f"{cfg.push_checkpoints.checkpoints_dir}. "
            "Check filename_pattern, checkpoint_glob, and include_special_named."
        )

    job_name_base = str(cfg.job_name)
    cfg.job_name = job_name_base

    run_dir = Path(cfg.paths.run_dir)
    slurm_output_file = run_dir / "%x.out"
    slurm_config = cast(
        Dict, omegaconf.OmegaConf.to_container(cfg.slurm, resolve=True)
    )
    slurm_config["output"] = str(slurm_output_file)
    slurm = Slurm(**slurm_config)

    for key, value in cfg.env.items():
        slurm.add_cmd(f"export {key}={value}")

    for env_key in ("HF_HUB_KEY", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        val = os.environ.get(env_key)
        if val:
            slurm.add_cmd(f"export {env_key}={shlex.quote(val)}")

    experiment = get_experiment_string(cfg)
    hub_repo_id = str(cfg.push_checkpoints.hub_repo_id)

    for ckpt_path, branch in jobs:
        inner_job = f"{job_name_base}__{branch.replace('/', '_')}"
        cmd = _build_one_push_cmd(
            cfg,
            job_name=inner_job,
            experiment=experiment,
            ckpt_path=ckpt_path,
            branch=branch,
            hub_repo_id=hub_repo_id,
        )
        quoted = [shlex.quote(arg) for arg in cmd]
        slurm.add_cmd("srun " + " ".join(quoted))

    script = slurm.script()
    print("Generated SLURM script:")
    print(script)

    if cfg.do == "submit":
        script_file = run_dir / "sbatch.sh"
        script_file.parent.mkdir(parents=True, exist_ok=True)
        with open(script_file, "w", encoding="utf-8") as f:
            f.write(script)
        print(f"\nSLURM script saved to: {script_file}")

        job_id = slurm.sbatch()
        print(f"Submitted job with ID: {job_id}")


if __name__ == "__main__":
    main()
