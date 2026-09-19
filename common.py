
import re
import sys
from typing import Any, Dict, List, Optional

import omegaconf
from omegaconf import DictConfig


def steal_args():
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
    return INNER_ARGS, OUTER_ARGS

def remove_dms(cfg, eval_or_train: str = "eval") -> List[str]:
    # DictConfig.get("eval.dms_to_remove") does not traverse dot paths; use select.
    path = f"{eval_or_train}.dms_to_remove"
    dms_to_remove = omegaconf.OmegaConf.select(cfg, path, default=[])
    if dms_to_remove is None:
        dms_to_remove = []
    remove_strs = []
    for dm in dms_to_remove:
        remove_strs.append(f"~datamodule.dataset_managers.{dm}")
    return remove_strs

def get_experiment_string(inner_cfg) -> str:
    experiment_value = inner_cfg.experiment
    if omegaconf.OmegaConf.is_list(experiment_value):
        experiment = f"[{','.join(experiment_value)}]"
    else:
        experiment = str(experiment_value)
    return experiment


def parse_gpu_count(value: Any) -> int:
    """Resolve a GPU count from slurm.gpus, slurm.gpus_per_node, or slurm.gres."""
    if value is None:
        raise ValueError(
            "No GPU count found in slurm.gpus, slurm.gpus_per_node, or slurm.gres"
        )
    if isinstance(value, bool):
        raise ValueError(f"Invalid GPU spec: {value}")
    if isinstance(value, int):
        return value
    text = str(value)
    if text.isdigit():
        return int(text)
    match = re.search(r"gpu:(\d+)", text)
    if match:
        return int(match.group(1))
    raise ValueError(f"Invalid GPU spec: {value}")


def register_gpu_count_resolver() -> None:
    if omegaconf.OmegaConf.has_resolver("parse_gpu_count"):
        return
    omegaconf.OmegaConf.register_new_resolver("parse_gpu_count", parse_gpu_count)


def slurm_kwargs(cfg: DictConfig, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convert cfg.slurm to simple_slurm kwargs, dropping null keys."""
    slurm_config = omegaconf.OmegaConf.to_container(cfg.slurm, resolve=True)
    if not isinstance(slurm_config, dict):
        raise TypeError(f"cfg.slurm must resolve to a dict, got {type(slurm_config)}")
    cleaned = {key: value for key, value in slurm_config.items() if value is not None}
    if extra:
        cleaned.update(extra)
    return cleaned


def is_aicr_slurm(cfg: DictConfig) -> bool:
    partition = str(omegaconf.OmegaConf.select(cfg, "slurm.partition", default="") or "")
    return any(token in partition for token in ("rtx-", "b200-"))


def _slurm_time_hours(time_val: Any) -> Optional[float]:
    if time_val is None:
        return None
    text = str(time_val)
    days = 0
    if "-" in text:
        day_part, text = text.split("-", 1)
        days = int(day_part)
    parts = [int(part) for part in text.split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes = parts
        seconds = 0
    elif len(parts) == 1:
        hours, minutes, seconds = parts[0], 0, 0
    else:
        return None
    return days * 24 + hours + minutes / 60 + seconds / 3600


def validate_aicr_slurm(cfg: DictConfig) -> None:
    """Reject Unity leftovers when the partition looks like AICR."""
    if not is_aicr_slurm(cfg):
        return
    slurm = cfg.slurm
    if not slurm.get("account"):
        raise ValueError("AICR jobs require slurm.account")
    for key in ("constraint", "qos", "gres"):
        value = slurm.get(key)
        if value not in (None, "", []):
            raise ValueError(f"AICR jobs must not set slurm.{key} (got {value!r})")
    partition = str(slurm.get("partition") or "")
    for token in ("gpu-preempt", "superpod"):
        if token in partition:
            raise ValueError(
                f"AICR jobs must not use Unity partitions (found {token!r} in {partition!r})"
            )
    if slurm.get("gpus") is None and slurm.get("gpus_per_node") is None:
        raise ValueError("AICR jobs require slurm.gpus or slurm.gpus_per_node")
    hours = _slurm_time_hours(slurm.get("time"))
    if hours is None:
        return
    if "devel" in partition and hours > 4:
        print(f"[Warning] AICR devel max is 4 h; slurm.time resolves to {hours:g} h")
    if "batch" in partition and hours > 24:
        print(f"[Warning] AICR batch max is 24 h; slurm.time resolves to {hours:g} h")
