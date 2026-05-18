
import sys
from typing import List

import omegaconf


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