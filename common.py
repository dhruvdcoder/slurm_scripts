
import sys
from typing import List


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

def remove_dms(cfg) -> List[str]:
    dms_to_remove = cfg.eval.get("dms_to_remove", [])
    remove_strs = []
    for dm in dms_to_remove:
        remove_strs.append(f"~datamodule.dataset_managers.{dm}")
    return remove_strs