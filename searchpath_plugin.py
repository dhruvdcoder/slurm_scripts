from pathlib import Path
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin
from hydra.core.config_search_path import ConfigSearchPath

import importlib.util
import xlm

# locate xlm-core
def locate_xlm_core() -> Path:
    package_spec = importlib.util.find_spec("xlm")
    location = package_spec.submodule_search_locations
    if location:
        location = location[0]
    else:
        raise ValueError("xlm-core not found")
    return Path(location)

class HydraCommonSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        xlm_core_path = locate_xlm_core()
        search_path.append(
            "file", str(xlm_core_path / "configs/common")
        )


Plugins.instance().register(HydraCommonSearchPathPlugin)