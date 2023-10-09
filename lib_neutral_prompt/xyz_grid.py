import sys
from types import ModuleType
from typing import Optional
from modules import scripts
from lib_neutral_prompt import global_state


def patch():
    xyz_module = find_xyz_module()
    if xyz_module is None:
        print("[sd-webui-neutral-prompt]", "xyz_grid.py not found.", file=sys.stderr)
        return

    xyz_module.axis_options.extend([
        xyz_module.AxisOption("[Neutral Prompt] CFG Rescale", int_or_float, apply_cfg_rescale()),
    ])


class XyzFloat(float):
    is_xyz: bool = True


def apply_cfg_rescale():
    def callback(_p, v, _vs):
        global_state.cfg_rescale = XyzFloat(v)

    return callback


def int_or_float(string):
    try:
        return int(string)
    except ValueError:
        return float(string)


def find_xyz_module() -> Optional[ModuleType]:
    for data in scripts.scripts_data:
        if data.script_class.__module__ in {"xyz_grid.py", "xy_grid.py"} and hasattr(data, "module"):
            return data.module

    return None
