from lib_neutral_prompt import global_state


def override_cfg_rescale(cfg_rescale: float):
    global_state.cfg_rescale_override = cfg_rescale
