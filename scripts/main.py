from typing import Dict

from lib_neutral_prompt import global_state, hijacker, prompt_parser, cfg_denoiser, ui
import importlib
importlib.reload(global_state)
importlib.reload(hijacker)
importlib.reload(prompt_parser)
importlib.reload(cfg_denoiser)
importlib.reload(ui)
from modules import scripts, processing, shared


class NeutralPromptScript(scripts.Script):
    def __init__(self):
        self.accordion_interface = ui.AccordionInterface()

    def title(self) -> str:
        return "Neutral Prompt"

    def show(self, is_img2img: bool):
        return scripts.AlwaysVisible

    def ui(self, is_img2img: bool):
        self.accordion_interface.arrange_components(is_img2img)
        self.accordion_interface.connect_events(is_img2img)
        self.infotext_fields = self.accordion_interface.get_infotext_fields()
        return self.accordion_interface.get_components()

    def process(self, p: processing.StableDiffusionProcessing, *args):
        args = self.accordion_interface.unpack_processing_args(*args)

        self.update_global_state(args)
        p.extra_generation_params.update(self.accordion_interface.get_extra_generation_params(args))

    def update_global_state(self, args: Dict):
        if shared.state.job_no > 0:
            return

        global_state.is_enabled = shared.opts.data.get('neutral_prompt_enabled', True)
        for k, v in args.items():
            try:
                getattr(global_state, k)
            except AttributeError:
                continue

            setattr(global_state, k, v)
