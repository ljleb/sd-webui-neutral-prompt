from lib_neutral_prompt import global_state, hijacker, prompt_parser, cfg_denoiser, ui
import importlib
importlib.reload(global_state)
importlib.reload(hijacker)
importlib.reload(prompt_parser)
importlib.reload(cfg_denoiser)
importlib.reload(ui)
from modules import scripts, processing


class NeutralPromptScript(scripts.Script):
    def __init__(self):
        self.gui = ui.GradioUserInterface()

    def title(self) -> str:
        return "Neutral Prompt"

    def show(self, is_img2img: bool):
        return scripts.AlwaysVisible

    def ui(self, is_img2img: bool):
        self.gui.arrange_components(is_img2img)
        self.gui.connect_events(is_img2img)
        return self.gui.get_components()

    def process(self, p: processing.StableDiffusionProcessing, *args):
        for k, v in self.gui.unpack_processing_args(*args).items():
            previous_value = getattr(global_state, k, None)
            if previous_value is None:
                continue

            setattr(global_state, k, v)
