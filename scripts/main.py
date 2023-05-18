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
        self.ui = ui.GradioUserInterface()

    def title(self) -> str:
        return "Neutral Prompt"

    def show(self, is_img2img: bool):
        return scripts.AlwaysVisible

    def ui(self, is_img2img: bool):
        self.ui.arrange_components()
        self.ui.connect_events()
        return self.ui.get_processing_components()

    def process(self, p: processing.StableDiffusionProcessing, *args):
        self.ui.update_global_state(*args)
