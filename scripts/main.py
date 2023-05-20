import functools

from lib_neutral_prompt import global_state, hijacker, prompt_parser, cfg_denoiser, ui
import importlib
importlib.reload(global_state)
importlib.reload(hijacker)
importlib.reload(prompt_parser)
importlib.reload(cfg_denoiser)
importlib.reload(ui)
from modules import scripts, processing, sd_samplers, sd_samplers_common


class NeutralPromptScript(scripts.Script):
    def __init__(self):
        self.gui = ui.GradioUserInterface()
        self.p_hijacker = None

    def title(self) -> str:
        return "Neutral Prompt"

    def show(self, is_img2img: bool):
        return scripts.AlwaysVisible

    def ui(self, is_img2img: bool):
        self.gui.arrange_components(is_img2img)
        self.gui.connect_events(is_img2img)
        return self.gui.get_processing_components()

    def process(self, p: processing.StableDiffusionProcessing, *args):
        self.gui.on_process(*args)
        if not global_state.is_enabled:
            return

        self.hijack_sampler(p)

    def hijack_sampler(self, p: processing.StableDiffusionProcessing):
        sampler = sd_samplers.all_samplers_map[p.sampler_name]
        def hijacked_sampler_constructor(model):
            constructed = sampler.constructor(model)
            cfg_denoiser_hijacker = hijacker.ModuleHijacker.install_or_get(module=constructed.model_wrap_cfg, hijacker_attribute='__neutral_prompt_hijacker')
            cfg_denoiser_hijacker.hijack('combine_denoised')(cfg_denoiser.combine_denoised_hijack)
            return constructed

        hijacked_sampler_name = f'{p.sampler_name}_neutral_prompt'
        self.p_hijacker = hijacker.ModuleHijacker.install_or_get(module=p, hijacker_attribute='__neutral_prompt_hijacker')
        self.p_hijacker.hijack_attribute('sampler_name')(hijacked_sampler_name)
        sd_samplers.all_samplers_map[self.p_hijacker.get_backup_attribute('sampler_name')] = sd_samplers_common.SamplerData(
            hijacked_sampler_name,
            hijacked_sampler_constructor,
            sampler.aliases,
            sampler.options,
        )

    def postprocess(self, p, processed, *args):
        del sd_samplers.all_samplers_map[self.p_hijacker.get_backup_attribute('sampler_name')]
        self.p_hijacker.reset_module()
