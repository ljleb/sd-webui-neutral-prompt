from lib_neutral_prompt import global_state, hijacker, neutral_prompt_parser, prompt_parser_hijack, cfg_denoiser_hijack, ui
from modules import scripts, processing, shared
from typing import Dict
import functools


class NeutralPromptScript(scripts.Script):
    def __init__(self):
        self.accordion_interface = ui.AccordionInterface()

    def title(self) -> str:
        return "Neutral Prompt"

    def show(self, is_img2img: bool):
        return scripts.AlwaysVisible

    def ui(self, is_img2img: bool):
        self.hijack_composable_lora(is_img2img)

        self.accordion_interface.arrange_components(is_img2img)
        self.accordion_interface.connect_events(is_img2img)
        self.infotext_fields = self.accordion_interface.get_infotext_fields()
        self.paste_field_names = self.accordion_interface.get_paste_field_names()
        self.accordion_interface.set_rendered()
        return self.accordion_interface.get_components()

    def process(self, p: processing.StableDiffusionProcessing, *args):
        args = self.accordion_interface.unpack_processing_args(*args)

        self.update_global_state(args)
        if global_state.is_enabled:
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

    def hijack_composable_lora(self, is_img2img):
        if self.accordion_interface.is_rendered:
            return

        lora_script = None
        script_runner = scripts.scripts_img2img if is_img2img else scripts.scripts_txt2img

        for script in script_runner.alwayson_scripts:
            if script.title().lower() == "composable lora":
                lora_script = script
                break

        if lora_script is not None:
            lora_script.process = functools.partial(composable_lora_process_hijack, original_function=lora_script.process)


def composable_lora_process_hijack(p: processing.StableDiffusionProcessing, *args, original_function, **kwargs):
    if not global_state.is_enabled:
        return original_function(p, *args, **kwargs)

    exprs = prompt_parser_hijack.parse_prompts(p.all_prompts)
    all_prompts, p.all_prompts = p.all_prompts, prompt_parser_hijack.transpile_exprs(exprs)
    original_function(p, *args, **kwargs)
    # restore original prompts
    p.all_prompts = all_prompts
