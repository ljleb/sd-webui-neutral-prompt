from typing import List

import torch
from modules import scripts, processing, prompt_parser, script_callbacks, sd_samplers_kdiffusion
from lib_neutral_prompt import hijacker
import gradio as gr
import re


prompt_parser_hijacker = hijacker.ModuleHijacker.install_or_get(
    module=prompt_parser,
    hijacker_attribute='__neutral_prompt',
    register_uninstall=script_callbacks.on_script_unloaded,
)
cfg_denoiser_hijacker = hijacker.ModuleHijacker.install_or_get(
    module=sd_samplers_kdiffusion.CFGDenoiser,
    hijacker_attribute='__neutral_prompt',
    register_uninstall=script_callbacks.on_script_unloaded,
)


AND_KEYWORD = 'AND'
AND_PERP_KEYWORD = 'AND_PERP'


is_enabled = False
perp_profile = []
cfg_rescale = 0


@cfg_denoiser_hijacker.hijack('combine_denoised')
def combine_denoise_hijack(self, x_out, conds_list, uncond, cond_scale, original_function):
    global is_enabled, cfg_rescale, perp_profile
    if not is_enabled or not perp_profile:
        return original_function(self, x_out, conds_list, uncond, cond_scale)

    x_uncond = x_out[-uncond.shape[0]:]
    denoised = torch.clone(x_uncond)

    for i, (conds, keywords) in enumerate(zip(conds_list, perp_profile)):
        keyword_cond_pairs = list(zip(keywords, conds))

        x_pos = torch.zeros_like(denoised[i])
        and_indices = [i for i, k in enumerate(keywords) if k == AND_KEYWORD]
        for keyword, (cond_index, weight) in [keyword_cond_pairs[i] for i in and_indices]:
            x_pos += weight * x_out[cond_index]

        x_delta_acc = torch.zeros_like(denoised[i])
        and_perp_indices = [i for i, k in enumerate(keywords) if k == AND_PERP_KEYWORD]
        for keyword, (cond_index, weight) in [keyword_cond_pairs[i] for i in and_perp_indices]:
            x_neutral = x_out[cond_index]
            x_pos_delta = x_pos - x_uncond[i]
            x_delta_acc -= weight * get_perpendicular_component(x_pos_delta, x_neutral - x_uncond[i])

        denoised[i] += cond_scale * (x_pos - x_uncond[i] - x_delta_acc)
        x_pos_std = torch.std(x_pos)
        x_cfg_std = torch.std(denoised[i])
        denoised[i] *= cfg_rescale * (x_pos_std / x_cfg_std - 1) + 1

    return denoised


def get_perpendicular_component(vector, neutral):
    assert vector.shape == neutral.shape
    return neutral * torch.sum(neutral * vector) / torch.norm(neutral) ** 2


and_perp_regex = re.compile(fr'\b({AND_KEYWORD}|{AND_PERP_KEYWORD})\b')


@prompt_parser_hijacker.hijack('get_multicond_learned_conditioning')
def get_multicond_learned_conditioning_hijack(model, prompts, steps, original_function):
    global is_enabled, perp_profile
    if not is_enabled:
        return original_function(model, prompts, steps)

    perp_profile.clear()
    for prompt in prompts:
        and_keywords = and_perp_regex.split(prompt)[1::2]
        perp_profile.append([AND_KEYWORD] + and_keywords)

    prompts = [and_perp_regex.sub(AND_KEYWORD, prompt) for prompt in prompts]
    prompts = [prompt.replace('\n', ' ') for prompt in prompts]
    return original_function(model, prompts, steps)


class NeutralPromptScript(scripts.Script):
    def __init__(self):
        self.txt2img_prompt_textbox = None
        self.img2img_prompt_textbox = None

    def title(self) -> str:
        return "Neutral Prompt"

    def show(self, is_img2img: bool):
        return scripts.AlwaysVisible

    def after_component(self, component, **kwargs):
        if getattr(component, 'elem_id', None) == 'txt2img_prompt':
            self.txt2img_prompt_textbox = component

        if getattr(component, 'elem_id', None) == 'img2img_prompt':
            self.img2img_prompt_textbox = component

    def ui(self, is_img2img: bool) -> List[gr.components.Component]:
        prompt_textbox = self.img2img_prompt_textbox if is_img2img else self.txt2img_prompt_textbox

        with gr.Accordion(label='Neutral Prompt', open=False):
            ui_enabled = gr.Checkbox(label='Enable', value=False)
            ui_cfg_rescale = gr.Slider(label='CFG rescale', minimum=0, maximum=1, value=0)
            with gr.Accordion(label='Prompt formatter', open=False):
                neutral_prompt = gr.Textbox(label='Neutral prompt', show_label=False, lines=3, placeholder='Neutral prompt (click on apply below to append this to the positive prompt textbox)')
                neutral_cond_scale = gr.Slider(label='Neutral CFG', minimum=-3, maximum=3, value=-1)
                append_to_prompt = gr.Button(value='Apply to prompt')

            append_to_prompt.click(
                fn=lambda init_prompt, prompt, scale: (f'{init_prompt} {AND_PERP_KEYWORD} {prompt} :{scale}', ''),
                inputs=[prompt_textbox, neutral_prompt, neutral_cond_scale],
                outputs=[prompt_textbox, neutral_prompt]
            )

        return [ui_enabled, ui_cfg_rescale]

    def process(self, p: processing.StableDiffusionProcessing, ui_enabled, ui_cfg_rescale):
        global is_enabled, cfg_rescale
        is_enabled = ui_enabled
        cfg_rescale = ui_cfg_rescale
