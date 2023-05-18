import torch
from modules import scripts, processing, prompt_parser, script_callbacks, sd_samplers_kdiffusion, shared
import gradio as gr
import re


is_enabled = False
perp_profile = None
cfg_rescale = 0


def combine_denoise_hijack(self, x_out, conds_list, uncond, cond_scale):
    global is_enabled, cfg_rescale
    if not is_enabled:
        return original_combine_denoise(self, x_out, conds_list, uncond, cond_scale)

    x_uncond = x_out[-uncond.shape[0]:]
    denoised = torch.clone(x_uncond)

    for i, (conds, keywords) in enumerate(zip(conds_list, perp_profile)):
        x_pos = torch.zeros_like(denoised[i])
        for keyword, (cond_index, weight) in [k for k in zip(keywords, conds) if k[0] == 'AND']:
            x_pos += weight * x_out[cond_index]

        x_delta_acc = torch.zeros_like(denoised[i])
        for keyword, (cond_index, weight) in [k for k in zip(keywords, conds) if k[0] == 'AND_PERP']:
            x_neutral = x_out[cond_index]
            x_pos_delta = x_pos - x_uncond[i]
            x_delta_acc += weight * get_perpendicular_component(x_pos_delta, x_neutral - x_uncond[i])

        denoised[i] += cond_scale * (x_pos - x_uncond[i] - x_delta_acc)
        x_pos_std = torch.std(x_pos)
        x_cfg_std = torch.std(denoised[i])
        denoised[i] *= cfg_rescale * (x_pos_std / x_cfg_std - 1) + 1

    return denoised


original_combine_denoise = getattr(sd_samplers_kdiffusion.CFGDenoiser, '__neutral_prompt_original_combine_denoise', sd_samplers_kdiffusion.CFGDenoiser.combine_denoised)
setattr(sd_samplers_kdiffusion.CFGDenoiser, '__neutral_prompt_original_combine_denoise', original_combine_denoise)
sd_samplers_kdiffusion.CFGDenoiser.combine_denoised = combine_denoise_hijack


def get_perpendicular_component(pos, neg):
    projected_neg = pos * torch.sum(neg * pos) / torch.norm(pos) ** 2
    return neg - projected_neg


def get_multicond_learned_conditioning_hijack(model, prompts, steps):
    global is_enabled, perp_profile
    if not is_enabled:
        return original_get_multicond_learned_conditioning(model, prompts, steps)

    perp_profile = []
    for prompt in prompts:
        and_keywords = re.split(r'\b(AND(?:_PERP)?)\b', prompt)[1::2]
        perp_profile.append(['AND'] + and_keywords)

    new_prompts = [re.sub(r'\b(AND(?:_PERP)?)\b', 'AND', prompt).replace('\n', ' ') for prompt in prompts]
    return original_get_multicond_learned_conditioning(model, new_prompts, steps)


original_get_multicond_learned_conditioning = getattr(prompt_parser, '__neutral_prompt_original_get_multicond_learned_conditioning', prompt_parser.get_multicond_learned_conditioning)
setattr(prompt_parser, '__neutral_prompt_original_get_multicond_learned_conditioning', original_get_multicond_learned_conditioning)
prompt_parser.get_multicond_learned_conditioning = get_multicond_learned_conditioning_hijack


def on_script_unloaded():
    prompt_parser.get_multicond_learned_conditioning = original_get_multicond_learned_conditioning
    sd_samplers_kdiffusion.CFGDenoiser.combine_denoise = original_combine_denoise


script_callbacks.on_script_unloaded(on_script_unloaded)


class NeutralPromptScript(scripts.Script):
    def title(self) -> str:
        return "Neutral Prompt"

    def show(self, is_img2img: bool):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(label='Neutral Prompt', open=False):
            ui_enabled = gr.Checkbox(label='Enable', value=False)
            ui_neutral_prompt = gr.Textbox(label='Neutral prompt ', show_label=False, lines=3, placeholder='Neutral prompt')
            ui_neutral_cond_scale = gr.Slider(label='Neutral CFG ', minimum=-3, maximum=0, value=-1)
            ui_cfg_rescale = gr.Slider(label='CFG Rescale ', minimum=0, maximum=1, value=0)

        return [ui_enabled, ui_neutral_prompt, ui_neutral_cond_scale, ui_cfg_rescale]

    def process(self, p: processing.StableDiffusionProcessing, ui_enabled, ui_neutral_prompt, ui_neutral_cond_scale, ui_cfg_rescale):
        global is_enabled, neutral_prompt, neutral_cond_scale, cfg_rescale
        is_enabled = ui_enabled
        neutral_prompt = ui_neutral_prompt
        neutral_cond_scale = ui_neutral_cond_scale
        cfg_rescale = ui_cfg_rescale
