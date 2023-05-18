import torch
from modules import scripts, processing, prompt_parser, script_callbacks, sd_samplers_kdiffusion, shared
import gradio as gr


is_enabled = False
neutral_prompt = ''
origin_cond_scale = 1.0


def combine_denoise_hijack(self, x_out, conds_list, uncond, cond_scale):
    global neutral_prompt
    if not is_enabled:
        return original_combine_denoise(self, x_out, conds_list, uncond, cond_scale)

    denoised_uncond = x_out[-uncond.shape[0]:]
    denoised = torch.clone(denoised_uncond)

    for i, conds in enumerate(conds_list):
        origin_cond = x_out[conds[0][0]]
        for cond_index, weight in conds[1:]:
            aligned_negative = origin_cond + origin_cond_scale * get_perpendicular_component(x_out[cond_index] - origin_cond, denoised_uncond[i] - origin_cond)
            denoised[i] += (x_out[cond_index] - aligned_negative) * (weight * cond_scale)

    return denoised


original_combine_denoise = getattr(sd_samplers_kdiffusion.CFGDenoiser, '__neutral_prompt_original_combine_denoise', sd_samplers_kdiffusion.CFGDenoiser.combine_denoised)
setattr(sd_samplers_kdiffusion.CFGDenoiser, '__neutral_prompt_original_combine_denoise', original_combine_denoise)
sd_samplers_kdiffusion.CFGDenoiser.combine_denoised = combine_denoise_hijack


def get_perpendicular_component(pos, neg):
    projected_neg = pos * torch.sum(neg * pos) / torch.norm(pos) ** 2
    return neg - projected_neg


def get_multicond_learned_conditioning_hijack(model, prompts, steps):
    global is_enabled, neutral_prompt
    if not is_enabled:
        return original_get_multicond_learned_conditioning(model, prompts, steps)

    res = original_get_multicond_learned_conditioning(model, prompts, steps)
    for l in res.batch:
        l.insert(0, prompt_parser.ComposableScheduledPromptConditioning(
            schedules=prompt_parser.get_learned_conditioning(model, [neutral_prompt], steps)[0],
            weight=0.
        ))
    return res


original_get_multicond_learned_conditioning = getattr(prompt_parser, '__neutral_prompt_original_get_multicond_learned_conditioning', prompt_parser.get_multicond_learned_conditioning)
setattr(prompt_parser, '__neutral_prompt_original_get_multicond_learned_conditioning', original_get_multicond_learned_conditioning)
prompt_parser.get_multicond_learned_conditioning = get_multicond_learned_conditioning_hijack


def on_script_unloaded():
    prompt_parser.get_multicond_learned_conditioning = original_get_multicond_learned_conditioning
    sd_samplers_kdiffusion.CFGDenoiser.combine_denoise = original_combine_denoise


script_callbacks.on_script_unloaded(on_script_unloaded)


def on_ui_settings():
    section = ('neutral-prompt', 'Neutral Prompt')
    shared.opts.add_option('neutral_prompt_enabled', shared.OptionInfo(True, 'Enabled', section=section))


script_callbacks.on_ui_settings(on_ui_settings)


class NeutralPromptScript(scripts.Script):
    def title(self) -> str:
        return "Neutral Prompt"

    def show(self, is_img2img: bool):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(label='Neutral Prompt', open=False):
            ui_neutral_prompt = gr.Textbox(placeholder='Neutral prompt')

        return [ui_neutral_prompt]

    def process(self, p: processing.StableDiffusionProcessing, ui_neutral_prompt):
        global is_enabled, neutral_prompt
        is_enabled = shared.opts.data.get('neutral_prompt_enabled', True)
        neutral_prompt = ui_neutral_prompt
