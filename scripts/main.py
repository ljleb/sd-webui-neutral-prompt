import torch
from modules import scripts, processing, prompt_parser, script_callbacks, sd_samplers_kdiffusion, shared


is_enabled = False


def combine_denoise_hijack(self, x_out, conds_list, neg_cond, cond_scale):
    if not is_enabled:
        return original_combine_denoise(self, x_out, conds_list, neg_cond, cond_scale)

    denoised_uncond = x_out[-neg_cond.shape[0]:]
    denoised = torch.clone(denoised_uncond)

    for i, conds in enumerate(conds_list):
        origin_cond = x_out[conds[0][0]]
        for cond_index, weight in conds[1:]:
            perp = origin_cond + get_perpendicular_component(x_out[cond_index] - origin_cond, denoised_uncond[i] - origin_cond)
            denoised[i] += (x_out[cond_index] - perp) * (weight * cond_scale)

    return denoised


original_combine_denoise = getattr(sd_samplers_kdiffusion.CFGDenoiser, '__prep_neg_original_combine_denoise', sd_samplers_kdiffusion.CFGDenoiser.combine_denoised)
setattr(sd_samplers_kdiffusion.CFGDenoiser, '__prep_neg_original_combine_denoise', original_combine_denoise)
sd_samplers_kdiffusion.CFGDenoiser.combine_denoised = combine_denoise_hijack


def get_perpendicular_component(pos, neg):
    projected_neg = pos * torch.sum(neg * pos) / torch.norm(pos) ** 2
    return neg - projected_neg


def get_multicond_learned_conditioning_hijack(model, prompts, steps):
    if not is_enabled:
        return original_get_multicond_learned_conditioning(model, prompts, steps)

    res = original_get_multicond_learned_conditioning(model, prompts, steps)
    for l in res.batch:
        l.insert(0, prompt_parser.ComposableScheduledPromptConditioning(
            schedules=prompt_parser.get_learned_conditioning(model, [''], steps)[0],
            weight=0.
        ))
    return res


original_get_multicond_learned_conditioning = getattr(prompt_parser, '__prep_neg_original_get_multicond_learned_conditioning', prompt_parser.get_multicond_learned_conditioning)
setattr(prompt_parser, '__prep_neg_original_get_multicond_learned_conditioning', original_get_multicond_learned_conditioning)
prompt_parser.get_multicond_learned_conditioning = get_multicond_learned_conditioning_hijack


def on_script_unloaded():
    prompt_parser.get_multicond_learned_conditioning = original_get_multicond_learned_conditioning
    sd_samplers_kdiffusion.CFGDenoiser.combine_denoise = original_combine_denoise


script_callbacks.on_script_unloaded(on_script_unloaded)


def on_ui_settings():
    section = ('prep-neg', 'Prep-Neg')
    shared.opts.add_option('prep_neg_enabled', shared.OptionInfo(True, 'Enabled', section=section))


script_callbacks.on_ui_settings(on_ui_settings)


class PrepNegScript(scripts.Script):
    def title(self) -> str:
        return "Prep-Neg"

    def show(self, is_img2img: bool):
        return scripts.AlwaysVisible

    def process(self, p: processing.StableDiffusionProcessing, *args):
        global is_enabled
        is_enabled = shared.opts.data.get('prep_neg_enabled', True)
