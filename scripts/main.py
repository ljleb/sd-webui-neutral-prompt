import torch

from modules import scripts, processing, script_callbacks, sd_samplers_kdiffusion, shared


is_enabled = False


def combine_denoise_hijack(self, x_out, conds_list, neg_cond, cond_scale, origin_cond=None):
    if not is_enabled or origin_cond is None:
        extra_args = [origin_cond] * (0 if origin_cond is None else 1)
        return original_combine_denoise(self, x_out, conds_list, neg_cond, cond_scale, *extra_args)

    denoised_uncond = x_out[-neg_cond.shape[0]:]
    denoised = torch.clone(denoised_uncond)

    for i, conds in enumerate(conds_list):
        for cond_index, weight in conds:
            perp = origin_cond + get_perpendicular_component(x_out[cond_index] - origin_cond, denoised_uncond[i] - origin_cond)
            denoised[i] += (x_out[cond_index] - perp) * (weight * cond_scale)

    return denoised


def get_perpendicular_component(pos, neg):
    projected_neg = pos * torch.sum(neg * pos) / torch.norm(pos) ** 2
    return neg - projected_neg


original_combine_denoise = getattr(sd_samplers_kdiffusion.CFGDenoiser, '__prep_neg_original_combine_denoise', sd_samplers_kdiffusion.CFGDenoiser.combine_denoised)
setattr(sd_samplers_kdiffusion.CFGDenoiser, '__prep_neg_original_combine_denoise', original_combine_denoise)
sd_samplers_kdiffusion.CFGDenoiser.combine_denoised = combine_denoise_hijack


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
