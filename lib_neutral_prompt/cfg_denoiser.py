from lib_neutral_prompt import hijacker, global_state, prompt_parser
from modules import script_callbacks, sd_samplers_kdiffusion
import torch


cfg_denoiser_hijacker = hijacker.ModuleHijacker.install_or_get(
    module=sd_samplers_kdiffusion.CFGDenoiser,
    hijacker_attribute='__neutral_prompt',
    register_uninstall=script_callbacks.on_script_unloaded,
)


@cfg_denoiser_hijacker.hijack('combine_denoised')
def combine_denoise_hijack(self, x_out, conds_list, uncond, cond_scale, original_function):
    if not global_state.is_enabled or not global_state.perp_profile:
        return original_function(self, x_out, conds_list, uncond, cond_scale)

    x_uncond = x_out[-uncond.shape[0]:]
    denoised = torch.clone(x_uncond)

    for i, (conds, keywords) in enumerate(zip(conds_list, global_state.perp_profile)):
        keyword_cond_pairs = list(zip(keywords, conds))

        x_pos = torch.zeros_like(denoised[i])
        and_indices = [i for i, k in enumerate(keywords) if k == prompt_parser.AND_KEYWORD]
        for keyword, (cond_index, weight) in [keyword_cond_pairs[i] for i in and_indices]:
            x_pos += weight * x_out[cond_index]

        x_delta_acc = torch.zeros_like(denoised[i])
        and_perp_indices = [i for i, k in enumerate(keywords) if k == prompt_parser.AND_PERP_KEYWORD]
        for keyword, (cond_index, weight) in [keyword_cond_pairs[i] for i in and_perp_indices]:
            x_neutral = x_out[cond_index]
            x_pos_delta = x_pos - x_uncond[i]
            x_delta_acc -= weight * get_perpendicular_component(x_pos_delta, x_neutral - x_uncond[i])

        denoised[i] += cond_scale * (x_pos - x_uncond[i] - x_delta_acc)
        x_pos_std = torch.std(x_pos)
        x_cfg_std = torch.std(denoised[i])
        denoised[i] *= global_state.cfg_rescale * (x_pos_std / x_cfg_std - 1) + 1

    return denoised


def get_perpendicular_component(vector, neutral):
    assert vector.shape == neutral.shape
    return neutral * torch.sum(neutral * vector) / torch.norm(neutral) ** 2
