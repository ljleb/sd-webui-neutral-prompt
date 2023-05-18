from lib_neutral_prompt import hijacker, global_state, prompt_parser
from modules import script_callbacks, sd_samplers_kdiffusion
import torch


cfg_denoiser_hijacker = hijacker.ModuleHijacker.install_or_get(
    module=sd_samplers_kdiffusion.CFGDenoiser,
    hijacker_attribute='__neutral_prompt',
    register_uninstall=script_callbacks.on_script_unloaded,
)


@cfg_denoiser_hijacker.hijack('combine_denoised')
def combine_denoised_hijack(self, x_out, conds_list, uncond, cond_scale, original_function):
    if not global_state.is_enabled or not global_state.perp_profile:
        return original_function(self, x_out, conds_list, uncond, cond_scale)

    uncond = x_out[-uncond.shape[0]:]
    denoised = torch.clone(uncond)

    for batch_i, (conds, keywords) in enumerate(zip(conds_list, global_state.perp_profile)):
        keyword_cond_pairs = list(zip(keywords, conds))

        positive_epsilon = combine_epsilons(x_out, keywords, keyword_cond_pairs)
        perp_epsilon_delta = combine_epsilon_deltas(x_out, positive_epsilon, uncond[batch_i], keywords, keyword_cond_pairs)

        denoised[batch_i] += cond_scale * (positive_epsilon - uncond[batch_i] - perp_epsilon_delta)
        denoised[batch_i] *= get_cfg_rescale_factor(denoised[batch_i], positive_epsilon)

    return denoised


def combine_epsilons(x_out, keywords, keyword_cond_pairs):
    x_pos = torch.zeros_like(x_out[0])
    indices = [i for i, k in enumerate(keywords) if k == prompt_parser.AND_KEYWORD]
    for keyword, (cond_index, weight) in [keyword_cond_pairs[i] for i in indices]:
        x_pos += weight * x_out[cond_index]

    return x_pos


def combine_epsilon_deltas(x_out, cond, uncond, keywords, keyword_cond_pairs):
    epsilon_delta = torch.zeros_like(x_out[0])
    perp_indices = [i for i, k in enumerate(keywords) if k == prompt_parser.AND_PERP_KEYWORD]
    for keyword, (cond_index, weight) in [keyword_cond_pairs[i] for i in perp_indices]:
        neutral_cond = x_out[cond_index]
        epsilon_delta -= weight * get_perpendicular_component(cond - uncond, neutral_cond - uncond)

    return epsilon_delta


def get_perpendicular_component(normal, vector):
    assert vector.shape == normal.shape
    return vector - normal * torch.sum(normal * vector) / torch.norm(normal) ** 2


def get_cfg_rescale_factor(denoised, positive_epsilon):
    x_pos_std = torch.std(positive_epsilon)
    x_cfg_std = torch.std(denoised)
    return global_state.cfg_rescale * (x_pos_std / x_cfg_std - 1) + 1
