from lib_neutral_prompt import global_state, prompt_parser
import torch


def combine_denoised_hijack(x_out, batch_cond_indices, noisy_uncond, cond_scale, original_function):
    original_batch_cond_indices = [
        [
            cond_index
            for k, cond_index in zip(combination_keywords, cond_indices)
            if k == prompt_parser.AND_KEYWORD
        ]
        for combination_keywords, cond_indices in zip(global_state.perp_profile, batch_cond_indices)
    ]
    denoised = original_function(x_out, original_batch_cond_indices, noisy_uncond, cond_scale)

    if not global_state.is_enabled or not global_state.perp_profile:
        return denoised

    uncond = x_out[-noisy_uncond.shape[0]:]

    for batch_i, (combination_keywords, cond_indices) in enumerate(zip(global_state.perp_profile, batch_cond_indices)):
        def get_cond_indices(filter_k):
            return [cond_indices[i] for i, k in enumerate(combination_keywords) if k == filter_k]

        cond_delta = combine_cond_deltas(x_out, uncond[batch_i], get_cond_indices(prompt_parser.AND_KEYWORD))
        perp_cond_delta = combine_perp_cond_deltas(x_out, cond_delta, uncond[batch_i], get_cond_indices(prompt_parser.AND_PERP_KEYWORD))

        cfg_cond = denoised[batch_i] - perp_cond_delta * cond_scale
        denoised[batch_i] = cfg_cond * get_cfg_rescale_factor(cfg_cond, uncond[batch_i] + cond_delta - perp_cond_delta)

    return denoised


def combine_cond_deltas(x_out, uncond, cond_indices):
    cond_delta = torch.zeros_like(x_out[0])
    for cond_index, weight in cond_indices:
        cond_delta += weight * (x_out[cond_index] - uncond)

    return cond_delta


def combine_perp_cond_deltas(x_out, cond_delta, uncond, cond_indices):
    perp_cond_delta = torch.zeros_like(x_out[0])
    for cond_index, weight in cond_indices:
        perp_cond = x_out[cond_index]
        perp_cond_delta -= weight * get_perpendicular_component(cond_delta, perp_cond - uncond)

    return perp_cond_delta


def get_perpendicular_component(normal, vector):
    assert vector.shape == normal.shape
    return vector - normal * torch.sum(normal * vector) / torch.norm(normal) ** 2


def get_cfg_rescale_factor(denoised, positive_epsilon):
    x_pos_std = torch.std(positive_epsilon)
    x_cfg_std = torch.std(denoised)
    return global_state.cfg_rescale * (x_pos_std / x_cfg_std - 1) + 1
