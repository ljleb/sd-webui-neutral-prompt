from lib_neutral_prompt import hijacker, global_state, prompt_parser, perp_parser
from modules import script_callbacks, sd_samplers, shared
from typing import Tuple, List
import functools
import torch
import sys
import textwrap


def combine_denoised_hijack(
    x_out: torch.Tensor,
    batch_cond_indices: List[List[Tuple[int, float]]],
    text_uncond: torch.Tensor,
    cond_scale: float,
    original_function,
) -> torch.Tensor:
    if not global_state.is_enabled:
        return original_function(x_out, batch_cond_indices, text_uncond, cond_scale)

    denoised = get_webui_denoised(x_out, batch_cond_indices, text_uncond, cond_scale, original_function)
    uncond = x_out[-text_uncond.shape[0]:]

    for batch_i, (prompt, cond_indices) in enumerate(zip(global_state.prompt_exprs, batch_cond_indices)):
        cond_delta = prompt.get_cond_delta(x_out, uncond[batch_i], cond_indices, 0)
        perp_cond_delta = prompt.get_perp_cond_delta(x_out, cond_delta, uncond[batch_i], cond_indices, 0)
        cfg_cond = denoised[batch_i] + perp_cond_delta * cond_scale
        denoised[batch_i] = cfg_cond * get_cfg_rescale_factor(cfg_cond, uncond[batch_i] + cond_delta + perp_cond_delta)

    return denoised


def get_webui_denoised(
    x_out: torch.Tensor,
    batch_cond_indices: List[List[Tuple[int, float]]],
    text_uncond: torch.Tensor,
    cond_scale: float,
    original_function,
):
    sliced_x_out = []
    sliced_batch_cond_indices = []

    for batch_i, (prompt, cond_indices) in enumerate(zip(global_state.prompt_exprs, batch_cond_indices)):
        sliced_batch_cond_indices.append([])
        for prompt_child, (cond_index, weight) in zip(prompt.children, cond_indices):
            if not isinstance(prompt_child, perp_parser.ComposablePrompt):
                continue

            sliced_x_out.append(x_out[cond_index])
            sliced_batch_cond_indices[-1].append((len(sliced_x_out) - 1, weight))

    sliced_x_out += [unc for unc in x_out[-text_uncond.shape[0]:]]
    sliced_x_out = torch.stack(sliced_x_out, dim=0)
    sliced_batch_cond_indices = [il for il in sliced_batch_cond_indices if il]
    return original_function(sliced_x_out, sliced_batch_cond_indices, text_uncond, cond_scale)


def get_cfg_rescale_factor(cfg_cond, cond):
    return global_state.cfg_rescale * (torch.std(cond) / torch.std(cfg_cond) - 1) + 1


sd_samplers_hijacker = hijacker.ModuleHijacker.install_or_get(
    module=sd_samplers,
    hijacker_attribute='__neutral_prompt_hijacker',
    on_uninstall=script_callbacks.on_script_unloaded,
)


@sd_samplers_hijacker.hijack('create_sampler')
def create_sampler_hijack(name: str, model, original_function):
    sampler = original_function(name, model)
    if name.startswith(('DDIM', 'PLMS', 'UniPC')):
        if global_state.is_enabled:
            warn_unsupported_sampler()

        return sampler

    sampler.model_wrap_cfg.combine_denoised = functools.partial(
        combine_denoised_hijack,
        original_function=sampler.model_wrap_cfg.combine_denoised
    )
    return sampler


def warn_unsupported_sampler():
    console_warn('''
        Neutral prompt relies on composition via AND, which the webui does not support when using any of the DDIM, PLMS and UniPC samplers
        The sampler will NOT be patched
        Falling back on original sampler implementation...
    ''')


def warn_projection_not_found():
    console_warn('''
        Could not find a projection for one or more AND_PERP prompts
        These prompts will NOT be made perpendicular
    ''')


def console_warn(message):
    if not global_state.verbose:
        return

    print(f'\n[sd-webui-neutral-prompt extension]{textwrap.dedent(message)}', file=sys.stderr)
