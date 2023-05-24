from lib_neutral_prompt import hijacker, global_state, neutral_prompt_parser
from modules import script_callbacks, sd_samplers, shared
from typing import Tuple, List
import dataclasses
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
        args = CombineDenoiseArgs(x_out, uncond[batch_i], cond_indices)
        cond_delta = prompt.accept(IntermediateCondDeltaVisitor(), args, 0)
        perp_cond_delta = prompt.accept(PerpCondDeltaVisitor(), args, cond_delta, 0)
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
    uncond = x_out[-text_uncond.shape[0]:]
    sliced_batch_x_out = []
    sliced_batch_cond_indices = []

    for batch_i, (prompt, cond_indices) in enumerate(zip(global_state.prompt_exprs, batch_cond_indices)):
        args = CombineDenoiseArgs(x_out, uncond[batch_i], cond_indices)
        sliced_x_out, sliced_cond_indices = prompt.accept(GatherWebuiCondsVisitor(), args, len(sliced_batch_x_out))
        sliced_batch_cond_indices.append(sliced_cond_indices)
        sliced_batch_x_out.extend(sliced_x_out)

    sliced_batch_x_out += list(uncond)
    sliced_batch_x_out = torch.stack(sliced_batch_x_out, dim=0)
    sliced_batch_cond_indices = [il for il in sliced_batch_cond_indices if il]
    return original_function(sliced_batch_x_out, sliced_batch_cond_indices, text_uncond, cond_scale)


def get_cfg_rescale_factor(cfg_cond, cond):
    return global_state.cfg_rescale * (torch.std(cond) / torch.std(cfg_cond) - 1) + 1


@dataclasses.dataclass
class CombineDenoiseArgs:
    x_out: torch.Tensor
    uncond: torch.Tensor
    cond_indices: List[Tuple[int, float]]


@dataclasses.dataclass
class GatherWebuiCondsVisitor:
    def visit_composable_prompt(self, *args, **kwargs) -> Tuple[List[torch.Tensor], List[Tuple[int, float]]]:
        return [], []

    def visit_composite_prompt(self, that: neutral_prompt_parser.CompositePrompt, args: CombineDenoiseArgs, index_offset: int) -> Tuple[List[torch.Tensor], List[Tuple[int, float]]]:
        sliced_x_out = []
        sliced_cond_indices = []

        index_in = 0
        for child in that.children:
            index_out = index_offset + len(sliced_x_out)
            child_x_out, child_cond_indices = child.accept(GatherWebuiCondsVisitor.CondIndexVisitor(), args.x_out, args.cond_indices[index_in], index_out)
            sliced_x_out.extend(child_x_out)
            sliced_cond_indices.extend(child_cond_indices)
            index_in += child.accept(neutral_prompt_parser.FlatSizeVisitor())

        return sliced_x_out, sliced_cond_indices

    @dataclasses.dataclass
    class CondIndexVisitor:
        def visit_composable_prompt(self, that: neutral_prompt_parser.ComposablePrompt, x_out: torch.Tensor, cond_info: Tuple[int, float], index: int) -> Tuple[List[torch.Tensor], List[Tuple[int, float]]]:
            return [x_out[cond_info[0]]], [(index, cond_info[1])]

        def visit_composite_prompt(self, *args, **kwargs) -> Tuple[List[torch.Tensor], List[Tuple[int, float]]]:
            return [], []


@dataclasses.dataclass
class IntermediateCondDeltaVisitor:
    def visit_composable_prompt(self, that: neutral_prompt_parser.ComposablePrompt, args: CombineDenoiseArgs, index: int) -> torch.Tensor:
        return torch.zeros_like(args.x_out[0])

    def visit_composite_prompt(self, that: neutral_prompt_parser.CompositePrompt, args: CombineDenoiseArgs, index: int) -> torch.Tensor:
        cond_delta = torch.zeros_like(args.x_out[0])

        for child in that.children:
            cond_delta += child.accept(ImmediateCondDeltaVisitor(), args, index)
            index += child.accept(neutral_prompt_parser.FlatSizeVisitor())

        return cond_delta


@dataclasses.dataclass
class ImmediateCondDeltaVisitor:
    def visit_composable_prompt(self, that: neutral_prompt_parser.ComposablePrompt, args: CombineDenoiseArgs, index: int) -> torch.Tensor:
        cond_info = args.cond_indices[index]
        if that.weight != cond_info[1]:
            console_warn(f'''
                An unexpected noise weight was encountered at prompt #{index}. Expected :{that.weight}, but got :{cond_info[1]}
            ''')

        return cond_info[1] * (args.x_out[cond_info[0]] - args.uncond)

    def visit_composite_prompt(self, that: neutral_prompt_parser.CompositePrompt, args: CombineDenoiseArgs, index: int) -> torch.Tensor:
        return torch.zeros_like(args.x_out[0])


@dataclasses.dataclass
class PerpCondDeltaVisitor:
    def visit_composable_prompt(self, that: neutral_prompt_parser.ComposablePrompt, args: CombineDenoiseArgs, cond_delta: torch.Tensor, index: int) -> torch.Tensor:
        return torch.zeros_like(args.x_out[0])

    def visit_composite_prompt(self, that: neutral_prompt_parser.CompositePrompt, args: CombineDenoiseArgs, cond_delta: torch.Tensor, index: int) -> torch.Tensor:
        perp_cond_delta = torch.zeros_like(args.x_out[0])

        for child in that.children:
            child_cond_delta = child.accept(IntermediateCondDeltaVisitor(), args, index)
            child_cond_delta += child.accept(self, args, cond_delta, index)
            perp_cond_delta += child.weight * get_perpendicular_component(cond_delta, child_cond_delta)
            index += child.accept(neutral_prompt_parser.FlatSizeVisitor())

        return perp_cond_delta


def get_perpendicular_component(normal: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    if (normal == 0).all():
        if shared.state.sampling_step <= 0:
            warn_projection_not_found()

        return vector

    return vector - normal * torch.sum(normal * vector) / torch.norm(normal) ** 2


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
