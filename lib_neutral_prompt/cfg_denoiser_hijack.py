import math

import scipy
import torchvision
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
        cond_delta = prompt.accept(CondDeltaChildVisitor(), args, 0)
        aux_cond_delta = prompt.accept(AuxCondDeltaChildVisitor(), args, cond_delta, 0)
        cfg_cond = denoised[batch_i] + aux_cond_delta * cond_scale
        denoised[batch_i] = cfg_cond * get_cfg_rescale_factor(cfg_cond, uncond[batch_i] + cond_delta + aux_cond_delta)

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
    def visit_leaf_prompt(self, *args, **kwargs) -> Tuple[List[torch.Tensor], List[Tuple[int, float]]]:
        return [], []

    def visit_composite_prompt(
        self,
        that: neutral_prompt_parser.CompositePrompt,
        args: CombineDenoiseArgs,
        index_offset: int,
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, float]]]:
        sliced_x_out = []
        sliced_cond_indices = []

        index_in = 0
        for child in that.children:
            index_out = index_offset + len(sliced_x_out)
            child_x_out, child_cond_indices = child.accept(GatherWebuiCondsVisitor.SingleCondVisitor(), args.x_out, args.cond_indices[index_in], index_out)
            sliced_x_out.extend(child_x_out)
            sliced_cond_indices.extend(child_cond_indices)
            index_in += child.accept(neutral_prompt_parser.FlatSizeVisitor())

        return sliced_x_out, sliced_cond_indices

    @dataclasses.dataclass
    class SingleCondVisitor:
        def visit_leaf_prompt(
            self,
            that: neutral_prompt_parser.LeafPrompt,
            x_out: torch.Tensor,
            cond_info: Tuple[int, float],
            index: int,
        ) -> Tuple[List[torch.Tensor], List[Tuple[int, float]]]:
            return [x_out[cond_info[0]]], [(index, cond_info[1])]

        def visit_composite_prompt(self, *args, **kwargs) -> Tuple[List[torch.Tensor], List[Tuple[int, float]]]:
            return [], []


@dataclasses.dataclass
class CondDeltaChildVisitor:
    def visit_leaf_prompt(
        self,
        that: neutral_prompt_parser.LeafPrompt,
        args: CombineDenoiseArgs,
        index: int,
    ) -> torch.Tensor:
        return torch.zeros_like(args.x_out[0])

    def visit_composite_prompt(
        self,
        that: neutral_prompt_parser.CompositePrompt,
        args: CombineDenoiseArgs,
        index: int,
    ) -> torch.Tensor:
        cond_delta = torch.zeros_like(args.x_out[0])

        for child in that.children:
            cond_delta += child.weight * child.accept(CondDeltaVisitor(), args, index)
            index += child.accept(neutral_prompt_parser.FlatSizeVisitor())

        return cond_delta


@dataclasses.dataclass
class CondDeltaVisitor:
    def visit_leaf_prompt(
        self,
        that: neutral_prompt_parser.LeafPrompt,
        args: CombineDenoiseArgs,
        index: int,
    ) -> torch.Tensor:
        cond_info = args.cond_indices[index]
        if that.weight != cond_info[1]:
            console_warn(f'''
                An unexpected noise weight was encountered at prompt #{index}
                Expected :{that.weight}, but got :{cond_info[1]}
                This is likely due to another extension also monkey patching the webui noise blending function
                Please open a github issue so that the conflict can be resolved
            ''')

        return args.x_out[cond_info[0]] - args.uncond

    def visit_composite_prompt(
        self,
        that: neutral_prompt_parser.CompositePrompt,
        args: CombineDenoiseArgs,
        index: int,
    ) -> torch.Tensor:
        cond_delta = torch.zeros_like(args.x_out[0])

        if that.conciliation is None:
            for child in that.children:
                child_cond_delta = child.accept(CondDeltaChildVisitor(), args, index)
                child_cond_delta += child.accept(AuxCondDeltaChildVisitor(), args, child_cond_delta, index)
                cond_delta += child.weight * child_cond_delta

        return cond_delta


@dataclasses.dataclass
class AuxCondDeltaChildVisitor:
    def visit_leaf_prompt(
        self,
        that: neutral_prompt_parser.LeafPrompt,
        args: CombineDenoiseArgs,
        cond_delta: torch.Tensor,
        index: int,
    ) -> torch.Tensor:
        return torch.zeros_like(args.x_out[0])

    def visit_composite_prompt(
        self,
        that: neutral_prompt_parser.CompositePrompt,
        args: CombineDenoiseArgs,
        cond_delta: torch.Tensor,
        index: int,
    ) -> torch.Tensor:
        aux_cond_delta = torch.zeros_like(args.x_out[0])
        salient_cond_deltas = []

        for child in that.children:
            child_cond_delta = child.accept(CondDeltaChildVisitor(), args, index)
            child_cond_delta += child.accept(self, args, child_cond_delta, index)
            if isinstance(child, neutral_prompt_parser.CompositePrompt):
                if child.conciliation == neutral_prompt_parser.ConciliationStrategy.PERPENDICULAR:
                    aux_cond_delta += child.weight * get_perpendicular_component(cond_delta, child_cond_delta)
                elif child.conciliation == neutral_prompt_parser.ConciliationStrategy.SALIENCE_MASK:
                    salient_cond_deltas.append((child_cond_delta, child.weight))

            index += child.accept(neutral_prompt_parser.FlatSizeVisitor())

        aux_cond_delta += salient_blend(cond_delta, salient_cond_deltas)
        return aux_cond_delta


def get_perpendicular_component(normal: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    if (normal == 0).all():
        if shared.state.sampling_step <= 0:
            warn_projection_not_found()

        return vector

    return vector - normal * torch.sum(normal * vector) / torch.norm(normal) ** 2


from PIL import Image
images = []


def salient_blend(normal: torch.Tensor, vectors: List[Tuple[torch.Tensor, float]]) -> torch.Tensor:
    salience_maps = [get_salience(normal, *get_hacky_config()[0:3])] + [get_salience(vector, *get_hacky_config()[3:6]) for vector in [vector for vector, weight in vectors]]
    mask = torch.argmax(torch.stack(salience_maps, dim=0), dim=0)
    blur = torchvision.transforms.GaussianBlur(get_hacky_config()[6], sigma=get_hacky_config()[7])

    result = torch.zeros_like(normal)
    for mask_i, (vector, weight) in enumerate(vectors, start=1):
        filtered_mask = blur((mask == mask_i).float())
        if shared.state.sampling_step == 0 and len(images) > 1:
            images.clear()

        images.append(filtered_mask)
        # similarities = torch.stack([torch.sum(torch_missing_mul(images[i:])) for i in range(len(images))], dim=0).reshape((len(images), 1, 1, 1))
        # filtered_mask = torch.softmax((torch.stack(images, dim=0)).sum(0).flatten(), dim=0).reshape_as(vector)
        # if torch.min(filtered_mask) < torch.max(filtered_mask):
        #     filtered_mask = (filtered_mask - torch.min(filtered_mask)) / (torch.max(filtered_mask) - torch.min(filtered_mask))

        if get_hacky_config()[8]:
            # import matplotlib.pyplot as plt
            # xs = torch.arange(0, vector.shape[1])
            # ys = torch.arange(0, vector.shape[2])
            # x, y = torch.meshgrid(xs, ys, indexing='xy')
            # z = filtered_mask.sum(0)
            # ax = plt.axes(projection='3d')
            # ax.plot_surface(x.cpu().numpy(), y.cpu().numpy(), z.cpu().numpy())
            # plt.show()

            image = torchvision.transforms.functional.to_pil_image(filtered_mask[:3] + filtered_mask[3] / 3)
            print(f'mask max: {torch.max(filtered_mask)}')
            image = image.resize((image.size[0] * 8, image.size[1] * 8), Image.Resampling.NEAREST)
            image.show()

        result += weight * filtered_mask * (vector - normal)

        # if shared.state.sampling_step >= shared.state.sampling_steps - 2:
        #     images_stack = torch.abs(weight * filtered_mask * (vector - normal))
        #     torchvision.transforms.functional.to_pil_image(images_stack).show()

    return result


def torch_missing_mul(vectors: List[torch.Tensor]) -> torch.Tensor:
    if len(vectors) == 1:
        return torch.zeros_like(vectors[0])

    res = torch.ones_like(vectors[0])

    for vector in vectors:
        res *= vector

    return res


def get_salience(vector: torch.Tensor, hardness: float, pre_blur_kernel: float, pre_blur_sigma: float) -> torch.Tensor:
    blur = torchvision.transforms.GaussianBlur(pre_blur_kernel, sigma=pre_blur_sigma)
    return blur(torch.softmax(hardness * torch.abs(vector).flatten(), dim=0).reshape_as(vector))


def low_pass(vector: torch.Tensor, ratio: float):
    dft = torch.fft.rfft2(vector)
    dft_filter = torch.arange(0, torch.numel(dft[0]), device='cuda').reshape_as(dft[0])
    dft_filter = torch.stack([dft_filter] * dft.size(0), dim=0)
    dft_filter = torch.maximum(dft_filter // dft.size(1) / dft.size(2), dft_filter % dft.size(2) / dft.size(1))
    dft_filter = (dft_filter < ratio).float()
    dft_filter[:, 0, 0] = 0
    return torch.fft.irfft2(dft * dft_filter)


def high_pass(vector: torch.Tensor, ratio: float):
    dft = torch.fft.rfft2(vector)
    dft_filter = torch.arange(0, torch.numel(dft[0]), device='cuda').reshape_as(dft[0])
    dft_filter = torch.stack([dft_filter] * dft.size(0), dim=0)
    dft_filter = torch.maximum(dft_filter // dft.size(1) / dft.size(2), dft_filter % dft.size(2) / dft.size(1))
    dft_filter = (dft_filter > ratio).float()
    return torch.fft.irfft2(dft * dft_filter)


def get_hacky_config():
    import os
    relative_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'hacky_conf.txt')
    with open(relative_file_path, 'r') as file:
        floats = [float(line.strip()) for line in file.readlines() if line.strip()]

    return floats


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
