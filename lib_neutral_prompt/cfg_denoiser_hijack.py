from lib_neutral_prompt import hijacker, global_state, neutral_prompt_parser
from modules import script_callbacks, sd_samplers, shared
from typing import Tuple, List
import dataclasses
import functools
import torch
import torch.nn.functional as F
import sys
import textwrap
import re


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
        cond_delta = prompt.accept(CondDeltaVisitor(), args, 0)
        aux_cond_delta = prompt.accept(AuxCondDeltaVisitor(), args, cond_delta, 0)
        cfg_cond = denoised[batch_i] + aux_cond_delta * cond_scale
        denoised[batch_i] = cfg_rescale(cfg_cond, uncond[batch_i] + cond_delta + aux_cond_delta)

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
        sliced_x_out, sliced_cond_indices = gather_webui_conds(prompt, args, 0, len(sliced_batch_x_out))
        if sliced_cond_indices:
            sliced_batch_cond_indices.append(sliced_cond_indices)
        sliced_batch_x_out.extend(sliced_x_out)

    sliced_batch_x_out += list(uncond)
    sliced_batch_x_out = torch.stack(sliced_batch_x_out, dim=0)
    return original_function(sliced_batch_x_out, sliced_batch_cond_indices, text_uncond, cond_scale)


def cfg_rescale(cfg_cond, cond):
    if global_state.cfg_rescale == 0:
        return cfg_cond

    global_state.apply_and_clear_cfg_rescale_override()
    cfg_cond_mean = cfg_cond.mean()
    cfg_rescale_mean = (1 - global_state.cfg_rescale) * cfg_cond_mean + global_state.cfg_rescale * cond.mean()
    cfg_rescale_factor = global_state.cfg_rescale * (cond.std() / cfg_cond.std() - 1) + 1
    return cfg_rescale_mean + (cfg_cond - cfg_cond_mean) * cfg_rescale_factor


@dataclasses.dataclass
class CombineDenoiseArgs:
    x_out: torch.Tensor
    uncond: torch.Tensor
    cond_indices: List[Tuple[int, float]]


def gather_webui_conds(
    prompt: neutral_prompt_parser.CompositePrompt,
    args: CombineDenoiseArgs,
    index_in: int,
    index_out: int,
) -> Tuple[List[torch.Tensor], List[Tuple[int, float]]]:
    sliced_x_out = []
    sliced_cond_indices = []

    for child in prompt.children:
        if child.conciliation is None:
            if isinstance(child, neutral_prompt_parser.LeafPrompt):
                child_x_out = args.x_out[index_in]
            else:
                child_x_out = child.accept(CondDeltaVisitor(), args, index_in)
                child_x_out += child.accept(AuxCondDeltaVisitor(), args, child_x_out, index_in)
                child_x_out += args.uncond
            index_offset = index_out + len(sliced_x_out)
            sliced_x_out.append(child_x_out)
            sliced_cond_indices.append((index_offset, child.weight))

        index_in += child.accept(neutral_prompt_parser.FlatSizeVisitor())

    return sliced_x_out, sliced_cond_indices


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
                This is likely due to another extension also monkey patching the webui `combine_denoised` function
                Please open a bug report here so that the conflict can be resolved:
                https://github.com/ljleb/sd-webui-neutral-prompt/issues
            ''')

        return args.x_out[cond_info[0]] - args.uncond

    def visit_composite_prompt(
        self,
        that: neutral_prompt_parser.CompositePrompt,
        args: CombineDenoiseArgs,
        index: int,
    ) -> torch.Tensor:
        cond_delta = torch.zeros_like(args.x_out[0])

        for child in that.children:
            if child.conciliation is None:
                child_cond_delta = child.accept(CondDeltaVisitor(), args, index)
                child_cond_delta += child.accept(AuxCondDeltaVisitor(), args, child_cond_delta, index)
                cond_delta += child.weight * child_cond_delta

            index += child.accept(neutral_prompt_parser.FlatSizeVisitor())

        return cond_delta


class AuxCondDeltaVisitor:
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
        align_blend_cond_deltas = []

        for child in that.children:
            if child.conciliation is not None:
                child_cond_delta = child.accept(CondDeltaVisitor(), args, index)
                child_cond_delta += child.accept(AuxCondDeltaVisitor(), args, child_cond_delta, index)

                if child.conciliation == neutral_prompt_parser.ConciliationStrategy.PERPENDICULAR:
                    aux_cond_delta += child.weight * get_perpendicular_component(cond_delta, child_cond_delta)
                elif child.conciliation == neutral_prompt_parser.ConciliationStrategy.SALIENCE_MASK:
                    salient_cond_deltas.append((child_cond_delta, child.weight))
                elif child.conciliation == neutral_prompt_parser.ConciliationStrategy.SEMANTIC_GUIDANCE:
                    aux_cond_delta += child.weight * filter_abs_top_k(child_cond_delta, 0.05)
                else:
                    match = re.match(r'AND_ALIGN_(\d+)_(\d+)', child.conciliation.value)
                    if match:
                        detail_size, structure_size = int(match.group(1)), int(match.group(2))
                        align_blend_cond_deltas.append((child_cond_delta, child.weight, detail_size, structure_size))

            index += child.accept(neutral_prompt_parser.FlatSizeVisitor())

        aux_cond_delta += salient_blend(cond_delta, salient_cond_deltas)
        aux_cond_delta += alignment_blend(cond_delta, align_blend_cond_deltas)
        return aux_cond_delta


def get_perpendicular_component(normal: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    if (normal == 0).all():
        if shared.state.sampling_step <= 0:
            warn_projection_not_found()

        return vector

    return vector - normal * torch.sum(normal * vector) / torch.norm(normal) ** 2


def salient_blend(normal: torch.Tensor, vectors: List[Tuple[torch.Tensor, float]]) -> torch.Tensor:
    """
        Blends the `normal` tensor with `vectors` in salient regions, weighting contributions by their weights.
        Salience maps are calculated to identify regions of interest.
        The blended result combines `normal` and vector information in salient regions.
    """

    salience_maps = [get_salience(normal)] + [get_salience(vector) for vector, _ in vectors]
    mask = torch.argmax(torch.stack(salience_maps, dim=0), dim=0)

    result = torch.zeros_like(normal)
    for mask_i, (vector, weight) in enumerate(vectors, start=1):
        vector_mask = (mask == mask_i).float()
        result += weight * vector_mask * (vector - normal)

    return result


def get_salience(vector: torch.Tensor) -> torch.Tensor:
    return torch.softmax(torch.abs(vector).flatten(), dim=0).reshape_as(vector)


def filter_abs_top_k(vector: torch.Tensor, k_ratio: float) -> torch.Tensor:
    k = int(torch.numel(vector) * (1 - k_ratio))
    top_k, _ = torch.kthvalue(torch.abs(torch.flatten(vector)), k)
    return vector * (torch.abs(vector) >= top_k).to(vector.dtype)


def compute_subregion_similarity_map(child_vector: torch.Tensor,
                                     parent_vector: torch.Tensor,
                                     region_size: int = 2) -> torch.Tensor:
    """
    Compute the local average cosine similarity of 2x2 subregions of RxR regions of parent and child diffusion gradients.

    :param child_vector: Latent score vector for the child prompt. Shape: [C, H, W]
    :param parent_vector: Latent score vector for the parent prompt. Shape: [C, H, W]
    :param region_size: Size R of the local region. Default is 2.
    :return: Local alignment map. Shape: [C, H, W]
    """

    C, H, W = child_vector.shape

    # Step 1: Adjust input shape to include batch dimension. Shape: [1, C, H, W]
    parent = parent_vector.unsqueeze(0)
    child = child_vector.unsqueeze(0)

    # Step 2: Extract local regions using Unfold and reshape.
    # Apply symmetric padding for odd region size R, else use asymmetric padding when region size is even.
    # This is necessary to ensure exactly H*W regions are unfolded. For example, when R=4, the tensors
    # will be padded by 1 extra row of 0s to the top and to the left, and two extra rows of 0s to the
    # bottom and to the right, ensuring we can extract exactly H*W regions of shape 4x4. The odd cases are
    # easier because the kernels are centered on the original pixels; for example in 5x5 we just pad by two
    # rows of extra 0s in all directions.
    region_radius = region_size // 2

    if region_size % 2 == 1:
        pad_size = (region_radius,) * 4
    else:
        pad_size = (region_radius - 1, region_radius,) * 2

    parent_regions = F.pad(parent, pad_size, "constant", 0)
    child_regions = F.pad(child, pad_size, "constant", 0)
    unfold = torch.nn.Unfold(kernel_size=region_size)
    parent_regions = unfold(parent_regions)
    child_regions = unfold(child_regions)

    # Step 3: Separate channel dimension, move spatial dimension to batch dimension, separate region spatial dimensions
    # The original spatial dimensions are treated as the new batch dimension, because we will later unfold a 2nd time
    # over the newly created region spatial dimensions only to extract the 2x2 subregions. To prepare for this, we need the
    # Shape to be [H*W, C, region_size, region_size]
    parent_regions = parent_regions.view(1, C, region_size**2, H*W).permute(3, 1, 2, 0).view(H*W, C, region_size, region_size)
    child_regions = child_regions.view(1, C, region_size**2, H*W).permute(3, 1, 2, 0).view(H*W, C, region_size, region_size)

    # Step 4: Extract local 2x2 subregions from each region using Unfold. Do not pad regions when extracting subregions.
    # Re-separate channel dimension from subregion spatial dimension after unfolding
    # Shape: [H*W, C*4, (region_size - 1)**2] -> [H*W, C, 4, (region_size - 1)**2]
    unfold = torch.nn.Unfold(kernel_size=2)
    parent_subregions = unfold(parent_regions).view(H*W, C, 4, (region_size - 1)**2)
    child_subregions = unfold(child_regions).view(H*W, C, 4, (region_size - 1)**2)

    # Step 5: Normalize the subregions and compute cosine similarity
    # Shape: [H*W, C, (region_size - 1)**2]
    parent_subregions = F.normalize(parent_subregions, p=2, dim=2)
    child_subregions = F.normalize(child_subregions, p=2, dim=2)
    subregion_similarity_map = (parent_subregions * child_subregions).sum(dim=2)

    # Step 6: Average subregion similarity per region and reshape back to original
    # Shape: [H*W, C] -> [C, H*W] -> [C, H, W]
    return subregion_similarity_map.mean(dim=2).permute(1, 0).view(C, H, W)


def alignment_blend(parent: torch.Tensor,
                   children: list[tuple[torch.Tensor, float, int, int]]) -> torch.Tensor:
    """
    Perform a locally weighted blend of parent and multiple children by comparing subregion similarity maps.
    For each child, two subregion similarity maps are computed against parent, using detail and structure scales.
    The child receives increased local weight when subregion structure alignment is relatively higher than subregion
    detail alignment; in other words, when the child can alter the details of the parent, without breaking the
    structure of the composition.

    :param parent: Latent score vector for the parent prompt. Shape: [C, H, W]
    :param children: List of tuples (child_vector, weight, detail_radius, structure_radius).
                                    child_vector is a latent gradient flow vector.
                                    weight is a global weight for that gradient flow.
                                    detail_radius is a kernel radius for detecting new details added by the child.
                                    structure_radius is a kernel radius for detecting when the added details are structure preserving.

    :return: Cond delta from parent latent gradient flow vector to alignment blended latent gradient flow vector. Shape: [C, H, W]
    """
    result = torch.zeros_like(parent)

    # loop over children, blending each into parent gradient flow
    for child, weight, detail_size, structure_size in children:
        detail_alignment = compute_subregion_similarity_map(child, parent, region_size=detail_size)
        structure_alignment = compute_subregion_similarity_map(child, parent, region_size=structure_size)

        detail_alignment = detail_alignment / detail_alignment.max()
        structure_alignment = structure_alignment / structure_alignment.max()

        # Compute alignment_weight as structure-to-detail alignment difference.
        # This is higher when child flow changes detail but preserves parent structure.
        # Clamping ensures the difference is in a valid range [0, 1]

        alignment_weight = structure_alignment - detail_alignment
        alignment_weight = torch.clamp(alignment_weight, min=0, max=1.0)

        # Blend the child into the parent using the computed weight and alignment weight
        result += (child - parent) * weight * alignment_weight

    return result


sd_samplers_hijacker = hijacker.ModuleHijacker.install_or_get(
    module=sd_samplers,
    hijacker_attribute='__neutral_prompt_hijacker',
    on_uninstall=script_callbacks.on_script_unloaded,
)


@sd_samplers_hijacker.hijack('create_sampler')
def create_sampler_hijack(name: str, model, original_function):
    sampler = original_function(name, model)
    if not hasattr(sampler, 'model_wrap_cfg') or not hasattr(sampler.model_wrap_cfg, 'combine_denoised'):
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
