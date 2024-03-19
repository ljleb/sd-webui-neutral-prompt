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
                child_x_out = args.x_out[args.cond_indices[index_in][0]]
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

            index += child.accept(neutral_prompt_parser.FlatSizeVisitor())

        aux_cond_delta += salient_blend(cond_delta, salient_cond_deltas)
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


try:
    from modules_forge import forge_sampler
    forge = True
except ImportError:
    forge = False


if forge:
    from ldm_patched.modules.samplers import sampling_function

    forge_sampler_hijacker = hijacker.ModuleHijacker.install_or_get(
        module=forge_sampler,
        hijacker_attribute='__forge_sample_hijacker',
        on_uninstall=script_callbacks.on_script_unloaded,
    )

    @forge_sampler_hijacker.hijack('forge_sample')
    def forge_sample(self, denoiser_params, cond_scale, cond_composition, original_function):
        if not global_state.is_enabled:
            return original_function(self, denoiser_params, cond_scale, cond_composition)

        model = self.inner_model.inner_model.forge_objects.unet.model
        control = self.inner_model.inner_model.forge_objects.unet.controlnet_linked_list
        extra_concat_condition = self.inner_model.inner_model.forge_objects.unet.extra_concat_condition
        x = denoiser_params.x
        timestep = denoiser_params.sigma
        model_options = self.inner_model.inner_model.forge_objects.unet.model_options
        seed = self.p.seeds[0]

        uncond = forge_sampler.cond_from_a1111_to_patched_ldm(denoiser_params.text_uncond)
        conds = forge_sampler.cond_from_a1111_to_patched_ldm_weighted(denoiser_params.text_cond, cond_composition)
        conds += uncond

        denoised = []

        for current_cond in conds:
            cond = [current_cond]
            cond[0]['strength'] = 1.0

            if extra_concat_condition is not None:
                image_cond_in = extra_concat_condition
            else:
                image_cond_in = denoiser_params.image_cond

            if isinstance(image_cond_in, torch.Tensor):
                if image_cond_in.shape[0] == x.shape[0] \
                        and image_cond_in.shape[2] == x.shape[2] \
                        and image_cond_in.shape[3] == x.shape[3]:
                    cond[0]['model_conds']['c_concat'] = CONDRegular(image_cond_in)

            if control is not None:
                cond[0]['control'] = control

            for modifier in model_options.get('conditioning_modifiers', []):
                model, x, timestep, _, cond, cond_scale, model_options, seed = modifier(model, x, timestep, None, cond, cond_scale, model_options, seed)

            model_options["disable_cfg1_optimization"] = True

            result = sampling_function(model, x, timestep, None, cond, 1.0, model_options, seed)
            denoised.append(result)

        cond_indices = cond_composition[0]
        prompt = global_state.prompt_exprs[0]

        # B, C, H, W
        denoised_uncond = denoised[-1]

        # N, B, C, H, W
        denoised_conds = torch.stack(denoised[:-1], dim=0)

        # N, 1, 1, 1, 1
        weights = torch.tensor([ weight for (_, weight) in cond_indices ], device=denoised_uncond.device)
        weights /= weights.abs().sum()
        weights = weights.view(-1, 1, 1, 1, 1)

        # B, C, H, W
        assert denoised_conds.shape[0] == weights.shape[0]
        denoised_cond = (denoised_conds * weights).sum(dim=0)
        forge_denoised = denoised_uncond + (denoised_cond - denoised_uncond) * cond_scale

        for batch_i in range(denoised_uncond.shape[0]):
            args = CombineDenoiseArgs(denoised_conds.unbind(dim=1)[batch_i], denoised_uncond[batch_i], cond_indices)
            cond_delta = prompt.accept(CondDeltaVisitor(), args, 0)
            aux_cond_delta = prompt.accept(AuxCondDeltaVisitor(), args, cond_delta, 0)
            cfg_cond = forge_denoised[batch_i] + aux_cond_delta * cond_scale
            forge_denoised[batch_i] = cfg_rescale(cfg_cond, denoised_uncond[batch_i] + cond_delta + aux_cond_delta)

        return forge_denoised


else:
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
