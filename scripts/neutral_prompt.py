import dataclasses

from lib_neutral_prompt import global_state, hijacker, neutral_prompt_parser, prompt_parser_hijack, cfg_denoiser_hijack, ui, xyz_grid
from modules import scripts, processing, shared, script_callbacks
from typing import Dict, List, Tuple
import torch
import functools


sampling_step = 0


class NeutralPromptScript(scripts.Script):
    def __init__(self):
        self.accordion_interface = None
        self._is_img2img = False

    @property
    def is_img2img(self):
        return self._is_img2img

    @is_img2img.setter
    def is_img2img(self, is_img2img):
        self._is_img2img = is_img2img
        if self.accordion_interface is None:
            self.accordion_interface = ui.AccordionInterface(self.elem_id)

    def title(self) -> str:
        return "Neutral Prompt"

    def show(self, is_img2img: bool):
        return scripts.AlwaysVisible

    def ui(self, is_img2img: bool):
        self.hijack_composable_lora(is_img2img)

        self.accordion_interface.arrange_components(is_img2img)
        self.accordion_interface.connect_events(is_img2img)
        self.infotext_fields = self.accordion_interface.get_infotext_fields()
        self.paste_field_names = self.accordion_interface.get_paste_field_names()
        self.accordion_interface.set_rendered()
        return self.accordion_interface.get_components()

    def process(self, p: processing.StableDiffusionProcessing, *args):
        args = self.accordion_interface.unpack_processing_args(*args)

        self.update_global_state(args)
        if global_state.is_enabled:
            p.extra_generation_params.update(self.accordion_interface.get_extra_generation_params(args))

        global sampling_step
        sampling_step = 0

    def update_global_state(self, args: Dict):
        if shared.state.job_no == 0:
            global_state.is_enabled = shared.opts.data.get('neutral_prompt_enabled', True)

        for k, v in args.items():
            try:
                getattr(global_state, k)
            except AttributeError:
                continue

            if getattr(getattr(global_state, k), 'is_xyz', False):
                xyz_attr = getattr(global_state, k)
                xyz_attr.is_xyz = False
                args[k] = xyz_attr
                continue

            if shared.state.job_no > 0:
                continue

            setattr(global_state, k, v)

    def hijack_composable_lora(self, is_img2img):
        if self.accordion_interface.is_rendered:
            return

        lora_script = None
        script_runner = scripts.scripts_img2img if is_img2img else scripts.scripts_txt2img

        for script in script_runner.alwayson_scripts:
            if script.title().lower() == "composable lora":
                lora_script = script
                break

        if lora_script is not None:
            lora_script.process = functools.partial(composable_lora_process_hijack, original_function=lora_script.process)


def composable_lora_process_hijack(p: processing.StableDiffusionProcessing, *args, original_function, **kwargs):
    if not global_state.is_enabled:
        return original_function(p, *args, **kwargs)

    exprs = prompt_parser_hijack.parse_prompts(p.all_prompts)
    all_prompts, p.all_prompts = p.all_prompts, prompt_parser_hijack.transpile_exprs(exprs)
    res = original_function(p, *args, **kwargs)
    # restore original prompts
    p.all_prompts = all_prompts
    return res


xyz_grid.patch()


@dataclasses.dataclass
class CombinePreNoiseArgs:
    x_out: torch.Tensor
    cond_indices: List[Tuple[int, float]]


noises = []


def on_cfg_denoiser(params: script_callbacks.CFGDenoiserParams):
    if not global_state.is_enabled:
        return

    global noises, sampling_step
    sampling_step += 1
    if sampling_step == 1:
        noises = params.x.clone()
        return

    for batch_i, (prompt, cond_indices) in enumerate(zip(global_state.prompt_exprs, global_state.batch_cond_indices)):
        args = CombinePreNoiseArgs(params.x, cond_indices)
        inv_transforms = prompt.accept(GlobalToLocalAffineVisitor(), args, 0)
        for cond_index, weight in cond_indices:
            # noisy_component = noises[cond_index] * torch.sum(noises[cond_index] * params.x[cond_index]) / torch.norm(noises[cond_index]) ** 2
            # params.x[cond_index] = apply_affine_transform(params.x[cond_index] - noisy_component, inv_transforms[cond_index]) + noisy_component
            params.x[cond_index] = apply_affine_transform(params.x[cond_index], inv_transforms[cond_index])


script_callbacks.on_cfg_denoiser(on_cfg_denoiser)


class GlobalToLocalAffineVisitor:
    def visit_leaf_prompt(
        self,
        that: neutral_prompt_parser.LeafPrompt,
        args: CombinePreNoiseArgs,
        index: int,
    ) -> Dict[int, torch.Tensor]:
        cond_index = args.cond_indices[index][0]
        transform = torch.linalg.inv(torch.vstack([that.local_transform, torch.tensor([0, 0, 1])]))[:-1] if that.local_transform is not None else torch.eye(3)[:-1]
        return {cond_index: transform}

    def visit_composite_prompt(
        self,
        that: neutral_prompt_parser.CompositePrompt,
        args: CombinePreNoiseArgs,
        index: int,
    ) -> Dict[int, torch.Tensor]:
        inv_transforms = {}

        for child in that.children:
            inv_transforms.update(child.accept(GlobalToLocalAffineVisitor(), args, index))
            index += child.accept(neutral_prompt_parser.FlatSizeVisitor())

        if that.local_transform is not None:
            that_inv_transform = torch.linalg.inv(torch.vstack([that.local_transform, torch.tensor([0, 0, 1])]))
            for inv_transform in inv_transforms.values():
                inv_transform[:] = that_inv_transform @ inv_transform

        return inv_transforms


import torch.nn.functional as F
def apply_affine_transform(tensor, affine):
    affine = affine.to(tensor.device)
    aspect_ratio = tensor.shape[-2] / tensor.shape[-1]
    affine[0, 1] *= aspect_ratio
    affine[1, 0] /= aspect_ratio

    grid = F.affine_grid(affine.unsqueeze(0), tensor.unsqueeze(0).size(), align_corners=False)
    transformed_tensors = F.grid_sample(tensor.unsqueeze(0), grid, align_corners=False)
    return transformed_tensors.squeeze(0)
