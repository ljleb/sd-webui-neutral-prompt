from lib_neutral_prompt import hijacker, global_state, neutral_prompt_parser
from modules import shared
import importlib
import torch
import torchvision
import sys
import textwrap
from typing import List, Tuple


mask_images = []


def reload(self):
    global mask_images
    images = mask_images
    importlib.reload(self)
    self.mask_images = images


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

    salience_maps = [get_salience(normal)] + [get_salience(vector, specific=True) for vector, weight in vectors]
    mask = torch.argmax(torch.stack(salience_maps, dim=0), dim=0)

    result = torch.zeros_like(normal)
    for mask_i, (vector, weight) in enumerate(vectors, start=1):
        vector_mask = (mask == mask_i).float()

        blur = torchvision.transforms.GaussianBlur(3, 1.)

        # vector_mask = blur(life(vector_mask))
        # vector_mask = life(vector_mask)
        for _ in range(6):
            vector_mask = life(vector_mask, lambda board, neighbors: (board == 1) & (neighbors >= board.size(0) * 5))

        for _ in range(2):
            vector_mask = life(vector_mask, thickify_rules)

        display_mask = vector_mask[:3] * 2/3 + vector_mask[3] / 3
        display_mask = torch.nn.functional.interpolate(display_mask.unsqueeze(0), scale_factor=8, mode='nearest-exact').squeeze(0)

        if shared.state.sampling_step == 0 and len(mask_images) >= 2:
            mask_images.clear()

        mask_images.append(torchvision.transforms.functional.to_pil_image(display_mask))
        result += weight * vector_mask * (vector - normal)

    return result


def thickify_rules(board, neighbors):
    population = board + neighbors
    return (board == 1) | (population >= 4)


def life(board: torch.Tensor, rules = None):
    if rules is None:
        rules = lambda board, neighbors: (neighbors == board.size(0) * 3) | ((board == 1) & (neighbors >= board.size(0) * 3) & (neighbors <= board.size(0) * 4))
    kernel = torch.tensor(
        [[[1, 1, 1],
          [1, 1, 1],
          [1, 1, 1]]] * board.size(0),
        dtype=board.dtype,
        device=board.device,
    )
    padded_board = torch.concatenate([board.clone(), board[:-1].clone()], dim=0)
    padded_board = torch.nn.functional.pad(padded_board, (1, 1, 1, 1, 0, 0), mode='constant', value=0)
    neighbors = torch.nn.functional.conv3d(
        padded_board.unsqueeze(0).unsqueeze(0),
        kernel.unsqueeze(0).unsqueeze(0),
        padding=0,
    ).squeeze(0).squeeze(0)
    return rules(board, neighbors - board).float()


def get_salience(vector: torch.Tensor, specific: bool = False) -> torch.Tensor:
    k = 1
    if specific:
        k = 20
    return torch.softmax(k * torch.abs(vector).flatten(), dim=0).reshape_as(vector)


def warn_projection_not_found():
    console_warn('''
        Could not find a projection for one or more AND_PERP prompts
        These prompts will NOT be made perpendicular
    ''')


def console_warn(message):
    if not global_state.verbose:
        return

    print(f'\n[sd-webui-neutral-prompt extension]{textwrap.dedent(message)}', file=sys.stderr)
