import dataclasses
import sys
import textwrap
from typing import List, Tuple
import re
import torch

from modules import shared


@dataclasses.dataclass
class Prompt:
    weight: float

    def get_webui_prompt(self) -> str:
        raise NotImplementedError

    def flat_size(self) -> int:
        raise NotImplementedError

    def get_shallow_cond_delta(
        self,
        x_out: torch.Tensor,
        uncond: torch.Tensor,
        cond_info: Tuple[int, float],
    ) -> torch.Tensor:
        raise NotImplementedError

    def get_perp_cond_delta(
        self,
        x_out: torch.Tensor,
        cond_delta: torch.Tensor,
        uncond: torch.Tensor,
        cond_indices: List[Tuple[int, float]],
        index: int,
    ) -> torch.Tensor:
        raise NotImplementedError


@dataclasses.dataclass
class ComposablePrompt(Prompt):
    prompt: str

    def get_webui_prompt(self) -> str:
        prompt = re.sub(r'\s+', ' ', self.prompt).strip()
        return f'{prompt} :{self.weight}'

    def flat_size(self) -> int:
        return 1

    def get_shallow_cond_delta(
        self,
        x_out: torch.Tensor,
        uncond: torch.Tensor,
        cond_info: Tuple[int, float]
    ) -> torch.Tensor:
        return cond_info[1] * (x_out[cond_info[0]] - uncond)

    def get_perp_cond_delta(
        self,
        x_out: torch.Tensor,
        cond_delta: torch.Tensor,
        uncond: torch.Tensor,
        cond_indices: List[Tuple[int, float]],
        index: int,
    ) -> torch.Tensor:
        return torch.zeros_like(x_out[0])


@dataclasses.dataclass
class CompositePrompt(Prompt):
    children: List[Prompt]

    def get_webui_prompt(self) -> str:
        return ' AND '.join(child.get_webui_prompt() for child in self.children)

    def flat_size(self) -> int:
        return sum(child.flat_size() for child in self.children) if self.children else 0

    def get_shallow_cond_delta(
        self,
        x_out: torch.Tensor,
        uncond: torch.Tensor,
        cond_info: Tuple[int, float]
    ) -> torch.Tensor:
        return torch.zeros_like(x_out[0])

    def get_cond_delta(
        self,
        x_out: torch.Tensor,
        uncond: torch.Tensor,
        cond_indices: List[Tuple[int, float]],
        index: int,
    ) -> torch.Tensor:
        cond_delta = torch.zeros_like(x_out[0])

        for child in self.children:
            cond_info = cond_indices[index]
            cond_delta += child.get_shallow_cond_delta(x_out, uncond, cond_info)
            index += child.flat_size()

        return cond_delta

    def get_perp_cond_delta(
        self,
        x_out: torch.Tensor,
        cond_delta: torch.Tensor,
        uncond: torch.Tensor,
        cond_indices: List[Tuple[int, float]],
        index: int,
    ) -> torch.Tensor:
        perp_cond_delta = torch.zeros_like(x_out[0])

        for child in self.children:
            if isinstance(child, CompositePrompt):
                child_cond_delta = child.get_cond_delta(x_out, uncond, cond_indices, index)
                child_cond_delta += child.get_perp_cond_delta(x_out, child_cond_delta, uncond, cond_indices, index)
                perp_cond_delta += child.weight * get_perpendicular_component(cond_delta, child_cond_delta)
            index += child.flat_size()

        return perp_cond_delta


def get_perpendicular_component(normal: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    if (normal == 0).all():
        if shared.state.sampling_step <= 0:
            warn_projection_not_found()

        return vector

    return vector - normal * torch.sum(normal * vector) / torch.norm(normal) ** 2


def warn_projection_not_found():
    console_warn('''
        Could not find a projection for one or more AND_PERP prompts
        These prompts will NOT be made perpendicular
    ''')


def console_warn(message):
    print(f'\n[sd-webui-neutral-prompt extension]{textwrap.dedent(message)}', file=sys.stderr)


def parse_root(string: str) -> Prompt:
    tokens = tokenize(string)
    prompts = parse_prompts(tokens)
    return CompositePrompt(1., prompts)


def parse_prompts(tokens: List[str]) -> List[Prompt]:
    prompts = [parse_prompt(tokens)]
    while tokens:
        if tokens[0] in [']']:
            break

        prompts.append(parse_prompt(tokens, first=False))

    return prompts


def parse_prompt(tokens: List[str], first: bool = True) -> Prompt:
    if first:
        prompt_type = 'AND'
    else:
        assert tokens[0] in ['AND', 'AND_PERP']
        prompt_type = tokens.pop(0)

    if prompt_type == 'AND':
        prompt, weight = parse_prompt_text(tokens)
        return ComposablePrompt(weight, prompt)
    else:
        if tokens[0] == '[':
            tokens.pop(0)
            prompts = parse_prompts(tokens)
            if tokens:
                assert tokens.pop(0) == ']'
            weight = parse_weight(tokens)
            return CompositePrompt(weight, prompts)
        else:
            prompt, weight = parse_prompt_text(tokens)
            return CompositePrompt(1., [ComposablePrompt(weight, prompt)])


def parse_prompt_text(tokens: List[str]) -> Tuple[str, float]:
    text = ''
    depth = 0
    weight = 1.
    while tokens:
        if tokens[0] == ']':
            if depth == 0:
                break
            depth -= 1
        elif tokens[0] == '[':
            depth += 1
        elif tokens[0] == ':':
            if len(tokens) >= 2 and is_float(tokens[1].strip()):
                if len(tokens) < 3 or tokens[2] in ['AND', 'AND_PERP'] or tokens[2] == ']' and depth == 0:
                    tokens.pop(0)
                    weight = float(tokens.pop(0).strip())
                    break
        elif tokens[0] in ['AND', 'AND_PERP']:
            break

        text += tokens.pop(0)

    return text, weight


def parse_weight(tokens: List[str]) -> float:
    weight = 1.
    if tokens and tokens[0] == ':':
        tokens.pop(0)
        if tokens:
            weight_str = tokens.pop(0)
            if is_float(weight_str):
                weight = float(weight_str)
    return weight


def tokenize(s: str):
    s = re.sub(r'\s+', ' ', s).strip()
    return [s for s in re.split(r'(\[|\]|:|\bAND_PERP\b|\bAND\b)', s) if s.strip()]


def is_float(string: str) -> bool:
    try:
        float(string)
        return True
    except ValueError:
        return False


if __name__ == '__main__':
    res = parse_root('''
    hello
    AND_PERP [
        arst
        AND defg : 2
        AND_PERP [
            very nested huh? what  [do you say :.0
        ]
    ]
    ''')
    pass
