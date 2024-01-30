import abc
import dataclasses
import re
from enum import Enum
from typing import List, Tuple, Any, Optional
import torch
import math


class PromptKeyword(Enum):
    AND = 'AND'
    AND_PERP = 'AND_PERP'
    AND_SALT = 'AND_SALT'
    AND_TOPK = 'AND_TOPK'


prompt_keywords = [e.value for e in PromptKeyword]


class ConciliationStrategy(Enum):
    PERPENDICULAR = PromptKeyword.AND_PERP.value
    SALIENCE_MASK = PromptKeyword.AND_SALT.value
    SEMANTIC_GUIDANCE = PromptKeyword.AND_TOPK.value


conciliation_strategies = [e.value for e in ConciliationStrategy]


affine_transforms = {
    'ROTATE': lambda t, angle=0, *_: t @ torch.tensor([[math.cos(angle*2*math.pi), -math.sin(angle*2*math.pi), 0], [math.sin(angle*2*math.pi), math.cos(angle*2*math.pi), 0], [0, 0, 1]]),
    'SLIDE': lambda t, x=0, y=0, *_: t @ torch.tensor([[1, 0, x], [0, 1, y], [0, 0, 1]]),
    'SCALE': lambda t, x=0, y=None, *_: t @ torch.tensor([[x, 0, 0], [0, y if y is not None else x, 0], [0, 0, 1]]),
    'SHEAR': lambda t, x=0, y=None, *_: t @ torch.tensor([[1, math.tan(x*2*math.pi), 0], [math.tan((y if y is not None else x)*2*math.pi), 1, 0], [0, 0, 1]]),
}


@dataclasses.dataclass
class PromptExpr(abc.ABC):
    weight: float
    conciliation: Optional[ConciliationStrategy]
    local_transform: Optional[torch.Tensor]

    @abc.abstractmethod
    def accept(self, visitor, *args, **kwargs) -> Any:
        pass


@dataclasses.dataclass
class LeafPrompt(PromptExpr):
    prompt: str

    def accept(self, visitor, *args, **kwargs):
        return visitor.visit_leaf_prompt(self, *args, **kwargs)


@dataclasses.dataclass
class CompositePrompt(PromptExpr):
    children: List[PromptExpr]

    def accept(self, visitor, *args, **kwargs):
        return visitor.visit_composite_prompt(self, *args, **kwargs)


class FlatSizeVisitor:
    def visit_leaf_prompt(self, that: LeafPrompt) -> int:
        return 1

    def visit_composite_prompt(self, that: CompositePrompt) -> int:
        return sum(child.accept(self) for child in that.children) if that.children else 0


def parse_root(string: str) -> CompositePrompt:
    tokens = tokenize(string)
    prompts = parse_prompts(tokens)
    return CompositePrompt(1., None, None, prompts)


def parse_prompts(tokens: List[str], *, nested: bool = False) -> List[PromptExpr]:
    prompts = [parse_prompt(tokens, first=True, nested=nested)]
    while tokens:
        if nested and tokens[0] in [']']:
            break

        prompts.append(parse_prompt(tokens, first=False, nested=nested))

    return prompts


def parse_prompt(tokens: List[str], *, first: bool, nested: bool = False) -> PromptExpr:
    if not first and tokens[0] in prompt_keywords:
        prompt_type = tokens.pop(0)
    else:
        prompt_type = PromptKeyword.AND.value
    conciliation = ConciliationStrategy(prompt_type) if prompt_type in conciliation_strategies else None

    affine_transform = parse_affine_transform(tokens)

    tokens_copy = tokens.copy()
    if tokens_copy and tokens_copy[0] == '[':
        tokens_copy.pop(0)
        prompts = parse_prompts(tokens_copy, nested=True)
        if tokens_copy:
            assert tokens_copy.pop(0) == ']'
        if len(prompts) > 1:
            tokens[:] = tokens_copy
            weight = parse_weight(tokens)
            return CompositePrompt(weight, conciliation, affine_transform, prompts)

    prompt_text, weight = parse_prompt_text(tokens, nested=nested)
    return LeafPrompt(weight, conciliation, affine_transform, prompt_text)


def parse_affine_transform(tokens: List[str]):
    tokens_copy = tokens.copy()
    if tokens_copy and not tokens_copy[0].strip():
        tokens_copy.pop(0)
    affine_funcs = []

    while tokens_copy and tokens_copy[0] in affine_transforms:
        func = affine_transforms[tokens_copy.pop(0)]
        args = []

        if tokens_copy and tokens_copy[0] == '[':
            tokens_copy.pop(0)
        else:
            break

        if tokens_copy and tokens_copy[0] != ']':
            if tokens_copy[0].strip():
                try:
                    args.extend(float(a.strip()) for a in tokens_copy.pop(0).split(','))
                except ValueError:
                    break
            else:
                tokens_copy.pop(0)

        if tokens_copy and tokens_copy[0] == ']':
            tokens_copy.pop(0)
        else:
            break

        affine_funcs.append(lambda t, f=func, a=args: f(t, *a))
        if tokens_copy and not tokens_copy[0].strip():
            tokens_copy.pop(0)
        tokens[:] = tokens_copy

    if not affine_funcs:
        return None

    transform = torch.eye(3)[:-1]
    for affine_func in reversed(affine_funcs):
        transform = affine_func(transform)
    return transform


def parse_prompt_text(tokens: List[str], *, nested: bool = False) -> Tuple[str, float]:
    text = ''
    depth = 0
    weight = 1.
    while tokens:
        if tokens[0] == ']':
            if depth == 0:
                if nested:
                    break
            else:
                depth -= 1
        elif tokens[0] == '[':
            depth += 1
        elif tokens[0] == ':':
            if len(tokens) >= 2 and is_float(tokens[1].strip()):
                if len(tokens) < 3 or tokens[2] in prompt_keywords or tokens[2] == ']' and depth == 0:
                    tokens.pop(0)
                    weight = float(tokens.pop(0).strip())
                    break
        elif tokens[0] in prompt_keywords:
            break

        text += tokens.pop(0)

    return text, weight


def parse_weight(tokens: List[str]) -> float:
    weight = 1.
    if len(tokens) >= 2 and tokens[0] == ':' and is_float(tokens[1]):
        tokens.pop(0)
        weight = float(tokens.pop(0))
    return weight


def tokenize(s: str):
    prompt_keywords_regex = '|'.join(rf'\b{keyword}\b' for keyword in prompt_keywords)
    transform_keywords_regex = '|'.join(rf'\b{keyword}\b' for keyword in affine_transforms.keys())
    return [s for s in re.split(rf'(\[|\]|:|{prompt_keywords_regex}|{transform_keywords_regex})', s) if s.strip()]


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
        AND_SALT [
            very nested huh? what do you say :.0
        ]
    ]
    ''')
    pass
