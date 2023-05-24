import abc
import dataclasses
import re
from typing import List, Tuple, Any


@dataclasses.dataclass
class Prompt(abc.ABC):
    weight: float

    @abc.abstractmethod
    def accept(self, visitor, *args, **kwargs) -> Any:
        pass


@dataclasses.dataclass
class ComposablePrompt(Prompt):
    prompt: str

    def accept(self, visitor, *args, **kwargs):
        return visitor.visit_composable_prompt(self, *args, **kwargs)


@dataclasses.dataclass
class CompositePrompt(Prompt):
    children: List[Prompt]

    def accept(self, visitor, *args, **kwargs):
        return visitor.visit_composite_prompt(self, *args, **kwargs)


class FlatSizeVisitor:
    def visit_composable_prompt(self, that: ComposablePrompt) -> int:
        return 1

    def visit_composite_prompt(self, that: CompositePrompt) -> int:
        return sum(child.accept(self) for child in that.children) if that.children else 0


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
