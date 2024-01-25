import abc
import dataclasses
import re
from enum import Enum
from typing import List, Tuple, Any, Optional


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


@dataclasses.dataclass
class PromptExpr(abc.ABC):
    weight: float
    conciliation: Optional[ConciliationStrategy]

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
    return CompositePrompt(1., None, prompts)


def parse_prompts(tokens: List[str], *, nested: bool = False) -> List[PromptExpr]:
    prompts = [parse_prompt(tokens, first=True, nested=nested)]
    while tokens:
        if nested and tokens[0] in [']']:
            break

        prompts.append(parse_prompt(tokens, first=False, nested=nested))

    return prompts


def parse_prompt(tokens: List[str], *, first: bool, nested: bool = False) -> PromptExpr:
    if first:
        prompt_type = PromptKeyword.AND.value
    elif tokens[0] in prompt_keywords:
        prompt_type = tokens.pop(0)
    else:
        prompt_type = PromptKeyword.AND.value

    tokens_copy = tokens.copy()
    if tokens_copy and tokens_copy[0] == '[':
        tokens_copy.pop(0)
        prompts = parse_prompts(tokens_copy, nested=True)
        if tokens_copy:
            assert tokens_copy.pop(0) == ']'
        if len(prompts) > 1:
            tokens[:] = tokens_copy
            weight = parse_weight(tokens)
            conciliation = ConciliationStrategy(prompt_type) if prompt_type in conciliation_strategies else None
            return CompositePrompt(weight, conciliation, prompts)

    prompt_text, weight = parse_prompt_text(tokens, nested=nested)
    return LeafPrompt(weight, ConciliationStrategy(prompt_type) if prompt_type in conciliation_strategies else None, prompt_text)


def parse_prompt_text(tokens: List[str], *, nested: bool = False) -> Tuple[str, float]:
    text = ''
    depth = 0
    weight = 1.
    while tokens:
        if tokens[0] == ']':
            if depth == 0:
                if not nested:
                    text += tokens.pop(0)
                break
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
    if tokens and tokens[0] == ':':
        tokens.pop(0)
        if tokens:
            weight_str = tokens.pop(0)
            if is_float(weight_str):
                weight = float(weight_str)
    return weight


def tokenize(s: str):
    prompt_keywords_regex = '|'.join(rf'\b{keyword}\b' for keyword in prompt_keywords)
    return [s for s in re.split(rf'(\[|\]|:|{prompt_keywords_regex})', s) if s.strip()]


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
