import abc
import dataclasses
import re
from enum import Enum
from typing import List, Tuple, Any, Optional


@dataclasses.dataclass
class PromptExpr(abc.ABC):
    weight: float

    @abc.abstractmethod
    def accept(self, visitor, *args, **kwargs) -> Any:
        pass


@dataclasses.dataclass
class LeafPrompt(PromptExpr):
    prompt: str

    def accept(self, visitor, *args, **kwargs):
        return visitor.visit_leaf_prompt(self, *args, **kwargs)


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
class CompositePrompt(PromptExpr):
    children: List[PromptExpr]
    conciliation: Optional[ConciliationStrategy]

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
    return CompositePrompt(1., prompts, None)


def parse_prompts(tokens: List[str]) -> List[PromptExpr]:
    prompts = [parse_prompt(tokens, first=True)]
    while tokens:
        if tokens[0] in [']']:
            break

        prompts.append(parse_prompt(tokens, first=False))

    return prompts


def parse_prompt(tokens: List[str], *, first: bool) -> PromptExpr:
    if first:
        prompt_type = PromptKeyword.AND.value
    else:
        assert tokens[0] in prompt_keywords
        prompt_type = tokens.pop(0)

        if tokens and tokens[0] == '[':
            tokens.pop(0)
            prompts = parse_prompts(tokens)
            if tokens:
                assert tokens.pop(0) == ']'
            weight = parse_weight(tokens)
            conciliation = ConciliationStrategy(prompt_type) if prompt_type in conciliation_strategies else None
            return CompositePrompt(weight, prompts, conciliation)

    prompt_text, weight = parse_prompt_text(tokens)
    prompt = LeafPrompt(weight, prompt_text)
    if prompt_type in conciliation_strategies:
        prompt.weight = 1.
        prompt = CompositePrompt(weight, [prompt], ConciliationStrategy(prompt_type))

    return prompt


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
    s = re.sub(r'\s+', ' ', s).strip()
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
