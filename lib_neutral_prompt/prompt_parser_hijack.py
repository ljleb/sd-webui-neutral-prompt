from typing import List

from lib_neutral_prompt import hijacker, global_state, neutral_prompt_parser
from modules import script_callbacks, prompt_parser
import re


prompt_parser_hijacker = hijacker.ModuleHijacker.install_or_get(
    module=prompt_parser,
    hijacker_attribute='__neutral_prompt_hijacker',
    on_uninstall=script_callbacks.on_script_unloaded,
)


@prompt_parser_hijacker.hijack('get_multicond_learned_conditioning')
def get_multicond_learned_conditioning_hijack(model, prompts, steps, original_function):
    if not global_state.is_enabled:
        return original_function(model, prompts, steps)

    global_state.prompt_exprs = parse_prompts(prompts)
    return original_function(model, transpile_exprs(global_state.prompt_exprs), steps)


def parse_prompts(prompts: List[str]) -> neutral_prompt_parser.PromptExpr:
    exprs = []
    for prompt in prompts:
        expr = neutral_prompt_parser.parse_root(prompt)
        exprs.append(expr)

    return exprs


def transpile_exprs(exprs: neutral_prompt_parser.PromptExpr):
    webui_prompts = []
    for expr in exprs:
        webui_prompts.append(expr.accept(WebuiPromptVisitor()))

    return webui_prompts


class WebuiPromptVisitor:
    def visit_leaf_prompt(self, that: neutral_prompt_parser.LeafPrompt) -> str:
        prompt = re.sub(r'\s+', ' ', that.prompt).strip()
        return f'{prompt} :{that.weight}'

    def visit_composite_prompt(self, that: neutral_prompt_parser.CompositePrompt) -> str:
        return ' AND '.join(child.accept(self) for child in that.children)
