from typing import List

from lib_neutral_prompt import hijacker, global_state, neutral_prompt_parser
from modules import script_callbacks, prompt_parser


prompt_parser_hijacker = hijacker.ModuleHijacker.install_or_get(
    module=prompt_parser,
    hijacker_attribute='__neutral_prompt_hijacker',
    on_uninstall=script_callbacks.on_script_unloaded,
)


@prompt_parser_hijacker.hijack('get_multicond_prompt_list')
def get_multicond_prompt_list_hijack(prompts, original_function):
    if not global_state.is_enabled:
        return original_function(prompts)

    global_state.prompt_exprs = parse_prompts(prompts)
    webui_prompts = transpile_exprs(global_state.prompt_exprs)
    if isinstance(prompts, getattr(prompt_parser, 'SdConditioning', type(None))):
        webui_prompts = prompt_parser.SdConditioning(webui_prompts, copy_from=prompts)

    return original_function(webui_prompts)


def parse_prompts(prompts: List[str]) -> List[neutral_prompt_parser.PromptExpr]:
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
        return f'{that.prompt} :{that.weight}'

    def visit_composite_prompt(self, that: neutral_prompt_parser.CompositePrompt) -> str:
        return ' AND '.join(child.accept(self) for child in that.children)


@prompt_parser_hijacker.hijack("reconstruct_multicond_batch")
def reconstruct_multicond_batch_hijack(*args, original_function, **kwargs):
    res = original_function(*args, **kwargs)
    global_state.batch_cond_indices = res[0]
    return res
