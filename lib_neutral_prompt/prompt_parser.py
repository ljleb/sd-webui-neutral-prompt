from lib_neutral_prompt import hijacker, global_state, perp_parser
from modules import script_callbacks, prompt_parser
from enum import Enum
import re


prompt_parser_hijacker = hijacker.ModuleHijacker.install_or_get(
    module=prompt_parser,
    hijacker_attribute='__neutral_prompt_hijacker',
    on_uninstall=script_callbacks.on_script_unloaded,
)


class PromptKeyword(Enum):
    AND = 'AND'
    AND_PERP = 'AND_PERP'


and_perp_regex = re.compile(rf'\b({"|".join([e.value for e in PromptKeyword])})\b')


@prompt_parser_hijacker.hijack('get_multicond_learned_conditioning')
def get_multicond_learned_conditioning_hijack(model, prompts, steps, original_function):
    if not global_state.is_enabled:
        return original_function(model, prompts, steps)

    global_state.prompt_exprs.clear()
    webui_prompts = []
    for prompt in prompts:
        expr = perp_parser.parse_root(prompt)
        global_state.prompt_exprs.append(expr)
        webui_prompts.append(expr.accept(WebuiPromptVisitor()))

    return original_function(model, webui_prompts, steps)


class WebuiPromptVisitor:
    def visit_composable_prompt(self, that: perp_parser.ComposablePrompt) -> str:
        prompt = re.sub(r'\s+', ' ', that.prompt).strip()
        return f'{prompt} :{that.weight}'

    def visit_composite_prompt(self, that: perp_parser.CompositePrompt) -> str:
        return ' AND '.join(child.accept(self) for child in that.children)
