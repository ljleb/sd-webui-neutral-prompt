from lib_neutral_prompt import hijacker, global_state
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

    global_state.perp_profile.clear()
    for prompt in prompts:
        global_state.perp_profile.append([PromptKeyword.AND] + [PromptKeyword[v] for v in and_perp_regex.split(prompt)[1::2]])

    prompts = [and_perp_regex.sub(PromptKeyword.AND.value, prompt) for prompt in prompts]
    prompts = [prompt.replace('\n', ' ') for prompt in prompts]
    return original_function(model, prompts, steps)
