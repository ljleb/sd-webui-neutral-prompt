from lib_neutral_prompt import hijacker, global_state
from modules import script_callbacks, prompt_parser
import re


prompt_parser_hijacker = hijacker.ModuleHijacker.install_or_get(
    module=prompt_parser,
    hijacker_attribute='__neutral_prompt',
    register_uninstall=script_callbacks.on_script_unloaded,
)


AND_KEYWORD = 'AND'
AND_PERP_KEYWORD = 'AND_PERP'
and_perp_regex = re.compile(rf'\b({AND_KEYWORD}|{AND_PERP_KEYWORD})\b')


@prompt_parser_hijacker.hijack('get_multicond_learned_conditioning')
def get_multicond_learned_conditioning_hijack(model, prompts, steps, original_function):
    if not global_state.is_enabled:
        return original_function(model, prompts, steps)

    global_state.perp_profile.clear()
    for prompt in prompts:
        and_keywords = and_perp_regex.split(prompt)[1::2]
        global_state.perp_profile.append([AND_KEYWORD] + and_keywords)

    prompts = [and_perp_regex.sub(AND_KEYWORD, prompt) for prompt in prompts]
    prompts = [prompt.replace('\n', ' ') for prompt in prompts]
    return original_function(model, prompts, steps)
