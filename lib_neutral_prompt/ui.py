from lib_neutral_prompt import global_state, neutral_prompt_parser
from modules import script_callbacks, shared
from typing import Dict, Tuple, List, Callable
from itertools import product
import gradio as gr
import dataclasses


txt2img_prompt_textbox = None
img2img_prompt_textbox = None


prompt_types = {
    'Perpendicular': neutral_prompt_parser.PromptKeyword.AND_PERP.value,
    'Saliency-aware': neutral_prompt_parser.PromptKeyword.AND_SALT.value,
    'Semantic guidance top-k': neutral_prompt_parser.PromptKeyword.AND_TOPK.value,
}
prompt_types_tooltip = '\n'.join([
    'AND - add all prompt features equally (webui builtin)',
    'Perpendicular - reduce the impact of contradicting prompt features',
    'Saliency-aware - strongest prompt features win',
    'Semantic guidance top-k - small targeted changes',
    'AND_ALIGN_D_S - blend new details with resolution DxD, preserving structure with resolution SxS',
])

for (i, j) in ( (i, j) for (i, j) in product(range(2, 33), repeat=2) if i != j ):
    prompt_key = f'Local alignment blend with detail kernel size {i} and structure kernel size {j}'
    prompt_types[prompt_key] = getattr(neutral_prompt_parser.PromptKeyword, f'AND_ALIGN_{i}_{j}').value

@dataclasses.dataclass
class AccordionInterface:
    get_elem_id: Callable

    def __post_init__(self):
        self.is_rendered = False

        self.cfg_rescale = gr.Slider(label='CFG rescale', minimum=0, maximum=1, value=0)
        self.neutral_prompt = gr.Textbox(label='Neutral prompt', show_label=False, lines=3, placeholder='Neutral prompt (click on apply below to append this to the positive prompt textbox)')
        self.neutral_cond_scale = gr.Slider(label='Prompt weight', minimum=-3, maximum=3, value=1)
        self.aux_prompt_type = gr.Dropdown(label='Prompt type', choices=list(prompt_types.keys()), value=next(iter(prompt_types.keys())), tooltip=prompt_types_tooltip, elem_id=self.get_elem_id('formatter_prompt_type'))
        self.append_to_prompt_button = gr.Button(value='Apply to prompt')

    def arrange_components(self, is_img2img: bool):
        if self.is_rendered:
            return

        with gr.Accordion(label='Neutral Prompt', open=False):
            self.cfg_rescale.render()
            with gr.Accordion(label='Prompt formatter', open=False):
                self.neutral_prompt.render()
                self.neutral_cond_scale.render()
                self.aux_prompt_type.render()
                self.append_to_prompt_button.render()

    def connect_events(self, is_img2img: bool):
        if self.is_rendered:
            return

        prompt_textbox = img2img_prompt_textbox if is_img2img else txt2img_prompt_textbox
        self.append_to_prompt_button.click(
            fn=lambda init_prompt, prompt, scale, prompt_type: (f'{init_prompt}\n{prompt_types[prompt_type]} {prompt} :{scale}', ''),
            inputs=[prompt_textbox, self.neutral_prompt, self.neutral_cond_scale, self.aux_prompt_type],
            outputs=[prompt_textbox, self.neutral_prompt]
        )

    def set_rendered(self, value: bool = True):
        self.is_rendered = value

    def get_components(self) -> Tuple[gr.components.Component]:
        return (
            self.cfg_rescale,
        )

    def get_infotext_fields(self) -> Tuple[Tuple[gr.components.Component, str]]:
        return tuple(zip(self.get_components(), (
            'CFG Rescale phi',
        )))

    def get_paste_field_names(self) -> List[str]:
        return [
            'CFG Rescale phi',
        ]

    def get_extra_generation_params(self, args: Dict) -> Dict:
        return {
            'CFG Rescale phi': args['cfg_rescale'],
        }

    def unpack_processing_args(
        self,
        cfg_rescale: float,
    ) -> Dict:
        return {
            'cfg_rescale': cfg_rescale,
        }


def on_ui_settings():
    section = ('neutral_prompt', 'Neutral Prompt')

    shared.opts.add_option('neutral_prompt_enabled', shared.OptionInfo(True, 'Enable neutral-prompt extension', section=section))
    global_state.is_enabled = shared.opts.data.get('neutral_prompt_enabled', True)

    shared.opts.add_option('neutral_prompt_verbose', shared.OptionInfo(False, 'Enable verbose debugging for neutral-prompt', section=section))
    shared.opts.onchange('neutral_prompt_verbose', update_verbose)


script_callbacks.on_ui_settings(on_ui_settings)


def update_verbose():
    global_state.verbose = shared.opts.data.get('neutral_prompt_verbose', False)


def on_after_component(component, **_kwargs):
    if getattr(component, 'elem_id', None) == 'txt2img_prompt':
        global txt2img_prompt_textbox
        txt2img_prompt_textbox = component

    if getattr(component, 'elem_id', None) == 'img2img_prompt':
        global img2img_prompt_textbox
        img2img_prompt_textbox = component


script_callbacks.on_after_component(on_after_component)
