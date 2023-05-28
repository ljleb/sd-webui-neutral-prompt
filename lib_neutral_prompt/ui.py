from lib_neutral_prompt import global_state, neutral_prompt_parser
from modules import script_callbacks, shared
from typing import Dict, Tuple
import gradio as gr
import dataclasses


txt2img_prompt_textbox = None
img2img_prompt_textbox = None


prompt_types = {
    'Perpendicular': neutral_prompt_parser.PromptKeyword.AND_PERP.value,
    'Saliency-aware': neutral_prompt_parser.PromptKeyword.AND_SALT.value,
}


@dataclasses.dataclass
class AccordionInterface:
    def __post_init__(self):
        self.cfg_rescale = gr.Slider(label='CFG rescale φ', minimum=0, maximum=1, value=0); self.cfg_rescale.unrender()
        self.neutral_prompt = gr.Textbox(label='Neutral prompt', show_label=False, lines=3, placeholder='Neutral prompt (click on apply below to append this to the positive prompt textbox)'); self.neutral_prompt.unrender()
        self.neutral_cond_scale = gr.Slider(label='Prompt weight', minimum=-3, maximum=3, value=1); self.neutral_cond_scale.unrender()
        self.aux_prompt_type = gr.Dropdown(label='Prompt type', choices=list(prompt_types.keys()), value=next(iter(prompt_types.keys())))
        self.append_to_prompt_button = gr.Button(value='Apply to prompt'); self.append_to_prompt_button.unrender()

    def arrange_components(self, is_img2img: bool):
        with gr.Accordion(label='Neutral Prompt', open=False):
            self.cfg_rescale.render()
            with gr.Accordion(label='Prompt formatter', open=False):
                self.neutral_prompt.render()
                self.neutral_cond_scale.render()
                self.aux_prompt_type.render()
                self.append_to_prompt_button.render()

    def connect_events(self, is_img2img: bool):
        prompt_textbox = img2img_prompt_textbox if is_img2img else txt2img_prompt_textbox
        self.append_to_prompt_button.click(
            fn=lambda init_prompt, prompt, scale, prompt_type: (f'{init_prompt}\n{prompt_types[prompt_type]} {prompt} :{scale}', ''),
            inputs=[prompt_textbox, self.neutral_prompt, self.neutral_cond_scale, self.aux_prompt_type],
            outputs=[prompt_textbox, self.neutral_prompt]
        )

    def get_components(self) -> Tuple[gr.components.Component]:
        return (
            self.cfg_rescale,
        )

    def get_infotext_fields(self) -> Tuple[Tuple[gr.components.Component, str]]:
        return tuple(zip(self.get_components(), (
            'CFG Rescale φ',
        )))

    def get_extra_generation_params(self, args: Dict) -> Dict:
        return {
            'CFG Rescale φ': args['cfg_rescale'],
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
    shared.opts.add_option('neutral_prompt_enabled', shared.OptionInfo(True, 'Enabled', section=section))


script_callbacks.on_ui_settings(on_ui_settings)


def on_after_component(component, **_kwargs):
    if getattr(component, 'elem_id', None) == 'txt2img_prompt':
        global txt2img_prompt_textbox
        txt2img_prompt_textbox = component

    if getattr(component, 'elem_id', None) == 'img2img_prompt':
        global img2img_prompt_textbox
        img2img_prompt_textbox = component


script_callbacks.on_after_component(on_after_component)
