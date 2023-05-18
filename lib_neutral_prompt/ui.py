from lib_neutral_prompt import global_state, prompt_parser
from modules import script_callbacks
import gradio as gr
import dataclasses


txt2img_prompt_textbox = None
img2img_prompt_textbox = None


@dataclasses.dataclass
class GradioUserInterface:
    is_img2img: bool

    def __post_init__(self):
        self.enabled = gr.Checkbox(label='Enable', value=False)
        self.cfg_rescale = gr.Slider(label='CFG rescale', minimum=0, maximum=1, value=0)
        self.neutral_prompt = gr.Textbox(label='Neutral prompt', show_label=False, lines=3, placeholder='Neutral prompt (click on apply below to append this to the positive prompt textbox)')
        self.neutral_cond_scale = gr.Slider(label='Neutral CFG', minimum=-3, maximum=3, value=-1)
        self.append_to_prompt_button = gr.Button(value='Apply to prompt')

    def arrange_components(self):
        with gr.Accordion(label='Neutral Prompt', open=False):
            self.enabled.render()
            self.cfg_rescale.render()

            with gr.Accordion(label='Prompt formatter', open=False):
                self.neutral_prompt.render()
                self.neutral_cond_scale.render()
                self.append_to_prompt_button.render()

    def connect_events(self):
        prompt_textbox = img2img_prompt_textbox if self.is_img2img else txt2img_prompt_textbox
        self.append_to_prompt_button.click(
            fn=lambda init_prompt, prompt, scale: (f'{init_prompt} {prompt_parser.AND_PERP_KEYWORD} {prompt} :{scale}', ''),
            inputs=[prompt_textbox, self.neutral_prompt, self.neutral_cond_scale],
            outputs=[prompt_textbox, self.neutral_prompt]
        )

    def get_processing_components(self):
        return (
            self.enabled,
            self.cfg_rescale,
        )

    def update_global_state(self, enabled, cfg_rescale):
        global_state.is_enabled = enabled
        global_state.cfg_rescale = cfg_rescale


def on_after_component(component, **_kwargs):
    if getattr(component, 'elem_id', None) == 'txt2img_prompt':
        global txt2img_prompt_textbox
        txt2img_prompt_textbox = component

    if getattr(component, 'elem_id', None) == 'img2img_prompt':
        global img2img_prompt_textbox
        img2img_prompt_textbox = component


script_callbacks.on_after_component(on_after_component)
