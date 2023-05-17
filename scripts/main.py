from modules import script_callbacks, sd_samplers_kdiffusion, shared


def combine_denoise_hijack(self, x_out, conds_list, uncond, cond_scale):
    if not shared.opts.data.get('prep_neg_enabled', True):
        return original_combine_denoise(self, x_out, conds_list, uncond, cond_scale)

    raise NotImplementedError


original_combine_denoise = sd_samplers_kdiffusion.CFGDenoiser.combine_denoised
sd_samplers_kdiffusion.CFGDenoiser.combine_denoised = combine_denoise_hijack


def on_ui_settings():
    section = ('prep-neg', 'Prep-Neg')
    shared.opts.add_option('prep_neg_enabled', shared.OptionInfo(True, 'Enabled', section=section))


script_callbacks.on_ui_settings(on_ui_settings)
