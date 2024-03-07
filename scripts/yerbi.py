import torch
import torch.nn as nn
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from modules import scripts, script_callbacks
import gradio as gr

CONFIG_PATH = Path(__file__).parent.resolve() / '../config.yaml'

class AdaptiveScaler(nn.Module):
    def __init__(self, upscale_model, output_channels):
        super().__init__()
        self.upscale_model = upscale_model
        # Ensuring the output has the correct number of channels
        self.adjust_channels = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=1)

    def forward(self, x):
        x = self.upscale_model(x)
        x = self.adjust_channels(x)
        return x

class Yerbi(scripts.Script):
    def __init__(self):
        super().__init__()
        try:
            self.config: DictConfig = OmegaConf.load(CONFIG_PATH)
        except Exception:
            self.config = DictConfig({})
        self.disable = False
        self.step_limit = 0
        self.infotext_fields = []

    def title(self):
        return "Yerbi Upscaler"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(label='Yerbi Upscaler', open=False):
            enable = gr.Checkbox(label='Enable Yerbi', value=False)
            model_choice = gr.Dropdown(['ESRGAN', 'Bicubic', 'Bilinear', 'Nearest'], label='Upscaling Model', value='ESRGAN')
            quality_enhancements = gr.Checkbox(label="Apply Quality Enhancements", value=True)

        ui = [enable, model_choice, quality_enhancements]
        for elem in ui:
            setattr(elem, "do_not_save_to_config", True)

        parameters = {
            'Yerbi_model': model_choice,
            'Yerbi_quality': quality_enhancements,
        }

        self.infotext_fields.clear()
        self.infotext_fields.append((enable, lambda d: d.get('Yerbi_model', False)))
        for k, element in parameters.items():
            self.infotext_fields.append((element, k))

        return ui

    def process(self, p, enable, model_choice, quality_enhancements):
        self.config = DictConfig({name: var for name, var in locals().items() if name not in ['self', 'p']})
        if not enable or self.disable:
            script_callbacks.remove_current_script_callbacks()
            return
        model = p.sd_model.model.diffusion_model
        # Assuming the model's input_blocks and output_blocks require 32 channels
        required_output_channels = 32
        upscale_model = self.select_upscale_model(model_choice, required_output_channels)

        def denoiser_callback(params: script_callbacks.CFGDenoiserParams):
            for i, block in enumerate(model.input_blocks + model.output_blocks):
                if not isinstance(block, AdaptiveScaler):
                    model.input_blocks[i] = AdaptiveScaler(upscale_model, required_output_channels)
                
            if quality_enhancements:
                # Apply post-processing enhancements here
                pass

        script_callbacks.on_cfg_denoiser(denoiser_callback)

    def select_upscale_model(self, model_choice, output_channels):
        # This method needs to instantiate and return the correct upscaling model based on the choice.
        # The output_channels parameter helps ensure compatibility with the rest of the model.
        # For simplicity, this is a placeholder. You need to replace it with actual model selection logic.
        return torch.nn.Identity()  # Placeholder implementation

    def postprocess(self, p, processed, *args):
        # Reset any modifications made to the model
        OmegaConf.save(self.config, CONFIG_PATH)
