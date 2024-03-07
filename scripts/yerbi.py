from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from modules import scripts, script_callbacks
import gradio as gr
import torch

CONFIG_PATH = Path(__file__).parent.resolve() / '../config.yaml'


class AdaptiveScaler(torch.nn.Module):
    def __init__(self, upscale_model):
        super().__init__()
        self.upscale_model = upscale_model
        
    def forward(self, x, *args):
        x = self.upscale_model(x)
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
        
        self.infotext_fields.clear()  # Reset infotext fields to avoid duplicates
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
        upscale_model = self.select_upscale_model(model_choice)
        
        def denoiser_callback(params: script_callbacks.CFGDenoiserParams):
            for i, block in enumerate(model.input_blocks + model.output_blocks):
                if isinstance(block, AdaptiveScaler):
                    continue  # Already replaced
                model.input_blocks[i] = AdaptiveScaler(upscale_model)
                
            if quality_enhancements:
                # Apply post-processing enhancements here
                pass

        script_callbacks.on_cfg_denoiser(denoiser_callback)

    def select_upscale_model(self, model_choice):
        # Placeholder for selecting and initializing the actual upscaling model based on user choice
        return torch.nn.Identity()  # Placeholder implementation

    def postprocess(self, p, processed, *args):
        # Reset any modifications made to the model
        OmegaConf.save(self.config, CONFIG_PATH)
