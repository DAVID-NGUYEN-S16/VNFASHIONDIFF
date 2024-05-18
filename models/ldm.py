
import torch
import torch.utils.checkpoint
import torch.nn as nn
# from diffusers import StableDiffusionPipeline
from .inference import PIPELINE_VNFASHION
class DiffusionWrapper(nn.Module):
    def __init__(self, diffusion_model: None):
        super().__init__()
        self.diffusion_model = diffusion_model
        
    def forward(self, x: torch.Tensor, time_steps: torch.Tensor, context: torch.Tensor):
        return self.diffusion_model(x, time_steps, context, return_dict=False)
        
class LatenFashionDIFF(nn.Module):
    def __init__(self, 
                 text_encoder, 
                 vae, 
                 unet, 
                 process_diffusion, 
                 tokenizer, 
                 max_length = 77, 
                 ):
        super().__init__()
        self.model = DiffusionWrapper(unet)
        self.text_encoder = text_encoder.requires_grad_(False)
        self.vae = vae.requires_grad_(False)
        self.process_diffusion = process_diffusion
        self.scaling_factor = vae.config.scaling_factor
        self.tokenizer = tokenizer
        self.max_length = max_length
        # self.set_up()
    def forward(self, pixel_values, input_ids, attention_mask):
        
        
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.scaling_factor
        # create distribution of latens        
        noise = torch.randn_like(latents)
        
        bsz = latents.shape[0]
        
        # get time step 
        timesteps = torch.randint(0, self.process_diffusion.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        
        #get value of P(x_1|x_0)
        noisy_latents = self.process_diffusion.add_noise(latents, noise, timesteps)
        
        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(input_ids, return_dict=False)[0]
        
        
        if self.process_diffusion.config.prediction_type == "epsilon":
            target = noise
        elif self.process_diffusion.config.prediction_type == "v_prediction":
            target = self.process_diffusion.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.process_diffusion.config.prediction_type}")
        
        model_pred = self.model(x = noisy_latents, time_steps = timesteps, context = encoder_hidden_states)[0]
        
        return target, model_pred
    def set_up(self):
        self.pipeline = PIPELINE_VNFASHION(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.model.diffusion_model,
            scheduler = self.process_diffusion,
            max_length = self.max_length,
            device = self.vae.device
        )
    def inference(self, text = None):
        
        if text is None:
            text = ["Cái quần màu đỏ, cái áo khoác màu xanh", "Bộ quần áo màu nâu"]
        if isinstance(text, str):
            text =[text]
        images = []
        for t in text:
            image = self.pipeline(t, num_inference_steps=60, height=128, width=128).images[0]
            images.append(image)
        return images, text
