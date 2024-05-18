import torch
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
class OutputVNFASHION:
    def __init__(self, images):
        self.images = images
class PIPELINE_VNFASHION:
    def __init__(self, 
                 vae, 
                 text_encoder, 
                 tokenizer, 
                 unet, 
                 scheduler, 
                 max_length = 77, 
                 dtype = torch.float32, 
                 device = 'cuda', 
                 guidance_scale = 7.5,
                 generator  = None, use_attention_mask = False):
        super().__init__()
        self.vae = vae.to(device)
        self.text_encoder = text_encoder.to(device)
        self.tokenizer = tokenizer
        self.unet = unet.to(device)
        self.scheduler = scheduler
        self.max_length = max_length
        self.dtype = dtype
        self.vae_scale_factor = 2**(len(vae.config.block_out_channels) - 1)
        self.generator  = generator 
        self.guidance_scale = guidance_scale
        self.device = device
        self.use_attention_mask= use_attention_mask
    def encoder_prompt(self, text):
        if isinstance(text, str):
            text = [text]
        
        text_tokenize = self.tokenizer(
            text, max_length = self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        uncond_input = self.tokenizer(
            [""] , max_length = self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        if self.use_attention_mask :
            with torch.no_grad():
                prompt_embeds = self.text_encoder(
                            text_tokenize['input_ids'].to(self.device), attention_mask=text_tokenize['attention_mask'].to(self.device), output_hidden_states=True
                        )[0]
                uncond_embeddings = self.text_encoder(
                            uncond_input['input_ids'].to(self.device), attention_mask=uncond_input['attention_mask'].to(self.device), output_hidden_states=True
                        )[0]
        else:
            with torch.no_grad():
                prompt_embeds = self.text_encoder(
                            text_tokenize['input_ids'].to(self.device), output_hidden_states=True
                        )[0]
                uncond_embeddings = self.text_encoder(
                            uncond_input['input_ids'].to(self.device), output_hidden_states=True
                        )[0]
        text_embeddings = torch.cat([uncond_embeddings, prompt_embeds]).to(self.device)

        return text_embeddings
        
        
    @torch.no_grad()
    def __call__(self, prompt, num_inference_steps = 50, height = 128, width = 128):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # encoder promte
        prompt_encoder = self.encoder_prompt(prompt)

        # Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        print(num_channels_latents)
        latents = randn_tensor(
            (1, num_channels_latents, int(height//self.vae_scale_factor), int(width//self.vae_scale_factor)),
            generator=self.generator,
            device=self.device,
            dtype = torch.float32
        )
        
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        
        # denoising 
        self.scheduler.set_timesteps(num_inference_steps)
        
        for t in tqdm(self.scheduler.timesteps):
            # using classifier-free guidance 
            latent_sub = torch.cat([latents]*2)
            latent_sub =  self.scheduler.scale_model_input(latent_sub, t)
            # precdict noise
            latent_sub = latent_sub.to(self.device)
            with torch.no_grad():
                noise_pred = self.unet(latent_sub, t, prompt_encoder)[0]
                
            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            
            noise = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
 
            noise = noise.to(self.device)
            latents = latents.to(self.device)
         
            latents = self.scheduler.step(noise, t, latents).prev_sample
            
        latents = latents / self.vae.config.scaling_factor
        images = []
        with torch.no_grad():
            image = self.vae.decode(latents).sample[0]
        image = (image/2 + 0.5).clamp(0, 1).squeeze()
        
        image = (image.permute(1, 2, 0)*255).to(torch.uint8).cpu().numpy()
        image = Image.fromarray(image)
        images.append(image)
        return OutputVNFASHION(images= images)