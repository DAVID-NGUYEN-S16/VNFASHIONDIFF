import torch
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm
class PiplineVNFASHION:
    def __init__(self, vae, text_encoder, tokenizer, unet, process_diffusion, max_length = 77, dtype = torch.float32):
        super().__init__()
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.unet = unet
        self.scheduler = process_diffusion
        self.max_length = max_length
        self.dtype = dtype
    def encoder_prompt(self, text):
        if isinstance(text, str):
            text = [text]
        text_tokenize = self.tokenizer(
            text, max_length = self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        prompt_embeds = self.text_encoder(
                    text_tokenize['input_ids'].to(self.device).squeeze(1), attention_mask=text_tokenize['attention_mask'].to(self.device).squeeze(1), output_hidden_states=True
                )
        return prompt_embeds
        
        
    @torch.no_grad()
    def __call__(self, prompt, num_inference_steps = 50, height = 128, width = 128, generator = None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # encoder promte
        prompt_encoder = self.text_encoder(prompt)
        
        # prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        # Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = randn_tensor((1, num_channels_latents, height, width), generator=generator, device=device, dtype=self.dtype)
        
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        
        # denoising 
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        # for iter in tqdm(range(num_inference_steps)):
        #    for i, t in enumerate(timesteps):
                
            