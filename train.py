import logging
import math
import os
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from transformers.utils import ContextManagers
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from tqdm import tqdm
from utils import load_config, deepspeed_zero_init_disabled_context_manager
from models.ldm import LatenFashionDIFF
from models.clip import VNCLIPEncoder
from transformers import CLIPTextModel, CLIPTokenizer

from dataset import DataFASSHIONDIFF
import time
from accelerate import notebook_launcher
import torch.multiprocessing as mp
import gc
import wandb
from accelerate.utils import GradientAccumulationPlugin
from multilingual_clip import pt_multilingual_clip
from safetensors.torch import load_model, save_model
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    
def load_models(config):
        # Diffusion process
        noise_scheduler = DDPMScheduler.from_pretrained(config.pretrained_model_name_or_path, subfolder="scheduler")
 
        # tokenizer = transformers.AutoTokenizer.from_pretrained('M-CLIP/XLM-Roberta-Large-Vit-L-14')
        tokenizer = CLIPTokenizer.from_pretrained(
            config.pretrained_model_name_or_path, subfolder="tokenizer", revision=config.revision
        )
        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            # model_encoderx = pt_multilingual_clip.MultilingualCLIP.from_pretrained('M-CLIP/XLM-Roberta-Large-Vit-L-14')
            
            # text_encoder = VNCLIPEncoder(model_encoderx, load_config("./config_clip.yaml"))
            tokenizer = CLIPTokenizer.from_pretrained(
                config.pretrained_model_name_or_path, subfolder="tokenizer", revision=config.revision
            )
            text_encoder = CLIPTextModel.from_pretrained(
                config.pretrained_model_name_or_path, subfolder="text_encoder", revision=config.revision, variant=config.variant
            )
            vae = AutoencoderKL.from_pretrained(
                config.pretrained_model_name_or_path, subfolder="vae", revision=config.revision, variant=config.variant
            )

        unet = UNet2DConditionModel.from_pretrained(
            config.pretrained_model_name_or_path, subfolder="unet", revision=config.non_ema_revision
        )
        
        # Đóng băng các thông số của text_encoder
        for param in text_encoder.parameters():
            param.requires_grad = False

        # Đóng băng các thông số của vae
        for param in vae.parameters():
            param.requires_grad = False

        
        #text_encoder, vae, unet, process_diffusion, scaling_factor
        model = LatenFashionDIFF(
            text_encoder = text_encoder,
            vae = vae, 
            unet = unet, 
            process_diffusion = noise_scheduler, 
            tokenizer = tokenizer,
            use_attention_mask = config.use_attention_mask,
            max_length = config.max_length
        )
        count_parameters(model)
        del text_encoder, unet, noise_scheduler, text_encoder
        torch.cuda.empty_cache()
        gc.collect()
        return model, tokenizer
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    attention_mask = torch.stack([example["attention_mask"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids, 'attention_mask': attention_mask}

def setting_optimizer(config):
    # Initialize the optimizer
    if config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        # https://huggingface.co/docs/bitsandbytes/main/en/optimizers

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    return optimizer_cls
def main():
    
    logger = get_logger(__name__, log_level="INFO")
    
    ## config global
    path_config  = "./config.yaml"
    
    config = load_config(path_config)

    wandb.login(key=config.wandb['key_wandb'])

    logging_dir = os.path.join(config.output_dir, config.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=config.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        
        log_with=config.report_to,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        # gradient_accumulation_plugin=gradient_accumulation_plugin,
        project_config=accelerator_project_config,
    )
    
    accelerator.init_trackers(
        project_name = config.wandb['project'],
        init_kwargs={"wandb": {"entity": "davidnguyen", 'tags': config.wandb['tags'], 'name': config.wandb['name']}}
        
    )
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # If passed along, set the training seed now.
    if config.seed is not None:
        set_seed(config.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)


    model, tokenizer = load_models(config)
    

    # descrease memmory of GPU and speech up process trainning by cut a part  intermediate  value of progress backpropagate 
    if config.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    optimizer_cls = setting_optimizer(config=config)
    
    optimizer = optimizer_cls(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )

    train_dataset = DataFASSHIONDIFF(
        path_meta = config.data['train'],
        size= config.data['size'],
        interpolation="bicubic",
        flip_p=0.5, 
        tokenizer = tokenizer,
        train = True,
        max_length = config.max_length
    )
    
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=config.train_batch_size,
        )
    
    lambda1 = lambda epoch: (1/math.sqrt(epoch + 1)) 
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda1)
    
    if config.path_fineturn_model:
        print(f"Update weight {config.path_fineturn_model}")
        accelerator.load_state(config.path_fineturn_model)
        # load_model(model, f"{config.path_fineturn_model}/model.safetensors")
        # Clean memory
        torch.cuda.empty_cache()
        gc.collect()
    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    weight_dtype = torch.float32


    print("Running training")
    global_step = 0
    first_epoch = 0

    initial_global_step = 0
    
    progress_bar = tqdm(
        # range(0, config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    min_loss = None
    if config.loss_previous:
        min_loss = config.loss_previous
    for epoch in range(config.num_train_epochs):

        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
     
            with accelerator.accumulate(model):
                
                # Convert images to latent space
                batch["pixel_values"] =batch["pixel_values"].to(accelerator.device).to(weight_dtype)
                batch["input_ids"] =batch["input_ids"].to(accelerator.device).to(weight_dtype).long()
                batch["attention_mask"] =batch["attention_mask"].to(accelerator.device).to(weight_dtype).long()
                
                # Predict the noise residual and compute loss
                target, model_pred = model(
                    pixel_values = batch["pixel_values"], 
                    input_ids = batch["input_ids"], 
                    attention_mask = batch['attention_mask'])

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
 

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(config.train_batch_size)).mean()
                train_loss += avg_loss.item() / config.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                
                optimizer.zero_grad()
                
            
            logs = {"step": f",{step}/{len(train_dataloader)}", "step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

           
            global_step+=1
            # if step == 1: break
        
        
        train_loss = round(train_loss/len(train_dataloader), 4)

        accelerator.log(
            {
                
                'Train loss': train_loss, 
                "lr_current": optimizer.param_groups[0]['lr']
                # 'Test loss': test_loss
            },
            step=epoch
        )
        lr_scheduler.step()
        
        if min_loss == None or train_loss <= min_loss:
            model.eval()
            save_path = os.path.join(config.output_dir, f"best")
            accelerator.save_state(save_path)
            min_loss = train_loss
            print("Save model")
            image_eval = []
            accelerator.unwrap_model(model).set_up()
            images, caption = accelerator.unwrap_model(model).inference()
            for img, cap in zip(images, caption):
                image = wandb.Image(img, caption=cap)
                image_eval.append(image)
            accelerator.log({"images": image_eval})
        
            logger.info(f"Saved state to {save_path}")
            
        print({
                'epoch':epoch, 
                'Train loss': train_loss, 
            })

        train_loss = 0.0
    accelerator.end_training()


if __name__ == "__main__":
    # notebook_launcher()
    # main()
    notebook_launcher(main, args=(), num_processes=2)

