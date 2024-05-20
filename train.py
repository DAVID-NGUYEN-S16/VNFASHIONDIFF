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

from dataset import DataFASSHIONDIFF
import time
from accelerate import notebook_launcher
import torch.multiprocessing as mp
import gc
import wandb
from accelerate.utils import GradientAccumulationPlugin
from multilingual_clip import pt_multilingual_clip
from safetensors.torch import load_model, save_model

def load_models(config):
        # Load scheduler, tokenizer and models.
        noise_scheduler = DDPMScheduler.from_pretrained(config.pretrained_model_name_or_path, subfolder="scheduler")
        # tokenizer = CLIPTokenizer.from_pretrained(
        #     config.pretrained_model_name_or_path, subfolder="tokenizer", revision=config.revision
        # )

        tokenizer = transformers.AutoTokenizer.from_pretrained('M-CLIP/XLM-Roberta-Large-Vit-L-14')

        
        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            # text_encoder = CLIPTextModel.from_pretrained(
            #     config.pretrained_model_name_or_path, subfolder="text_encoder", revision=config.revision, variant=config.variant
            # )
            model_encoderx = pt_multilingual_clip.MultilingualCLIP.from_pretrained('M-CLIP/XLM-Roberta-Large-Vit-L-14')
            
            text_encoder = VNCLIPEncoder(model_encoderx, load_config("./config_clip.yaml"))
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
        
        del model_encoderx, unet, noise_scheduler, text_encoder
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

    
    gradient_accumulation_plugin = GradientAccumulationPlugin(
        num_steps=config.gradient_accumulation_steps,
        sync_each_batch = True
        )
    accelerator = Accelerator(
        
        log_with=config.report_to,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        gradient_accumulation_plugin=gradient_accumulation_plugin,
        project_config=accelerator_project_config,
    )
    
    accelerator.init_trackers(
        project_name = config.wandb['project'],
        init_kwargs={"wandb": {"entity": "davidnguyen", 'tags': config.wandb['tags'], 'name': config.wandb['name']}}
        
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

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

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    
    if config.max_train_steps is None:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
    )
    
    # load_model(model, f"{config.path_fineturn_model}/model.safetensors")

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    if config.path_fineturn_model:
        print(f"Update weight {config.path_fineturn_model}")
        accelerator.load_state(config.path_fineturn_model)
        # Clean memory
        torch.cuda.empty_cache()
        gc.collect()

    weight_dtype = torch.float32

    # Move text_encode and vae to gpu and cast to weight_dtype
    accelerator.unwrap_model(model).text_encoder.to(accelerator.device, dtype=weight_dtype)
    accelerator.unwrap_model(model).vae.to(accelerator.device, dtype=weight_dtype)
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    
    if overrode_max_train_steps:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
        
    # Afterwards we recalculate our number of training epochs
    config.num_train_epochs = math.ceil(config.max_train_steps / num_update_steps_per_epoch)

    print("Running training")

    global_step = 0
    first_epoch = 0

    initial_global_step = 0
    
    progress_bar = tqdm(
        range(0, config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    
    
    
    min_loss = None
    start_time = time.time()

    for epoch in range(first_epoch, config.num_train_epochs):
        end_time = time.time()
        training_time = end_time - start_time
        max_training_time = 11 * 3600  
        if training_time > max_training_time:
            print("Training time exceeded 11 hours. Stopping training...")
                
        
        
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
                lr_scheduler.step()
                optimizer.zero_grad()

            logs = {"step": f",{step}/{len(train_dataloader)}", "step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

           
            global_step+=1

        accelerator.wait_for_everyone() 
        
        train_loss = round(train_loss/len(train_dataloader), 4)

        accelerator.log(
            {
                
                'Train loss': train_loss, 
                # 'Test loss': test_loss
            },
            step=global_step
        )
        
        if min_loss == None or train_loss <= min_loss:
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
    main()
