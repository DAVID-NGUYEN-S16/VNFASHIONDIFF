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
from multilingual_clip import pt_multilingual_clip
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from safetensors.torch import load_model, save_model
from accelerate.utils import GradientAccumulationPlugin

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "0.0.0.0"
    os.environ["MASTER_PORT"] = "8080"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
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

def load_dataset(config, tokenizer, world_size, rank):
    train_dataset = DataFASSHIONDIFF(
        path_meta = config.data['train'],
        size= config.data['size'],
        interpolation="bicubic",
        flip_p=0.5, 
        tokenizer = tokenizer,
        train = True,
        max_length = config.max_length
    )
    sampler = DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=config.train_batch_size,
            sampler=DistributedSampler(train_dataset)
        )
    return train_dataloader, sampler
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

def setting_compute_model(config, model):
    # If passed along, set the training seed now.
    if config.seed is not None:
        set_seed(config.seed)
        
    # descrease memmory of GPU and speech up process trainning by cut a part  intermediate  value of progress backpropagate 
    if config.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    return model

def setting_accelerate(config, rank):
    
    logger = get_logger(__name__, log_level="INFO")

    wandb.login(key=config.wandb['key_wandb'])

    logging_dir = os.path.join(config.output_dir, config.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=config.output_dir, logging_dir=logging_dir)
    
    # https://huggingface.co/docs/accelerate/concept_guides/gradient_synchronization
    # When accumulation gradient update it out of memory
    gradient_accumulation_plugin = GradientAccumulationPlugin(
        num_steps=config.gradient_accumulation_steps,
        sync_each_batch = True
        )

    accelerator = Accelerator(
        log_with=config.report_to,
        gradient_accumulation_plugin=gradient_accumulation_plugin,
        
        mixed_precision=config.mixed_precision,
        device_placement=True, 
        project_config=accelerator_project_config,
    )
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    accelerator.state.device = device
    
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

    

    # Handle the repository creation
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
    return accelerator

def train(gpu_id, world_size, rank):
    path_config  = "./config.yaml"
    
    config = load_config(path_config)
    
    accelerator = setting_accelerate(config=config, rank=gpu_id)

    model, tokenizer = load_models(config)
    if config.path_fineturn_model:
        print(f"Update weight {config.path_fineturn_model}")
        load_model(model, f"{config.path_fineturn_model}/model.safetensors")
        # accelerator.load_state(config.path_fineturn_model)
        print('oke')

        # Clean memory
        torch.cuda.empty_cache()
        gc.collect()
    
    model.to(gpu_id)
    model = DDP(model, device_ids=[gpu_id])
    model = setting_compute_model(config=config, model=model)
    
    optimizer_cls = setting_optimizer(config=config)
    optimizer = optimizer_cls(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )

    
    train_dataloader, sampler = load_dataset(config=config, tokenizer= tokenizer, world_size = world_size, rank = rank)
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
    
    
    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    
    
    

    weight_dtype = torch.float32

    # Move text_encode and vae to gpu and cast to weight_dtype
    model.module.text_encoder.to(gpu_id, dtype=weight_dtype)
    model.module.vae.to(gpu_id, dtype=weight_dtype)
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
    
    for epoch in range(first_epoch, config.num_train_epochs):
        sampler.set_epoch(epoch)
        
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            torch.cuda.empty_cache()
            with accelerator.accumulate(model):
                # Convert images to latent space
                batch["pixel_values"] =batch["pixel_values"].to(gpu_id).to(weight_dtype)
                batch["input_ids"] =batch["input_ids"].to(gpu_id).to(weight_dtype).long()
                batch["attention_mask"] =batch["attention_mask"].to(gpu_id).to(weight_dtype).long()
                
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
            # if step == 1: break


        accelerator.wait_for_everyone() 
        
        train_loss = round(train_loss/len(train_dataloader), 4)

        
        
        if min_loss == None or train_loss <= min_loss and gpu_id == 0:
            save_path = os.path.join(config.output_dir, f"best")
            accelerator.save_state(save_path)
            min_loss = train_loss
            print("Save model")
            image_eval = []
            accelerator.unwrap_model(model).set_up(gpu_id)
            images, caption = accelerator.unwrap_model(model).inference()
            for img, cap in zip(images, caption):
                image = wandb.Image(img, caption=cap)
                image_eval.append(image)
            accelerator.log({"images": image_eval})
        if gpu_id == 0:
            accelerator.log(
                {
                    
                    'Train loss': train_loss, 
                    # 'Test loss': test_loss
                },
                step=global_step
            )
            print({
                    'epoch':epoch, 
                    'Train loss': train_loss, 
                })

        train_loss = 0.0
    accelerator.end_training()

def process(rank, world_size):
    ddp_setup(rank, world_size)
    train(rank, rank, world_size)
    destroy_process_group()
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f" Using {world_size} GPUs")
    mp.spawn(process, args=(world_size, ), nprocs=world_size, join=True)

