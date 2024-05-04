import os
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from utils import load_config
from models.ldm import LatenFashionDIFF
from dataset import DataFASSHIONDIFF
import time
from torch import autocast 
if is_wandb_available():
    import wandb

## config global
path_config  = "./config.yaml"
config = load_config(path_config)

wandb.login(key=config.wandb['key_wandb'])

run = wandb.init(
    # Set the project where this run will be logged
    project=config.wandb['project'],

    tags = config.wandb['tags'], 
    name = config.wandb['name']
)

def load_models(config):
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(config.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="tokenizer", revision=config.revision
    )

    
    
    text_encoder = CLIPTextModel.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="text_encoder", revision=config.revision, variant=config.variant
    ).eval()
    vae = AutoencoderKL.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="vae", revision=config.revision, variant=config.variant
    ).eval()

    unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="unet", revision=config.non_ema_revision
    )
    
    #text_encoder, vae, unet, process_diffusion, scaling_factor
    model = LatenFashionDIFF(
        text_encoder = text_encoder,
        vae = vae, 
        unet = unet, 
        process_diffusion = noise_scheduler, 
        tokenizer = tokenizer
    )
    return model, tokenizer

def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}
    
def main():


    # If passed along, set the training seed now.
    if config.seed is not None:
        torch.manual_seed(config.seed)

    if config.output_dir is not None:
        os.makedirs(config.output_dir, exist_ok=True)


    model, tokenizer = load_models(config)


    model.train()


    # descrease memmory of GPU and speech up process trainning by cut a part  intermediate  value of progress backpropagate 
    if config.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True


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
        train = True
    )
    
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=config.train_batch_size,
        )
    
    test_dataset = DataFASSHIONDIFF(
        path_meta = config.data['train'],
        size= config.data['size'],
        interpolation="bicubic",
        flip_p=0.5, 
        tokenizer = tokenizer,
        train = False
    )
    
    test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=config.train_batch_size,
    )
    
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=config.max_train_steps ,
    )


    weight_dtype = torch.float32
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Move text_encode and vae to gpu and cast to weight_dtype
    model.text_encoder.to(device)
    model.vae.to(device)
    model.to(device)
    global_step = 0
    first_epoch = 0  
    
    min_loss = None
    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()

    if config.path_checkpoint:
        checkpoint = torch.load(config.path_checkpoint)

        first_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        config.num_train_epochs += first_epoch
        print(f"Load model: {config.path_checkpoint}")
        
    for epoch in range(first_epoch, config.num_train_epochs):
        end_time = time.time()
        training_time = end_time - start_time
        max_training_time = 10 * 3600  
        if training_time > max_training_time:
            print("Training time exceeded 11 hours. Stopping training...")
            break
                
        
        
        model.train()
        train_loss = 0.0
        test_loss = 0.0
        step_counts = 0
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            # Convert images to latent space
            batch["pixel_values"] =batch["pixel_values"].to(device)
            batch["input_ids"] =batch["input_ids"].to(device).long()
            
                
            with autocast(device_type="cuda", dtype=weight_dtype):
                # Predict the noise residual and compute loss
                target, model_pred = model(pixel_values = batch["pixel_values"], input_ids = batch["input_ids"])

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean") / config.gradient_accumulation_steps

            train_loss += loss.item() 

            # Backpropagate
            scaler.scale(loss).backward()
            
            
            
            if (step + 1) % int(config.gradient_accumulation_steps) == 0:
                step_counts +=1
                # Make sure that parameter had return origin value gradient before apply clip_grad_norm
                scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                lr_scheduler.step()
                scaler.update()
            global_step+=1
        
        model.eval()
        
        for step, batch in enumerate(test_dataloader):
            # Convert images to latent space
            batch["pixel_values"] =batch["pixel_values"].to(device)
            batch["input_ids"] =batch["input_ids"].to(device).long()
            
            with autocast(device_type="cuda", dtype=weight_dtype):
                # Predict the noise residual and compute loss
                target, model_pred = model(pixel_values = batch["pixel_values"], input_ids = batch["input_ids"])

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            
            test_loss += loss.item() / config.gradient_accumulation_steps
    
        
        train_loss = round(train_loss/len(train_dataloader), 4)
        test_loss = round(test_loss/len(test_dataloader), 4)

        wandb.log(
            {
                
                'Train loss': train_loss, 
                'Test loss': test_loss,
                'lr': lr_scheduler.get_last_lr()[0],
                'step_counts': step_counts
            }
        )
        total_loss = train_loss*0.3 + test_loss*0.7
        if min_loss == None or total_loss <= min_loss:
            print("Update model")
            save_path = os.path.join(config.output_dir, f"best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                }, save_path)
            min_loss = total_loss
            print("Save model")
            image_eval = []
            model.set_up()
            images, caption = model.inference()
            for img, cap in zip(images, caption):
                image = wandb.Image(img, caption=cap)
                image_eval.append(image)
            wandb.log({"images": image_eval})
        
        print({
                'epoch':epoch, 
                'Train loss': train_loss, 
                'Test loss': test_loss,
                
            })

        train_loss = 0.0
        test_loss = 0.0

if __name__ == "__main__":
    main()