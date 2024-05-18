import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import logging
import math
import os
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
import glob
from datasetadv import DataImageADV
from accelerate import notebook_launcher
from ..utils import load_config, write_json
def main():
    
    logger = get_logger(__name__, log_level="INFO")
    
    ## config global
    path_config  = "./config_caption.yaml"
    
    config = load_config(path_config)


    logging_dir = os.path.join(config.output_dir, config.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=config.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        log_with=config.report_to,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        
        project_config=accelerator_project_config,
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


    model = BlipForConditionalGeneration.from_pretrained(config.name_model).to(accelerator.device)
    

    test_dataset = DataImageADV(
        config=config
    )
    
    test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=True,
            batch_size=config.batch_size,
    )
    


    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
 
    weight_dtype = torch.float32

 
    data = {'image': [], "text": []}
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            with accelerator.accumulate(model):
                # Convert images to latent space
                inputs = batch['image'].to(accelerator.device).to(weight_dtype)
                
                outs = model.generate(**inputs)
                texts = test_dataset.processor.batch_decode(outs, skip_special_tokens=True)
                data['image'] += batch['path_image']
                data['text'] += texts
                
    accelerator.wait_for_everyone() 
    
    write_json(f"{config.name_data}.json", data)
        
    accelerator.end_training()


if __name__ == "__main__":
    
    paths = glob.glob("./DatasetAdv/*.yaml")
    notebook_launcher(main, args=(), num_processes=1)

