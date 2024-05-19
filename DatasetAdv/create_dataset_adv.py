from transformers import  BlipForConditionalGeneration
import os
import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import glob
from datasetadv import DataImageADV
from accelerate import notebook_launcher
from tqdm import tqdm

from utils import load_config, write_json
def main():
    
    
    ## config global
    path_config  = "./config_caption_GLAMI.yaml"
    
    config = load_config(path_config)

    accelerator_project_config = ProjectConfiguration(project_dir=config.output_dir)

    accelerator = Accelerator(
        
        project_config=accelerator_project_config,
    )


    # Handle the repository creation
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)


    model = BlipForConditionalGeneration.from_pretrained(config.name_model).to(accelerator.device)
    

    test_dataset = DataImageADV(
        config=config
    )
    print(f"Length Dataset: {len(test_dataset)}")
    print(f"Batch size: {config.batch_size}")
    test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.batch_size,
    )
    


    # Prepare everything with our `accelerator`.
    model = accelerator.prepare(
        model
    )
    
 
    weight_dtype = torch.float32

 
    data = {'image': [], "text": []}
    model.eval()
    progress_bar = tqdm(
        range(0, len(test_dataloader)),
        initial=0,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            with accelerator.accumulate(model):
                # Convert images to latent space
                inputs = batch['image'].to(accelerator.device).to(weight_dtype)
             
                inputs['pixel_values'] = inputs['pixel_values'].squeeze(1)
                
                outs = accelerator.unwrap_model(model).generate(**inputs)
                texts = test_dataset.processor.batch_decode(outs, skip_special_tokens=True)
            data['image'] += batch['path_image']
            data['text'] += texts
            logs = {"step": f",{step}/{len(test_dataloader)}"}
            progress_bar.set_postfix(**logs)
            write_json(f"{config.name_data}.json", data)
            
    accelerator.wait_for_everyone() 
    write_json(f"{config.name_data}.json", data)
    
        
    accelerator.end_training()


if __name__ == "__main__":
    
    # paths = glob.glob("./DatasetAdv/*.yaml")
    # for path in paths:
    notebook_launcher(main, args=(), num_processes=2)

