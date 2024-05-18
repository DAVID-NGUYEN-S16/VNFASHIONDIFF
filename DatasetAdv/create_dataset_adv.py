from transformers import  BlipForConditionalGeneration
import os
import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import glob
from datasetadv import DataImageADV
from accelerate import notebook_launcher
from utils import load_config, write_json
def main():
    
    
    ## config global
    path_config  = "./config_caption_nike.yaml"
    
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
    
    test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=True,
            batch_size=config.batch_size,
    )
    


    # Prepare everything with our `accelerator`.
    model = accelerator.prepare(
        model
    )
    
 
    weight_dtype = torch.float32

 
    data = {'image': [], "text": []}
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            with accelerator.accumulate(model):
                # Convert images to latent space
                inputs = batch['image'].to(accelerator.device).to(weight_dtype)
                
                outs = accelerator.unwrap_model(model).generate(**inputs)
                texts = test_dataset.processor.batch_decode(outs, skip_special_tokens=True)
                data['image'] += batch['path_image']
                data['text'] += texts
                
    accelerator.wait_for_everyone() 
    
    write_json(f"{config.name_data}.json", data)
        
    accelerator.end_training()


if __name__ == "__main__":
    
    # paths = glob.glob("./DatasetAdv/*.yaml")
    # for path in paths:
    notebook_launcher(main, args=(), num_processes=2)

