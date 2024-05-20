from transformers import  BlipForConditionalGeneration
import os
import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import glob
from dataset_translate import DataImageADV
from accelerate import notebook_launcher
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from utils import load_config, write_json
def main(path_config):

    
    config = load_config(path_config)

    accelerator_project_config = ProjectConfiguration(project_dir=config.output_dir)

    accelerator = Accelerator(
        
        project_config=accelerator_project_config,
    )


    # Handle the repository creation
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)


    model = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-en2vi-v2")
    

    test_dataset = DataImageADV(
        config=config
    )
    print(f"Length Dataset: {len(test_dataset)}")
    print(f"Batch size: {config.batch_size}")
    test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=1,
    )
    


    # Prepare everything with our `accelerator`.
    model = accelerator.prepare(
        model
    )
    
 
 
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
                inputs = batch['text']
                inputs['input_ids'] = inputs['input_ids'].squeeze(1).to(accelerator.device).long()
                inputs['attention_mask'] = inputs['attention_mask'].squeeze(1).to(accelerator.device).long()
                output_ids = accelerator.unwrap_model(model).generate(
                    **inputs,
                    decoder_start_token_id=test_dataset.tokenizer.lang_code_to_id["vi_VN"],
                    num_return_sequences=1,
                    num_beams=5,
                    early_stopping=True
                )
                vi_texts = test_dataset.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            data['image'] += batch['image']
            data['text'] += vi_texts
            logs = {"step": f",{step}/{len(test_dataloader)}"}
            progress_bar.set_postfix(**logs)
            write_json(f"{config.name_data}.json", data)
            
    accelerator.wait_for_everyone() 
    write_json(f"{config.name_data}_translate.json", data)
    
        
    accelerator.end_training()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Translate text')
    parser.add_argument('path_config', type=str, help='path config')
   
    args = parser.parse_args()
    notebook_launcher(main, args=(args.path_config, ), num_processes=2)

