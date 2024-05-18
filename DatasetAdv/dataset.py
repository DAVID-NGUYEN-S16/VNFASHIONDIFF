import glob
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset, DataLoader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
import numpy as np
import json
from transformers import BlipProcessor, BlipForConditionalGeneration

class DataFASSHIONDIFF(Dataset):
    
    def __init__(self,
                 config
                 ):
        self.processor = BlipProcessor.from_pretrained(config.name_model)
        self.path_images
        
 
        
        

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        path_image = self.meta['image'][i]

            
        text = self.meta['text'][i]
        
        image = Image.open(path_image)
        
        if not image.mode == "RGB":
            image = image.convert("RGB")


        
        example = dict()
        text_tokenize = self.tokenizer(
            text, max_length = self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        
        example['input_ids'] = text_tokenize.input_ids
        example['attention_mask'] = text_tokenize.attention_mask
        
        
        try:
            example["pixel_values"] = self.transform(image)
        except Exception as e:
            print(f"Error processing image at path: {path_image}")
            print(f"Error message: {str(e)}")
        # print(example.keys())
        return example
