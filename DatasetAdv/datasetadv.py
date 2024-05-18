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
import glob
class DataImageADV(Dataset):
    
    def __init__(self,
                 config
                 ):
        self.processor = BlipProcessor.from_pretrained(config.name_model)
        self.path_images = glob.glob(f"{config.path_data}*.jpg")
        
 
        
        

    def __len__(self):
        return len(self.path_images)

    def __getitem__(self, i):
        
        image = Image.open(self.path_images[i]).convert('RGB')
        
        pro_image = self.processor(image, return_tensors="pt")
        # print(example.keys())
        return {
            "image": pro_image,
            "path_image": self.path_images[i]
        }
