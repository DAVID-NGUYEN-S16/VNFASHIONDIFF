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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class DataImageADV(Dataset):
    
    def __init__(self,
                 config
                 ):
        self.tokenizer = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-en2vi-v2")

        # self.path_images = glob.glob(f"{config.path_data}*.jpg")
 
        with open(config.path_json, "r") as file:
            self.path_images = json.load(file)
        self.path_images = self.path_images['image']
        self.path_images = [f for f in self.path_images if "cv-ck-dataset" in f]
        

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
