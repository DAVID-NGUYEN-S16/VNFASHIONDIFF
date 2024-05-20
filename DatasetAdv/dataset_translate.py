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
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/vinai-translate-en2vi-v2", src_lang="en_XX")

 
        with open(config.path_json, "r") as file:
            self.path_images = json.load(file)
        

    def __len__(self):
        return len(self.path_images['image'])

    def __getitem__(self, i):
        
        input_ids = self.tokenizer(self.path_images['text'][i], padding=True, return_tensors="pt")


        return {
            "image": self.path_images['image'][i],
            "text": input_ids
        }
