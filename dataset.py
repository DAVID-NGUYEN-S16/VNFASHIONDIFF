import glob
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
import numpy as np
import json
from utils import check_path_image
class ImageDataset(Dataset):
    def __init__(self, real_dir, transform=None):
        """
        Args:
            real_dir (string): Directory with all the real images.
            gen_dir (string): Directory with all the generated images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dir = real_dir
        
        self.images = [os.path.join(real_dir, file) for file in glob.glob(f"{real_dir}/*.jpg")]
        
        self.images = [img for img in self.images if check_path_image(img) == True]
        print(len(self.images))
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def __len__(self):
        # The dataset length is the combined length of real and generated images
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        image = self.transform(image)

        return image
    
class DataFASSHIONDIFF(Dataset):
    
    def __init__(self,
                 path_meta,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5, 
                 tokenizer = None, 
                 train = False,
                 max_length = 77
                 ):
#         super(DataFASSHIONDIFF, self).__init__()
        with open(path_meta, 'r') as file:
            self.meta = json.load(file)
        
        self._length = len(self.meta['image'])
        print(path_meta)
        print(self._length)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.size = size
 
        
        if train:
            # Define the training data augmentation pipeline

            self.transform = transforms.Compose(
                [
                    transforms.Resize((self.size, self.size), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.RandomHorizontalFlip(p= flip_p),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

        else:
            # Define the validation and testing pipeline
            self.transform = transforms.Compose(
                [
                    transforms.Resize((self.size, self.size), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        path_image = self.meta['image'][i]

            
        text = self.meta['text'][i]
        
        image = Image.open(path_image)
        
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        # img = np.array(image).astype(np.uint8)
        # crop = min(img.shape[0], img.shape[1])
        # h, w, = img.shape[0], img.shape[1]
        # img = img[(h - crop) // 2:(h + crop) // 2,
        #       (w - crop) // 2:(w + crop) // 2]

        # image = Image.fromarray(img)
        
        
        
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
