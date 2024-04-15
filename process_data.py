
from tqdm.notebook import tqdm
import re
import os
from PIL import Image
import pandas as pd
import json
def write_json(path, data):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)  # indent cho mục đích định dạng để dễ đọc hơn

def preprocess_text(text: str) -> str:    
    text = re.sub(r"['\",\.\?:\-!]", "", text)
    text = text.strip()
    text = " ".join(text.split())
    text = text.lower() 
    return text
def proces_data(list_path_datasets):
    data = {"image": [], "text": []}

    for (path_dataset, name_dataset) in tqdm(list_path_datasets):

        meta = pd.read_csv(f"{path_dataset}/{name_dataset}.csv")
        if name_dataset == "GLAMI-1M":
            col_image = 'image_file'
            folder_img = 'images'
        else:
            col_image = 'img_list'
            folder_img = name_dataset.split('_')[0]
        for i in tqdm(range(len(meta))):
        
            if name_dataset == "GLAMI-1M":
                image_paths = [meta[col_image][i]]
                text = meta.name[i] + " " +  meta.description[i]
                
            
            else:
                text = ""
                for col in ['title', 'description', 'color']:
                    if isinstance(meta[col][i], str):
                        text += " " + meta[col][i]
                image_paths = meta[col_image][i].replace("[", "").replace("]", "").replace("'", "").split(", ")
                
            text = preprocess_text(text)
            
            for path in image_paths:
                name_img = os.path.basename(path)
                path = f"{path_dataset}/{folder_img}/{name_img}"

                try:
                    image = Image.open(path)
                except:
                    print(path)
                    continue
                if os.path.exists(path) == False:
                    print(meta[col_image][i])
                    print(f"Not found image with path: {path}")
                    continue
                data["image"].append(path)
                data["text"].append(text)
    return data
data = proces_data([
#     ("/kaggle/input/cv-ck-dataset", "GLAMI-1M"), 
#     ("/kaggle/input/cv-ck-dataset-crawl/icondenim_final", "icondenim_final"), 
    ("/kaggle/input/cv-ck-dataset-crawl/uniqlo_final", "uniqlo_final"), 
#     ("/kaggle/input/cv-ck-dataset-crawl/yame_final", "yame_final"), 
])
write_json("/kaggle/working/data.json", data)