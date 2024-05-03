
from tqdm import tqdm


import pandas as pd
import ast
from utils import translate, write_json, check_path_image, preprocess_text
import logging
from multiprocessing import Pool
import os
logging.basicConfig(filename='process_data.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')


def process_data_glami(path):
    meta = pd.read_csv(path).reset_index()
    data = {"image": [], "text": []}
    cnt = 0
    for i in tqdm(meta.index):
        title = meta.name[i]
        description = meta.description[i]
        name_image = os.path.basename(meta.image_file[i])
        if isinstance(description, str) == False and isinstance(title, str) == False:
            continue

        # print("title", title)
        # print("description", description)
        content = ""
        if isinstance(description, str) == False or isinstance(title, str) == False:
            content = title if description is None else title
        else: content = title + " " + description

        idx = len(data['image'])
        path_img = f"./data/glami-1m_final/glami-1m/{name_image}"
        if check_path_image(path_img) == False:
            cnt +=1
            continue
        data['image'].append(path_img)
        data['text'].append(preprocess_text(content))
    logging.info("Dataset: glami")
    logging.info(f"Complete data: {path}")
    logging.info(f"{cnt} error")
    logging.info(f"{len(data['text'])} sucess")
    return data
def process_data_adidas(path):
    meta = pd.read_csv(path).reset_index()
    data = {"image": [], "text": []}
    cnt = 0
    for i in tqdm(meta.index):
        title = translate(meta.title[i])
        description = meta.decripsion[i]
        color = translate( meta.color[i])

        if isinstance(description, str) == False and isinstance(title, str) == False:
            continue

        content = ""
        if isinstance(description, str) == False or isinstance(title, str) == False:

            content = title if description is None else title
            content = content + f" với màu sắc {color} "
        else: content = title + f" với màu sắc {color} " + description

        idx = len(data['image'])
        image_lists = ast.literal_eval(meta.images[i])

        for path_img in image_lists:
            path_img = f"./data/adidas_final/adidas/{path_img}"
            if check_path_image(path_img) == False:
                cnt +=1
                continue

            data['image'].append(path_img)
            data['text'].append(preprocess_text(content))
    logging.info("Dataset: adidas")
    logging.info(f"Complete data: {path}")
    logging.info(f"{cnt} error")
    logging.info(f"{len(data['text'])} sucess")
    
    return data

def process_data_uniqlo(path):
    meta = pd.read_csv(path).reset_index()
    data = {"image": [], "text": []}
    cnt = 0
    for i in tqdm(meta.index):
        title = translate(meta.title[i])
        description = meta.description[i]

        if isinstance(description, str) == False and isinstance(title, str) == False:
            continue

        content = title
        if isinstance(description, str) == False :
            description = ""
            
        idx = len(data['image'])
        image_lists = ast.literal_eval(meta.img_list[i])
        colors = ast.literal_eval(meta.color[i])
        if len(colors) != len(image_lists): continue
        for path_img, color in zip(image_lists, colors):
            path_img = f"./data/uniqlo_final/uniqlo/{path_img}"
            if check_path_image(path_img) == False:
                cnt +=1
                continue
            color = translate(color)
            content = content + f" với màu sắc {color} " + description
            data['image'].append(path_img)
            data['text'].append(preprocess_text(content))
    logging.info("Dataset: uniqlo")
    logging.info(f"Complete data: {path}")
    logging.info(f"{cnt} error")
    logging.info(f"{len(data['text'])} sucess")
    
    return data 
def process_data_icondenim(path):
    meta = pd.read_csv(path).reset_index()
    data = {"image": [], "text": []}
    cnt = 0
    for i in tqdm(meta.index):
        title = translate(meta.title[i])
        description = meta.description[i]

        if isinstance(description, str) == False and isinstance(title, str) == False:
            continue

        content = title
        if isinstance(description, str) == False :
            description = ""
            
        idx = len(data['image'])
        image_lists = ast.literal_eval(meta.img_list[i])
        colors = ast.literal_eval(meta.color[i])
        if len(colors) != len(image_lists): continue
        for path_img, color in zip(image_lists, colors):
            path_img = f"./data/icondenim_final/icondenim/{path_img}"
            if check_path_image(path_img) == False:
                cnt +=1
                continue
            color = translate(color)
            content = content + f" với màu sắc {color} " + description
            data['image'].append(path_img)
            data['text'].append(preprocess_text(content))
    logging.info("Dataset: icondenim")
    logging.info(f"Complete data: {path}")
    logging.info(f"{cnt} error")
    logging.info(f"{len(data['text'])} sucess")
    
    return data 

def process_data_yame(path):
    meta = pd.read_csv(path).reset_index()
    data = {"image": [], "text": []}
    cnt = 0
    for i in tqdm(meta.index):
        title = meta.title[i]
        description = meta.description[i]

        if isinstance(description, str) == False and isinstance(title, str) == False:
            continue

        content = title
        if isinstance(description, str) == False :
            description = ""
            
        idx = len(data['image'])
        image_lists = ast.literal_eval(meta.img_list[i])
        color = meta.color[i]
        for path_img in image_lists:
            path_img = f"./data/yame_final/yame/{path_img}"
            if check_path_image(path_img) == False:
                cnt +=1
                continue
            content = content + f" với màu sắc {color} " + description
            data['image'].append(path_img)
            data['text'].append(preprocess_text(content))
    logging.info("Dataset: yame")
    logging.info(f"Complete data: {path}")
    logging.info(f"{cnt} error")
    logging.info(f"{len(data['text'])} sucess")
    
    return data 



def proces_data(list_path_datasets, file_name = "data"):
    data = {"image": [], "text": []}
    for (path_dataset, name_dataset) in tqdm(list_path_datasets):
        if name_dataset in "glami-1m_final":
            sub_meta = process_data_glami(f"{path_dataset}/{name_dataset}/{name_dataset}.csv")
        if name_dataset in "adidas_final":
            sub_meta = process_data_adidas(f"{path_dataset}/{name_dataset}/{name_dataset}.csv")
        
        if name_dataset in "uniqlo_final":
            sub_meta = process_data_uniqlo(f"{path_dataset}/{name_dataset}/{name_dataset}.csv")
        if name_dataset in "yame_final":
            sub_meta = process_data_yame(f"{path_dataset}/{name_dataset}/{name_dataset}.csv")
        if name_dataset in "icondenim_final":
            sub_meta = process_data_icondenim(f"{path_dataset}/{name_dataset}/{name_dataset}.csv")
        
        data.update(sub_meta)
        write_json(file_name, data)
    logging.info(f"Dataset: FINAL \n  Have {len(data[text])} samples")

        

    
                
    return data
data_train = proces_data([
    ("./data/", "glami-1m_final"), 
    ("./data/", "adidas_final"),
    ("./data/", "uniqlo_final"),
    ("./data/", "icondenim_final"),
    ("./data/", "yame_final"),
    # ("./data/", "uniqlo_final")
#     ("/kaggle/input/cv-ck-dataset-crawl/icondenim_final", "icondenim_final"), 
    # ("/kaggle/input/cv-ck-dataset-crawl/uniqlo_final", "uniqlo_final"), 
#     ("/kaggle/input/cv-ck-dataset-crawl/yame_final", "yame_final"), 
], file_name="./data/data_train.json")
data_test = proces_data([
    # ("./data/", "glami-1m_final"), 
    # ("./data/", "adidas_final"),
    # ("./data/", "uniqlo_final"),
    # ("./data/", "yame_final"),
    ("./data/", "uniqlo_final")
#     ("/kaggle/input/cv-ck-dataset-crawl/icondenim_final", "icondenim_final"), 
    # ("/kaggle/input/cv-ck-dataset-crawl/uniqlo_final", "uniqlo_final"), 
#     ("/kaggle/input/cv-ck-dataset-crawl/yame_final", "yame_final"), 
], file_name="./data/data_test.json")
