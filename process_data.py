
from tqdm.notebook import tqdm


import pandas as pd
import ast
from utils import translate, write_json, check_path_image, preprocess_text
import logging
from multiprocessing import Pool
import os
logging.basicConfig(filename='process_data.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')


def process_data_glami(path, path_img_root):
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
        path_img = f"{path_img_root}{name_image}"
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
def process_data_adidas(path, path_img_root):
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
            path_img = f"{path_img_root}{path_img}"
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

def process_data_uniqlo(path, path_img_root):
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
            path_img = f"{path_img_root}{path_img}"
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
def process_data_icondenim(path, path_img_root):
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
            path_img = f"{path_img_root}{path_img}"
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

def process_data_yame(path, path_img_root):
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
            path_img = f"{path_img_root}{path_img}"
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
        name = name_dataset.split('_')[0]
        path_img = f"{path_dataset}/{name_dataset}/{name}/"
        if name_dataset in "GLAMI-1M":
            path_img = f"{path_dataset}/images/"
            sub_meta = process_data_glami(f"{path_dataset}/{name_dataset}.csv", path_img)
        if name_dataset in "adidas_final":
            sub_meta = process_data_adidas(f"{path_dataset}/{name_dataset}/{name_dataset}.csv", path_img)
        
        if name_dataset in "uniqlo_final":
            sub_meta = process_data_uniqlo(f"{path_dataset}/{name_dataset}/{name_dataset}.csv", path_img)
        if name_dataset in "yame_final":
            sub_meta = process_data_yame(f"{path_dataset}/{name_dataset}/{name_dataset}.csv", path_img)
        if name_dataset in "icondenim_final":
            sub_meta = process_data_icondenim(f"{path_dataset}/{name_dataset}/{name_dataset}.csv", path_img)
        
        data.update(sub_meta)
        write_json(file_name, data)
    logging.info(f"Dataset: FINAL \n  Have {len(data['text'])} samples")

        

    
                
    return data
data_train = proces_data([
    ("/kaggle/input/cv-ck-dataset/", "GLAMI-1M"), 
    ("/kaggle/input/cv-ck-crawl/", "adidas_final"),
    ("/kaggle/input/cv-ck-crawl/", "uniqlo_final"),
    ("/kaggle/input/cv-ck-crawl/", "icondenim_final"),
    ("/kaggle/input/cv-ck-crawl/", "yame_final"),

], file_name="data_train.json")
data_test = proces_data([

    ("/kaggle/input/cv-ck-crawl/", "uniqlo_final")

], file_name="data_test.json")
