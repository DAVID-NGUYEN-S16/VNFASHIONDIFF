import yaml
from accelerate.state import AcceleratorState
import accelerate
import json
import re
import os
from PIL import Image
from omegaconf import OmegaConf
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer_en2vi = AutoTokenizer.from_pretrained("vinai/vinai-translate-en2vi-v2", src_lang="en_XX")
model_en2vi = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-en2vi-v2")
device_en2vi = torch.device("cuda")
model_en2vi.to(device_en2vi)


def translate_en2vi(en_texts: str) -> str:
    input_ids = tokenizer_en2vi(en_texts, padding=True, return_tensors="pt").to(device_en2vi)
    output_ids = model_en2vi.generate(
        **input_ids,
        decoder_start_token_id=tokenizer_en2vi.lang_code_to_id["vi_VN"],
        num_return_sequences=1,
        num_beams=5,
        early_stopping=True
    )
    vi_texts = tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True)
    return vi_texts

# The input may consist of multiple text sequences, with the number of text sequences in the input ranging from 1 up to 8, 16, 32, or even higher, depending on the GPU memory.

def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)  

def load_config(file_path):
    """
    Load a YAML configuration file using OmegaConf.
    """
    config = OmegaConf.load(file_path)
    return config

def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]
    
from googletrans import Translator

translator = Translator()

def translate(text):
    if isinstance(text, str) == False:
        return text
    try:
        return translate_en2vi([text])[0]
    except:
        return text
# print(translate( "quần easy short nhiều màu với màu sắc off white"))

def preprocess_text(text: str) -> str:    
    text = re.sub(r"['\",\.\?:\-!]", "", text)
    text = text.strip()
    text = " ".join(text.split())
    text = text.lower() 
    return text
def check_path_image(path):
    check  = True

    check = os.path.exists(path)
    try:
        image = Image.open(path)
    except:
        check = False
    return check