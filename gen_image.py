from train import load_models,load_config
import json
import os 
from tqdm import tqdm
from safetensors.torch import load_model, save_model

config = load_config("./config.yaml")
model, tokenizer = load_models(config=config)
print( f"{config.path_fineturn_model}/model.safetensors")
load_model(model, f"{config.path_fineturn_model}/model.safetensors")
model.to("cuda")
model.eval()
model.set_up()
def gen_image(path_data, path_output):
    with open(path_data, 'r') as f:
        data = json.load(f)
    
    for i in tqdm(range(len(data['image']))):
        name = os.path.basename(data['image'][i])
        images, _  = model.inference(text = data['text'][i], step = 50)
        images[0].save(f"{path_output}/{name}")
    
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gen images") 
    parser.add_argument('--path_data', type=str, help='path path data')
    parser.add_argument('--path_output', type=str, help='path path output')
    args = parser.parse_args()
    gen_image(path_data = args.path_data, path_output = args.path_output )
        