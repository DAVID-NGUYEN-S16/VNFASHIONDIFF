import torch.nn as nn
import torch
from multilingual_clip import pt_multilingual_clip
import gc
class VNCLIPEncoder(nn.Module):
    def __init__(self, model, config = None):
        super().__init__()
        
        self.config = config
        self.dtype = torch.float32
        model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(config.name_model)
        self.text_encoder = model.transformer
        self.linear_proj = model.LinearTransformation
        del model
        torch.cuda.empty_cache()
        gc.collect()
    def forward(self, input_ids, attention_mask, return_dict=False, output_hidden_states = True):
        '''
        Return last_hidden_state from input
        Parameters:
        
            inputs: (B, S, D)
        '''
   
        if len(input_ids.size()) == 3:
            input_ids = input_ids.squeeze(1)
            attention_mask = attention_mask.squeeze(1)
        inp = self.text_encoder(input_ids,attention_mask = attention_mask ).last_hidden_state
        inp = self.linear_proj(inp)
        return [inp]
