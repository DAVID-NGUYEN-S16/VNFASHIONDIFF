import torch.nn as nn
import torch
class VNCLIPEncoder(nn.Module):
    def __init__(self, model, config = None):
        super().__init__()
        self.config = config
        self.dtype = torch.float32
        self.text_encoder = model.transformer
        self.linear_proj = model.LinearTransformation
        
    def forward(self, input_ids, attention_mask, return_dict=False, output_hidden_states = True):
        '''
        Return last_hidden_state from input
        Parameters:
        
            inputs: (B, S, D)
        '''
        print("+++++++++++++++++++++++")
        print(input_ids.size)
        inp = self.text_encoder(input_ids,attention_mask = attention_mask ).last_hidden_state
        inp = self.linear_proj(inp)
        return [inp]
