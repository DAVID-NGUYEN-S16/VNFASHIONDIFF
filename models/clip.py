import torch.nn as nn
from multilingual_clip import pt_multilingual_clip

class VNCLIP_model(nn.Module):
    def __init__(self, name_model = 'M-CLIP/XLM-Roberta-Large-Vit-L-14'):
        super().__init__()
        self.model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(name_model)
        self.text_encoder = self.model.transformer
        self.linear_proj = self.model.LinearTransformation
    def forward(self, inputs):
        '''
        Return last_hidden_state from input
        Parameters:
        
            inputs: (B, S, D)
        '''
        inp = self.text_encoder(inputs)
        inp = self.linear_proj(inp)
        return inp
