import torch
from torch import nn 
import torch.nn.functional as F
class keys_pool(nn.Module):
    def __init__(self, input_size,d_dim,pool_size):
        super().__init__()
        self.key_encoder = nn.Linear(input_size,d_dim)
        self.query       = nn.Parameter(torch.randn((pool_size,d_dim)))

    def forward(self,x):
        encoded_keys = self.key_encoder(x)
      
        prompt_norm  = self.l2_normalize(self.query  , dim=1)  # Pool_size, C
        x_embed_norm = self.l2_normalize(encoded_keys, dim=1)  # B, C
        similarity = torch.matmul(x_embed_norm, prompt_norm.t())  # B, Pool_size
        idx = torch.argmax(similarity, dim=1)
        return idx
    @staticmethod
    def l2_normalize(x, dim=None, epsilon=1e-12):
        return torch.nn.functional.normalize(x, p=2.0, dim=dim, eps=epsilon)