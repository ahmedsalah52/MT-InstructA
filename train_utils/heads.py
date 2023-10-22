import torch.nn as nn
import torch

class fc_head(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.Flatten(),
                                nn.Linear(6*512+args.pos_emp, 512),
                                nn.LayerNorm(512),
                                nn.ReLU(),
                                nn.Linear(512,4))
    def forward(self,embeddings):
        return self.head(embeddings)
    
    def get_opt_params(self,args):
        return  [
            {"params": self.head.parameters()}
             ]