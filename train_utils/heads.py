import torch.nn as nn
import torch

class fc_head(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        input_dim = args.imgs_emps*len(args.cams)+args.instuction_emps+args.pos_emp
        output_dim = args.action_dim
        acts_funs = {
            'tanh':nn.Tanh
        }
        self.head = nn.Sequential(nn.Flatten(),
                                nn.Linear(input_dim, 512),
                                nn.LayerNorm(512),
                                nn.ReLU(),
                                nn.Linear(512,output_dim))
        
        if args.act_fun:
            self.head.append(acts_funs[args.act_fun]())
            
    def forward(self,embeddings):
        return self.head(embeddings)
    
    def get_opt_params(self):
        return  [
            {"params": self.parameters()}
             ]