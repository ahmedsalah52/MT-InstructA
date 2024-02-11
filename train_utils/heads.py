import torch.nn as nn
import torch

class fc_head(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.use_instruciton = args.head_use_instruction
        if self.use_instruciton:
            input_dim = args.imgs_emps*len(args.cams)+args.instuction_emps+args.pos_emp
        else:
            input_dim = args.imgs_emps*len(args.cams)+args.pos_emp


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
        if self.use_instruciton:
            print('feed with  instruction',[s.shape for s in embeddings])
            return self.head(torch.cat(embeddings,dim=1))
        else:
            images_emps,text_emps,pos_emps = embeddings
            print('feed with no instruction',images_emps.shape,pos_emps.shape)
            return self.head(torch.cat([images_emps,pos_emps],dim=1))
    
    def get_opt_params(self):
        return  [
            {"params": self.parameters()}
             ]