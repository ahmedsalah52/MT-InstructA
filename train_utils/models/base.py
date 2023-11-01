import torch.functional as F
from train_utils.backbones import *
from train_utils.necks import *
from train_utils.heads import *

def ret_None(args):
    return None
class arch(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.backbones = {'simple_clip':ClIP,'open_ai_clip':Open_AI_CLIP}
        self.necks = {'transformer':transformer_encoder,None:ret_None}
        self.heads = {'fc':fc_head}
        self.loss_funs = {'cross_entropy':nn.CrossEntropyLoss,
                     'mse':nn.MSELoss}
        
        self.args = args
        self.dummy_param = nn.Parameter(torch.empty(0))

    def eval_step(self,input_step):
        return self.forward(input_step)[0]

class base_model(arch):
    def __init__(self,args) -> None:
        super().__init__(args)
        self.args = args
        self.loss_fun  = self.loss_funs[args.loss_fun]()
        self.backbone  = self.backbones[args.backbone](args)
        self.preprocess_image = self.backbone.preprocess_image
        if args.neck:
            self.neck = self.necks[args.neck](args)
        self.head = self.heads[args.head](args.imgs_instuction_emps+args.pos_emp,args.action_dim)

        
    def forward(self,batch):
        x = self.backbone(batch)
        if self.args.neck:
            x = self.neck(x)
        x = self.head(x)
        return x
    
    def train_step(self,batch,device,opts=None):
        batch = {k : v.to(device) if k != 'instruction' else v  for k,v in batch.items()}
        
        logits = self.forward(batch)

        y = batch['action']
        return self.loss_fun(logits, y)
    def get_opt_params(self):
        params = self.backbone.get_opt_params() + self.head.get_opt_params()
        if self.args.neck:
            params += self.neck.get_opt_params()
        return params
    def get_optimizer(self):
        params = self.get_opt_params()

        return torch.optim.Adam(params,lr=self.args.lr)

   