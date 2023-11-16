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
        self.necks = {'transformer':transformer_encoder,'cross_attention':CrossAttentionNeck,None:ret_None}
        self.heads = {'fc':fc_head}
        self.loss_funs = {'cross_entropy':nn.CrossEntropyLoss,
                     'mse':nn.MSELoss}
        
        self.args = args

    def eval_step(self,input_step):
        return self.forward(input_step)[0]
    def reset_memory(self):
        pass
class base_model(arch):
    def __init__(self,args) -> None:
        super().__init__(args)
        self.args = args
        self.loss_fun  = self.loss_funs[args.loss_fun]()
        self.backbone  = self.backbones[args.backbone](args)
        self.preprocess_image = self.backbone.preprocess_image
        if args.neck:
            self.neck = self.necks[args.neck](args)
        self.head = self.heads[args.head](args.imgs_emps*len(args.cams)+args.instuction_emps+args.pos_emp,args.action_dim)

        self.cat_backbone_out = args.neck not in ['cross_attention']
        
    def forward(self,batch):
        x = self.backbone(batch,cat=self.cat_backbone_out)
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

   