
from train_utils.backbones import *
from train_utils.necks import *
from train_utils.heads import *

class arch(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.backbones = {'simple_clip':ClIP,'open_ai_clip':Open_AI_CLIP}
        self.necks = {'transformer':transformer_encoder}
        self.heads = {'fc':fc_head}
        self.loss_funs = {'cross_entropy':nn.CrossEntropyLoss,
                     'mse':nn.MSELoss}
        self.args = args

class base_model(arch):
    def __init__(self,args) -> None:
        super().__init__(args)

        self.loss_fun  = self.loss_funs[args.loss_fun]()
        self.backbone  = self.backbones[args.backbone](args)
        self.preprocess_image = self.backbone.preprocess_image
        if args.neck:
            self.neck = self.necks[args.neck](args)
        self.head =     self.heads[args.head](6*512+args.pos_emp,4)

        
    def forward(self,batch):
        x = self.backbone(batch)
        print('before neck',x.shape)

        if self.args.neck:
            x = self.neck(x)
        print('after neck',x.shape)
        x = self.head(x)
        return x
    
    def train_step(self,batch,device):
        batch = {k : v.to(device) if k != 'instruction' else v  for k,v in batch.items()}
        
        logits = self.forward(batch)

        y = batch['action']
        return self.loss_fun(logits, y)


class Discriminator(arch):
    def __init__(self,args) -> None:
        super().__init__(args)

        self.loss_fun  = self.loss_funs[args.loss_fun]
        self.backbone  = self.backbones[args.backbone](args)
        self.preprocess_image = self.backbone.preprocess_image
        if args.neck:
            self.neck = self.necks[args.neck](args)
        self.head     = self.heads[args.head](6*512+args.pos_emp,4)

        self.generator     = self.backbones[args.backbone](args)
        self.discriminator = self.backbones[args.backbone](args)
        
    def forward(self,batch):
        x = self.backbone(batch)
        if self.args.neck:
            x = self.neck(x)
        x = self.head(x)
        return x
    
    def train_step(self,batch,loss_fun,device):
        batch = {k : v.to(device) if k != 'instruction' else v  for k,v in batch.items()}
        
        logits = self.forward(batch)

        y = batch['action']
        return self.loss_fun(logits, y)
