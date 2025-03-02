import torch.functional as F
from train_utils.backbones import *
from train_utils.necks.transformer import *
from train_utils.necks.cross_attention import *
from train_utils.necks.film import *
from train_utils.heads import *
from train_utils.loss_funs import RelativeMSELoss,MSE

def ret_None(args):
    return None
class arch(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.backbones = {'open_ai_clip':Open_AI_CLIP}
        self.necks = {'transformer':transformer_encoder,
                      'cross_attention':CrossAttentionNeck,
                      'film':Film,
                      None:ret_None}
        self.heads = {'fc':fc_head}
        self.loss_funs = {
                     'mse':MSE,
                     'relative_mse':RelativeMSELoss}
        
        self.args = args
        self.dataset_specs = None
        self.success_idx = [] 
    def fill_success_idx(self,success_dict):
        for task,poses in success_dict.items():
            success_rate = torch.mean(torch.tensor([c/self.args.evaluation_episodes for c in poses.values()])).item()
            task_idx = self.args.tasks.index(task)
            if success_rate >= self.args.success_threshold:
                if task_idx not in self.success_idx:
                    self.success_idx.append(task_idx)
                print(f'task: {task} reached the success rate threshold')

    def eval_step(self,input_step):
        return self.forward(input_step)[0]
    def reset_memory(self):
        pass
    @property
    def device(self):
        # Return the device of the first parameter of the model
        return next(self.parameters()).device
    def set_dataset_specs(self,dataset_specs):
        self.dataset_specs = dataset_specs
         
class base_model(arch):
    def __init__(self,args) -> None:
        super().__init__(args)
        self.args = args
        self.loss_fun  = self.loss_funs[args.loss_fun](args)
        self.backbone  = self.backbones[args.backbone](args)
        self.preprocess_image = self.backbone.preprocess_image
        if args.neck:
            self.neck = self.necks[args.neck](args)
        self.head = self.heads[args.head](args)

        self.cat_out = args.neck not in ['cross_attention','film']
        #self.dummy_param = nn.Parameter(torch.zeros(0))
    def forward(self,batch):
        x = self.backbone(batch,cat=self.cat_out)
        
        if self.args.neck:
            x = self.neck(x,cat=self.cat_out)

        x = self.head(x)
        return x
    
    def train_step(self,batch,device,opts=None):
        batch = {k : v.to(device) if type(v) == torch.tensor else v  for k,v in batch.items()}
        
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
        
        return torch.optim.AdamW(params,lr=self.args.lr,weight_decay=self.args.weight_decay)

   