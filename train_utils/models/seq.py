import torch.nn as nn
import torch
from train_utils.models.base import arch


class seq_model(arch):
    def __init__(self,args):
        super().__init__(args)
        self.args = args
        self.loss_fun  = self.loss_funs[args.loss_fun]()
        self.backbone  = self.backbones[args.backbone](args)
        self.preprocess_image = self.backbone.preprocess_image
        if args.neck:
            self.neck = self.necks[args.neck](args)
        self.seq_module = nn.LSTM(args.imgs_instuction_emps+args.pos_emp,hidden_size=512,batch_first=False,bidirectional=False,num_layers=2)

        self.head = self.heads[args.head](512,args.action_dim)
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.hidden_state = None
        self.normalize = nn.LayerNorm(args.imgs_instuction_emps+args.pos_emp)
        #self.memory = deque([torch.zeros(1, self.model.backbone.out_channels, 152, 272).cuda() for _ in range(8)], maxlen=8)  
    def forward(self,batch):
        embeddings = []
        for i in range(self.args.seq_len):
            batch_step = {}
            for k,vs in batch.items():
                batch_step[k] = vs[i]
           
            batch_step = {k : v.to(self.dummy_param.device) if k != 'instruction' else v  for k,v in batch_step.items()}
            x = self.backbone(batch_step)
            del batch_step
          

            self.normalize(x)
            if self.args.neck:
                x = self.neck(x)
            embeddings.append(x)
        del batch
        embeddings = torch.stack(embeddings,dim=0)
        xs , h = self.seq_module(embeddings)

        outs = [self.head(x) for x in xs]
        outs = torch.stack(outs,dim=0)
        return outs

    def eval_step(self,input_step):
        input_step = {k : v.to(self.dummy_param.device) if k != 'instruction' else v  for k,v in input_step.items()}

        x = self.backbone(input_step)
        if self.args.neck:
            x = self.neck(x)

        x , self.hidden_state = self.seq_module(x.unsqueeze(0),self.hidden_state)
        return self.head(x)
    def train_step(self,batch,device,opts=None):
        y = torch.stack(batch['action'],dim=0).to(device)

        logits = self.forward(batch)

        return self.loss_fun(logits, y)
    
    def get_opt_params(self):
        params = self.backbone.get_opt_params() + self.head.get_opt_params() + [{"params": self.seq_module.parameters()}]
        if self.args.neck:
            params += self.neck.get_opt_params()
        return params
    def get_optimizer(self):
        params = self.get_opt_params()

        return torch.optim.Adam(params,lr=self.args.lr)