from train_utils.models.base import arch

import torch.nn as nn
import torch
from train_utils.models.seq import seq_model
class Generator(arch):
    def __init__(self,args) -> None:
        super().__init__(args)
        self.args = args
        self.backbone  = self.backbones[args.backbone](args)
        self.preprocess_image = self.backbone.preprocess_image
        if args.neck:
            self.neck = self.necks[args.neck](args)
        self.head     = self.heads[args.head](6*512+args.pos_emp+args.noise_len,args.action_dim)
        self.noise_len = args.noise_len
    
    def forward(self,batch):
        x = self.backbone(batch)
        noise = torch.rand(x.shape[0],self.noise_len).to(x.device)
        x = torch.cat((x,noise),dim=1)
        
        if self.args.neck:
            x = self.neck(x)
        x = self.head(x)
        return x
    def get_opt_params(self):
        params = self.backbone.get_opt_params() + self.head.get_opt_params()
        if self.args.neck:
            params += self.neck.get_opt_params()
        return params
    
class Discriminator(arch):
    def __init__(self,args) -> None:
        super().__init__(args)
        self.args = args
        self.loss_fun = nn.BCELoss()
        self.backbone  = self.backbones[args.backbone](args)
        self.preprocess_image = self.backbone.preprocess_image
        if args.neck:
            self.neck = self.necks[args.neck](args)
        self.head     = self.heads[args.head](6*512+args.pos_emp+args.action_emp,1)
        self.action_emp = nn.Linear(args.action_dim,args.action_emp)
        
        self.sigmoid = nn.Sigmoid()
    def forward(self,batch,actions):

        x = self.backbone(batch)
        x = torch.cat((x,self.action_emp(actions)),dim=1)
        if self.args.neck:
            x = self.neck(x)
        x = self.head(x)
        return self.sigmoid(x)
    
    def get_opt_params(self):
        params = self.backbone.get_opt_params() + self.head.get_opt_params()
        if self.args.neck:
            params += self.neck.get_opt_params()
        return params

class simple_GAN(seq_model):
    def __init__(self,args) -> None:
        super().__init__(args)
        self.noise_len = args.noise_len
        self.discriminator = Discriminator(args)
    def encode(self,batch):
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
        return torch.stack(embeddings,dim=0)
    

    def forward(self,embeddings):
        xs , h = self.seq_module(embeddings)

        outs = [self.head(x) for x in xs]
        outs = torch.stack(outs,dim=0)
        return outs
    def generator_step(self,embeddings,actions):
        real = torch.ones(actions.size(0), 1)
        real = real.type_as(actions)

        return self.discriminator.loss_fun(self.discriminator(embeddings,self(embeddings)),real)
    
    def discriminator_step(self,embeddings,actions,generated_embeddings,generated_actions):
       #real
        real = torch.ones(actions.size(0), 1)
        real = real.type_as(actions)
        real_loss = self.discriminator.loss_fun(self.discriminator(embeddings,actions), real)

        #fake
        fake = torch.zeros(generated_actions.size(0), 1)
        fake = fake.type_as(generated_actions)
        fake_loss = self.discriminator.loss_fun(self.discriminator(generated_embeddings,generated_actions.detach()), fake)

        return (real_loss + fake_loss) / 2
    
    def get_optimizer(self):
        generator_params     = self.generator.get_opt_params()
        discriminator_params = self.discriminator.get_opt_params()

        return [torch.optim.Adam(generator_params,lr=self.args.lr),
                torch.optim.Adam(discriminator_params,lr=self.args.lr)]

