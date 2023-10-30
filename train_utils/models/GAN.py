from train_utils.models.base import arch

import torch.nn as nn
import torch
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

class simple_GAN(nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        self.args = args
        self.generator = Generator(args)
        self.discriminator = Discriminator(args)
        self.preprocess_image = self.generator.preprocess_image
   
    def forward(self,batch):
        return self.generator(batch)
    
    def generator_step(self,batch):
        real = torch.ones(batch['action'].size(0), 1)
        real = real.type_as(batch['action'])

        return self.discriminator.loss_fun(self.discriminator(batch,self(batch)),real)
    
    def discriminator_step(self,batch):
       #real
        real = torch.ones(batch['action'].size(0), 1)
        real = real.type_as(batch['action'])
        real_loss = self.discriminator.loss_fun(self.discriminator(batch,batch['action']), real)

        #fake
        fake = torch.zeros(batch['action'].size(0), 1)
        fake = fake.type_as(batch['action'])
        fake_loss = self.discriminator.loss_fun(self.discriminator(batch,self(batch).detach()), fake)

        return (real_loss + fake_loss) / 2
    
    def get_optimizer(self):
        generator_params     = self.generator.get_opt_params()
        discriminator_params = self.discriminator.get_opt_params()

        return [torch.optim.Adam(generator_params,lr=self.args.lr),
                torch.optim.Adam(discriminator_params,lr=self.args.lr)]

   