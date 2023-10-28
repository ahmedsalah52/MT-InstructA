import torch.functional as F
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

   
class Generator(arch):
    def __init__(self,args) -> None:
        super().__init__(args)
        self.args = args
        self.loss_fun  = self.loss_funs[args.loss_fun]
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
        self.loss_fun  = self.loss_funs[args.loss_fun]
        self.backbone  = self.backbones[args.backbone](args)
        self.preprocess_image = self.backbone.preprocess_image
        if args.neck:
            self.neck = self.necks[args.neck](args)
        self.head     = self.heads[args.head](6*512+args.pos_emp+args.action_emp,1)
        self.action_emp = nn.Linear(args.action_dim,args.action_emp)
        
        self.sigmoid = nn.Sigmoid()
        self.loss_fun = nn.BCELoss()
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

   