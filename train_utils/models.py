
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
        self.head =     self.heads[args.head](6*512+args.pos_emp,args.action_dim)

        
    def forward(self,batch):
        x = self.backbone(batch)
        if self.args.neck:
            x = self.neck(x)
        x = self.head(x)
        return x
    
    def train_step(self,batch,device,opts):
        batch = {k : v.to(device) if k != 'instruction' else v  for k,v in batch.items()}
        
        logits = self.forward(batch)

        y = batch['action']
        return self.loss_fun(logits, y)
    def get_opt_params(self,args):
        params = self.backbone.get_opt_params(args) + self.head.get_opt_params(args)
        if args.neck:
            params += self.neck.get_opt_params(args)
        return params
    
class Generator(arch):
    def __init__(self,args) -> None:
        super().__init__(args)

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
    
class Discriminator(arch):
    def __init__(self,args) -> None:
        super().__init__(args)

        self.loss_fun  = self.loss_funs[args.loss_fun]
        self.backbone  = self.backbones[args.backbone](args)
        self.preprocess_image = self.backbone.preprocess_image
        if args.neck:
            self.neck = self.necks[args.neck](args)
        self.head     = self.heads[args.head](6*512+args.pos_emp+args.action_emp,1)
        self.action_emp = nn.Linear(args.action_dim,args.action_emp)
        
        self.sigmoid = nn.Sigmoid()
    def forward(self,batch,generated_actions):
        x = self.backbone(batch)
        x = torch.cat((x,generated_actions),dim=1)
        if self.args.neck:
            x = self.neck(x)
        x = self.head(x)
        return self.sigmoid(x)
    
   

class simple_GAN(nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        self.args = args
        self.generator = Generator(args)
        self.discriminator = Discriminator(args)

    def generator_step(self,batch):
        self.generator.zero_grad()

        # Calculate generator output
        fake_output = self.discriminator(fake_data, data)

        # Calculate the loss
        loss = -torch.log(fake_output).mean()

    def discriminator_step(self,batch,generated_actions):
        self.discriminator.zero_grad()
        # Calculate discriminator outputs for real and fake data
        real_output = self.discriminator(batch, batch['action'])
        fake_output = self.discriminator(batch,generated_actions)

        # Calculate the loss
        real_loss = torch.log(real_output)
        fake_loss = torch.log(1 - fake_output)
        loss = -(real_loss.mean() + fake_loss.mean())

        return loss
    
    def training_step(self, batch):
        imgs, _ = batch

        optimizer_g, optimizer_d = self.optimizers()

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)

        # train generator
        # generate images
        self.toggle_optimizer(optimizer_g)
        self.generated_imgs = self(z)

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        # adversarial loss is binary cross-entropy
        g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        self.toggle_optimizer(optimizer_d)

        # how well can it label as real?
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

        # how well can it label as fake?
        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(imgs)

        fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)
