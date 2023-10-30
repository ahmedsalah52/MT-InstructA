import pytorch_lightning as pl

import torch.nn as nn
import torch
import numpy as np
import random
from PIL import Image 

from torch.optim.lr_scheduler import StepLR
from train_utils.models.base import base_model
from train_utils.models.GAN import simple_GAN
from train_utils.models.seq import seq_model
from train_utils.models.decision_transformer import DL_model

class TL_model(pl.LightningModule):
    def __init__(self,args,tasks_commands,env,wandb_logger,seed):
        super().__init__()
        self.model_name = args.model

        self.tasks_commands = tasks_commands
        self.evaluate_every = args.evaluate_every
        self.evaluation_episodes = args.evaluation_episodes
        self.tasks = args.tasks
        self.batch_size = args.batch_size
        self.env = env
        self.wandb_logger = wandb_logger
        models = {'base':base_model,'GAN':simple_GAN,'seq':seq_model,'dt':DL_model}
        self.model = models[args.model](args)
        self.preprocess = self.model.preprocess_image
        
        self.opt = self.model.get_optimizer()
        
        self.automatic_optimization =  self.model_name != 'GAN'

        #self.my_scheduler = StepLR(self.opt, step_size=10, gamma=0.5)
    def base_training_step(self, batch, batch_idx):
       
        loss = self.model.train_step(batch,self.device)
        self.log("train_loss", loss,sync_dist=True,batch_size=self.batch_size,prog_bar=True)
        return loss
    def gan_training_step(self, batch, batch_idx):
        batch = {k : v.to(self.device) if k != 'instruction' else v  for k,v in batch.items()}

        optimizer_g, optimizer_d = self.optimizers()
        
        self.toggle_optimizer(optimizer_g)

        #generator step
        g_loss = self.model.generator_step(batch)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        self.toggle_optimizer(optimizer_d)

        d_loss = self.model.discriminator_step(batch)
        
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

        self.log("g_loss", g_loss,sync_dist=True,batch_size=self.batch_size,prog_bar=True)
        self.log("d_loss", d_loss,sync_dist=True,batch_size=self.batch_size,prog_bar=True)

    
    def training_step(self, batch, batch_idx):
        if self.model_name == 'GAN':
            return self.gan_training_step(batch,batch_idx)
        else:
            return self.base_training_step(batch,batch_idx)
        
    def on_train_epoch_end(self):
        if (self.current_epoch % self.evaluate_every == 0) and self.current_epoch > 0:
            print(f"\n epoch {self.current_epoch}  evaluation on device {self.device}")
            success_rate = self.evaluate_model()
            self.log("success_rate", success_rate,sync_dist=True,batch_size=self.batch_size,prog_bar=True) # type: ignore

        #torch.cuda.empty_cache()
        return
    def evaluate_model(self):
        total_success = []
        success_dict = {}
        for task in self.tasks:
            success_rate_row = []
            for pos in [0,1,2]:
                pos_success = 0
                for i in range(self.evaluation_episodes):
                    success = self.run_epi(task,pos)
                    pos_success+=success
                success_rate_row.append(float(pos_success)/self.evaluation_episodes)
                total_success+= success_rate_row
            
            success_dict[task]=success_rate_row[:]

        for task , row in success_dict.items(): print(f'success rate in {task} with mean {np.mean(row)} and detailed {row}')
        success_rate =  np.mean(total_success)
        print('total success rate',success_rate)         
            #self.log(task, np.mean(success_rate_row),sync_dist=True,batch_size=self.batch_size) # type: ignore
        return success_rate


    def run_epi(self,task,pos):
        env = self.env(task,pos,save_images=True,wandb_render = False,wandb_log = False,general_model=True)
        obs , info = env.reset()
        instruction = random.choice(self.tasks_commands[task])
        #rendered_seq = []
        i = 0
        a = torch.tensor([0,0,0,0],dtype=torch.float16)
        while 1:
            with torch.no_grad():
                step_input = {'instruction':[instruction]}
                images = [self.model.preprocess_image(Image.fromarray(np.uint8(img))) for img in info['images']]
                step_input['images']   = torch.stack(images).unsqueeze(0).to(self.device)
                step_input['hand_pos'] = torch.tensor(np.concatenate((obs[0:4],obs[18:22]),axis =0)).to(torch.float32).unsqueeze(0).to(self.device)
                step_input['timesteps'] = torch.tensor([i],dtype=torch.int16).unsqueeze(0).to(self.device)
                step_input['action']    = a.unsqueeze(0).to(self.device)

                a = self.model.eval_step(step_input)
                obs, reward, done,success, info = env.step(a.detach().cpu().numpy()[0]) 
                #rendered_seq.append(env.get_visual_obs_log())
                i+=1
                if (success or done): break 
        
        env.close()
        return success
       

    def configure_optimizers(self):
        #return self.opt
        return self.model.get_optimizer()#, [{"scheduler": self.my_scheduler,  'name': 'lr_scheduler'}]



def load_checkpoint(model,checkpoint_path):
    """
    Load a PyTorch Lightning checkpoint and ignore differences between
    the loaded model and the current model.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (pl.LightningModule): The current PyTorch Lightning model.

    Returns:
        model (pl.LightningModule): The model with the loaded checkpoint's state.
    """
    if not checkpoint_path:
        return model
    print('load weights')
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

    # Retrieve the state dictionary from the checkpoint
    checkpoint_state_dict = checkpoint['state_dict']

    # Check for parameter mismatches and load matching parameters
    model_state_dict = model.state_dict()
    for name, param in checkpoint_state_dict.items():
        if name in model_state_dict:
            if model_state_dict[name].shape == param.shape:
                model_state_dict[name] = param
            else:
                print(f"Ignored parameter '{name}' due to shape mismatch.")
        else:
            print(f"Ignored parameter '{name}' not found in the current model.")

    # Load the modified state dictionary into the current model
    model.load_state_dict(model_state_dict)

    return model


def freeze_layers(model,args):
    if args.freeze_modules:
        frozen_modules = args.freeze_modules.split(',')
        for name, param in model.named_parameters():
            for layer_name in frozen_modules:
                if layer_name in name:
                    param.requires_grad = False
                    print('freeze layer ',name)
    return model