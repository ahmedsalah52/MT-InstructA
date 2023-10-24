import pytorch_lightning as pl

import torch.nn as nn
import torch
import numpy as np
import random
from PIL import Image 

from torch.optim.lr_scheduler import StepLR
from train_utils.models import *


class TL_model(pl.LightningModule):
    def __init__(self,args,tasks_commands,env,wandb_logger,seed):
        super().__init__()
        self.tasks_commands = tasks_commands
        self.generate_data_every = args.generate_data_every
        self.evaluate_every = args.evaluate_every
        self.evaluation_episodes = args.evaluation_episodes
        self.tasks = args.tasks
        self.batch_size = args.batch_size
        self.env = env
        self.wandb_logger = wandb_logger
        models = {'base':base_model,'GAN':simple_GAN}
        self.model = models[args.model](args)
        self.automatic_optimization =  args.model != 'GAN'
        self.preprocess = self.model.preprocess_image

        
        params = self.model.get_opt_params(args)
        self.opt = torch.optim.Adam(
                params,
                lr=args.lr,
            )
        self.my_scheduler = StepLR(self.opt, step_size=10, gamma=0.5)
    def base_training_step(self, batch, batch_idx):
       
        loss = self.model.train_step(batch,self.device)
        self.log("train_loss", loss,sync_dist=True,batch_size=self.batch_size,prog_bar=True)
        return loss
    def gan_training_step(self, batch, batch_idx):
       
        loss = self.model.train_step(batch,self.device)
        self.log("train_loss", loss,sync_dist=True,batch_size=self.batch_size,prog_bar=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        if self.args.model == 'GAN':
            return self.gan_training_step(batch,batch_idx)
        else:
            return self.base_training_step(batch,batch_idx)
        
    def on_train_epoch_end(self):
        if (self.current_epoch % self.evaluate_every == 0):
            print(f"\n epoch {self.current_epoch}  evaluation on device {self.device}")
            total_success = 0 
            for task in self.tasks:
                success_rate_row = []
                for pos in [0,1,2]:
                    for i in range(self.evaluation_episodes):
                        success = self.run_epi(task,pos)
                        total_success+=success
                        success_rate_row.append(success)
                       
                self.log(task, np.mean(success_rate_row),sync_dist=True,batch_size=self.batch_size) # type: ignore
            self.log("success_rate", float(total_success)/(len(self.tasks)*3*self.evaluation_episodes),sync_dist=True,batch_size=self.batch_size,prog_bar=True) # type: ignore
        #torch.cuda.empty_cache()
        return

    def run_epi(self,task,pos):
        env = self.env(task,pos,save_images=True,wandb_render = False,wandb_log = False,general_model=True)
        obs , info = env.reset()
        instruction = random.choice(self.tasks_commands[task])
        #rendered_seq = []
        while 1:
            with torch.no_grad():
                step_input = {'instruction':[instruction]}
                images = [self.model.preprocess_image(Image.fromarray(np.uint8(img))) for img in info['images']]
                step_input['images']   = torch.stack(images).unsqueeze(0).to(self.device)
                step_input['hand_pos'] = torch.tensor(np.concatenate((obs[0:4],obs[18:22]),axis =0)).to(torch.float32).unsqueeze(0).to(self.device)
                a = self.model(step_input)
                obs, reward, done,success, info = env.step(a.detach().cpu().numpy()[0]) 
                #rendered_seq.append(env.get_visual_obs_log())
                if (success or done): break 
        
        env.close()
        return success
       

    def configure_optimizers(self):
        #return self.opt
        return [self.opt], [{"scheduler": self.my_scheduler,  'name': 'lr_scheduler'}]

