import pytorch_lightning as pl

import torch.nn as nn
import torch
import numpy as np
import random
from PIL import Image 
#from timm.scheduler import TanhLRScheduler
#import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import StepLR
from train_utils.backbones import *
from train_utils.necks import *
from train_utils.heads import *


class base_model(nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        self.args = args

        backbones = {'simple_clip':ClIP,'open_ai_clip':Open_AI_CLIP}
        necks = {'transformer':transformer_encoder}
        heads = {'fc':fc_head}
        
        self.backbone  = backbones[args.backbone](args)
        self.preprocess_image = self.backbone.preprocess_image
        if args.neck:
            self.neck = necks[args.neck](args)
        self.head = heads[args.head](args)

        
    def forward(self,batch):
        x = self.backbone(batch)
        if self.args.neck:
            x = self.neck(x)
        x = self.head(x)
        return x
        

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
        loss_funs = {'cross_entropy':nn.CrossEntropyLoss(),
                     'mse':nn.MSELoss()}
        
        self.loss_fun = loss_funs[args.loss_fun]
        self.model = base_model(args) 
        self.preprocess = self.model.preprocess_image

        params = self.model.backbone.get_opt_params(args) + self.model.head.get_opt_params(args)
        if args.neck:
            params += self.model.neck.get_opt_params(args)

        self.opt = torch.optim.Adam(
                params,
                lr=args.lr,
            )
        self.my_scheduler = StepLR(self.opt, step_size=10, gamma=0.5)
    def training_step(self, batch, batch_idx):
        
        

        batch = {k : v.to(self.device) if k != 'instruction' else v  for k,v in batch.items()}
        
        logits = self.model(batch)

        y = batch['action']
        loss = self.loss_fun(logits, y)
        self.log("train_loss", loss,sync_dist=True,batch_size=self.batch_size,prog_bar=True)
        return loss
    
    

    def on_train_epoch_end(self):
        if (self.current_epoch % self.evaluate_every == 0):
            
            print(f"\nepoch {self.current_epoch}  evaluation on device {self.device}")
            total_success = 0
            total_vids =[]
            success_rate_table = []
            for task in self.tasks:
                videos = []
                success_rate_row = []
                for pos in [0,1,2]:
                    env = self.env(task,pos,save_images=True,wandb_render = False,wandb_log = False,general_model=True)
                    for i in range(self.evaluation_episodes):
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
                                total_success+=success
                                #rendered_seq.append(env.get_visual_obs_log())
                                if (success or done): break 
                        
                        env.close()
                        success_rate_row.append(success)
                        #rendered_seq = np.array(rendered_seq, dtype=np.uint8)
                        #rendered_seq = rendered_seq.transpose(0,3, 1, 2)
                        #videos.append(wandb.Video(rendered_seq, fps=30))
                self.log(task, np.mean(success_rate_row),sync_dist=True,batch_size=self.batch_size) # type: ignore

                #total_vids.append(list(reversed(videos)))
                #success_rate_table.append([task]+list(reversed(success_rate_row))) 

            #self.wandb_logger.log_table(key="videos"            ,  columns=['Left','Mid','Right']            ,data=total_vids,step=self.current_epoch)
            #self.wandb_logger.log_table(key="success_rate_table",  columns=['task_name','Left','Mid','Right'],data=success_rate_table,step=self.current_epoch)
            #self.log("samples",total_vids    ,on_epoch=True,sync_dist=True)
            #self.log("evaluations",total_vids,on_epoch=True,sync_dist=True)
        
            self.log("success_rate", float(total_success)/(len(self.tasks)*3*self.evaluation_episodes),sync_dist=True,batch_size=self.batch_size,prog_bar=True) # type: ignore
        #torch.cuda.empty_cache()




    def configure_optimizers(self):
        #return self.opt
        return [self.opt], [{"scheduler": self.my_scheduler,  'name': 'lr_scheduler'}]

