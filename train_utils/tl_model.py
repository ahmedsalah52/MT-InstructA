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
from train_utils.models.decision_transformer2 import DT_model
from train_utils.models.decision_transformer_obs2 import DL_model_obs

from tqdm import tqdm
from collections import defaultdict
class TL_model(pl.LightningModule):
    def __init__(self,args,tasks_commands,env,wandb_logger,seed):
        super().__init__()
        self.poses = args.poses
        self.model_name = args.model
        self.cams_ids =  args.cams
        self.tasks_commands = tasks_commands
        self.evaluate_every = args.checkpoint_every
        self.evaluation_episodes = args.evaluation_episodes
        self.eval_checkpoint = not args.trainloss_checkpoint
        self.tasks = args.tasks
        self.batch_size = args.batch_size
        self.env = env
        self.wandb_logger = wandb_logger
        models = {'base':base_model,'GAN':simple_GAN,'seq':seq_model,'dt':DT_model,'dt_obs':DL_model_obs}
        #print('TL model device is ',str(self.device))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        self.model = models[args.model](args).to(device)
        self.preprocess = self.model.preprocess_image
        
        self.opt = self.model.get_optimizer()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.opt,
                patience=args.opt_patience,
                verbose=True
            )
        self.automatic_optimization =  self.model_name != 'GAN'
        self.max_return_to_go = None
        
        #self.my_scheduler = StepLR(self.opt, step_size=args.schedular_step, gamma=0.5)
    def base_training_step(self, batch, batch_idx):
       
        loss = self.model.train_step(batch,self.device)
        self.log("train_loss", loss,sync_dist=True,batch_size=self.batch_size,prog_bar=True)
        return loss
    def gan_training_step(self, batch, batch_idx):
        batch = {k : v.to(self.device) if k != 'instruction' else v  for k,v in batch.items()}

        optimizer_g, optimizer_d = self.optimizers()
        
        self.toggle_optimizer(optimizer_g)
        embeddings = self.model.encode(batch)

        #generator step
        generated_embeddings,generated_actions = self.generate_batch()
        g_loss = self.model.generator_step(generated_embeddings,generated_actions)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        self.toggle_optimizer(optimizer_d)

        d_loss = self.model.discriminator_step(embeddings,batch['action'],generated_embeddings,generated_actions)
        
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
        if self.eval_checkpoint and ((self.current_epoch % self.evaluate_every == 0) and self.current_epoch > 0):
            print(f"\n epoch {self.current_epoch}  evaluation on device {self.device}")
            success_rate = self.evaluate_model()
            self.log("success_rate", success_rate,sync_dist=True,batch_size=self.batch_size,prog_bar=True) # type: ignore

        #torch.cuda.empty_cache()
        return
    def evaluate_model(self):
        self.eval()
        total_success = []
        success_dict = defaultdict(lambda:defaultdict(lambda:0.0))
        pbar = tqdm(total=len(self.tasks)*len(self.poses)*self.evaluation_episodes,desc=f"Evaluation on GPU : {self.device}",leave=True)  

        pbar.set_description(f"Evaluation on GPU : {self.device}")
        for i in range(self.evaluation_episodes):
            for task in self.tasks:
                for pos in self.poses:
                    success = self.run_epi(task,pos)
                    total_success.append(success)
                    success_dict[task][pos] += float(success)
                    pbar.set_description(f"success rate {round(np.mean(total_success),3)} on GPU : {self.device}")
                    pbar.update(1)

            
        pbar.close()
        for task , row in success_dict.items(): print(f'success rate in {task} with mean {np.mean([c/self.evaluation_episodes for c in row.values()])} and detailed {[[k,c/self.evaluation_episodes] for k,c in row.items()]}')
        success_rate =  np.mean(total_success)
        print('total success rate',success_rate)         
            #self.log(task, np.mean(success_rate_row),sync_dist=True,batch_size=self.batch_size) # type: ignore
        self.train()
        return success_rate


    def run_epi(self,task,pos):
        env = self.env(task,pos,save_images='obs' not in self.model_name,wandb_render = False,wandb_log = False,general_model=True,cams_ids=self.cams_ids)
        self.model.reset_memory()
        obs , info = env.reset()
        instruction = random.choice(self.tasks_commands[task])
        #rendered_seq = []
        i = 0
        a = torch.tensor([0,0,0,0],dtype=torch.float16)
        reward = 0
        while 1:
            with torch.no_grad():
                step_input = {'instruction':[instruction],'task_id':torch.tensor([self.tasks.index(task)],dtype=torch.int)}
                if 'obs' in self.model_name:
                    step_input['obs'] = torch.tensor(obs).to(torch.float32).unsqueeze(0).to(self.device)
                else:
                    images = [self.model.preprocess_image(Image.fromarray(np.uint8(img))) for  img in info['images']]
                    step_input['images']   = torch.stack(images).unsqueeze(0).to(self.device)
                    
                step_input['hand_pos'] = torch.tensor(np.concatenate((obs[0:4],obs[18:22]),axis =0)).to(torch.float32).unsqueeze(0).to(self.device)
                step_input['timesteps'] = torch.tensor([i],dtype=torch.int).to(self.device)
                step_input['action']    = a.unsqueeze(0).to(self.device)
                step_input['reward']    = torch.tensor([reward]).to(self.device)

                a = self.model.eval_step(step_input)
                obs, reward, done,success, info = env.step(a.detach().cpu().numpy()) 
                #rendered_seq.append(env.get_visual_obs_log())
                i+=1
                if (success or done): break 
        
        env.close()
        return success
       

    def configure_optimizers(self):
        #return self.opt
        #return self.model.get_optimizer(), [{"scheduler": self.my_scheduler,  'name': 'lr_scheduler'}]
       
        return {"optimizer":  self.opt,"lr_scheduler": self.scheduler,"monitor":"train_loss"}

   
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
    non_frozen_modules = []
    if args.freeze_except:
        non_frozen_modules = args.freeze_except.split(',')

    if args.freeze_modules:
        frozen_modules = args.freeze_modules.split(',')
        for name, param in model.named_parameters():
            for layer_name in frozen_modules:
                if layer_name in name :
                    if len(set(name.split('.')) & set(non_frozen_modules))>0:
                        param.requires_grad = True
                        print('unfreeze layer ',name)
                    else:
                        param.requires_grad = False
                        print('freeze layer ',name)
    return model