
from stable_baselines3.common.torch_layers import MlpExtractor,BaseFeaturesExtractor
import torch
from torch import nn 
from train_utils.tl_model import TL_model,load_checkpoint,freeze_layers
from train_utils.args import  parser ,process_args
import gymnasium as gym

from meta_env import meta_env,task_manager
from train_utils.metaworld_dataset import split_dict

import cv2
from PIL import Image
import numpy as np
import random
import os
import shutil
import json

class genaral_model(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 256,GM_args:parser = None) -> None:
        super().__init__(observation_space, features_dim)
        #observation_dim = observation_space['obs'].shape[-1]
        #model = TL_model.load_from_checkpoint(GM_args.load_checkpoint_path,args=GM_args,tasks_commands=None,env=meta_env,wandb_logger=None,seed=None)
       
        self.commands = {}# convert command dict to work with idx
        tasks_commands = json.load(open(GM_args.tasks_commands_dir))
        tasks_commands = {k:list(set(tasks_commands[k])) for k in GM_args.tasks} #the commands dict should have the same order as args.tasks list
   
        commands_dicts = split_dict(tasks_commands,GM_args.commands_split_ratio,seed=42)
        max_len = max([len(com_dict[task]) for task in GM_args.tasks for com_dict in commands_dicts])
        self.commands_array = [[com_dict[task] + [''] * (max_len - len(com_dict[task])) for task in GM_args.tasks] for com_dict in commands_dicts]

        model = TL_model(args=GM_args,tasks_commands=None,env=meta_env,wandb_logger=None,seed=GM_args.seed)

        model = load_checkpoint(model,GM_args.load_weights)
        model = freeze_layers(model , GM_args)
        self.model = model
    def forward(self, observations):
        #observations = observations['obs'].to(torch.float32)
        batch_step = {k: v[:, -1] for k, v in observations.items()} #take only the last step of the seq
        b,cam,h,w,c = batch_step['images'].shape
        batch_step['images'] = self.model.preprocess(batch_step['images'].reshape(b*cam,h,w,c)).reshape(b,cam,c,h,w)
        
        command_dict_idx = batch_step['command_dict_idx'].cpu().numpy()
        task_id = batch_step['task_id'].cpu().numpy()
        command_id = batch_step['command_id'].cpu().numpy()
        batch_step['instruction'] = self.commands_array[command_dict_idx][task_id][command_id]

        x = self.model.backbone(batch_step,cat=self.cat_backbone_out)
        x = self.model.neck(x)

        return x
