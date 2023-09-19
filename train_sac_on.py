import torch as th
import torch.nn as nn
from gymnasium import spaces

from PIL import Image
import os
import json
import numpy as np
import torch
import random
import json 
import metaworld
import random
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS
from metaworld.policies import * #SawyerBasketballV2Policy
import numpy as np
import json
import os



#from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC
from meta_env import meta_env,meta_Callback,Custom_replay_buffer

from stable_baselines3.common.torch_layers import MlpExtractor,BaseFeaturesExtractor
from typing import Callable
from stable_baselines3.common.callbacks import CheckpointCallback,CallbackList ,EvalCallback
from stable_baselines3.common.monitor import Monitor

import sys
import wandb
import random
class CustomMLP(BaseFeaturesExtractor):


    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256,emb_dim:int = 0):
        super().__init__( observation_space = observation_space,features_dim = features_dim)
        n_input_channels = observation_space.shape[0]
        self.Pos_embeddings_flag = False

        if emb_dim > 0:
            n_input_channels = n_input_channels + emb_dim -1
            self.Pos_embeddings_flag = True
            self.embedding = nn.Embedding(3, emb_dim)

        self.linear = nn.Sequential(nn.Linear(n_input_channels, features_dim),
                                    nn.LayerNorm(features_dim))
    def forward(self, observations: th.Tensor) -> th.Tensor:
        if self.Pos_embeddings_flag:
            poses = observations[:,0]
            observations = observations[:,1:]
            embs = self.embedding(poses.int())
            observations = th.cat([embs,observations],dim=-1)
        return self.linear(observations)

class LeakyReLU(nn.LeakyReLU):
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False) -> None:
        super().__init__(negative_slope, inplace)


def linear_schedule(initial_value: float,min_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        ret =  progress_remaining * initial_value
        
        return max(ret,min_value) 

    return func


def main():
    random.seed(42)
    task_name  = sys.argv[1]  # "button-press-topdown-v2" #'basketball-v2' #'assembly-v2' "button-press-topdown-v2"#
    task_pos   = int(sys.argv[2])
    task_poses = ['Right','Mid','Left','Mix']

    logs_dir = '/system/user/publicdata/mansour_datasets/metaworld'
    
    configs = json.load(open(os.path.join('training_configs',task_name+'.json')))
    configs['task_name'] = task_name
    configs['task_pos']  = task_pos

    
    run_name   = task_name + '_' + task_poses[task_pos] + '_ID' + str(configs['run_id'])
    save_path  = logs_dir+"/logs/"+run_name
    load_path = os.path.join(save_path, f"{run_name}_{configs['load_from']}_steps")
    
    policy_kwargs = dict(
    features_extractor_class=getattr(sys.modules[__name__], configs['features_extractor_class']),
    features_extractor_kwargs=dict(features_dim=configs['features_dim'],emb_dim=configs['pos_emb_dim']),
    activation_fn= getattr(sys.modules[__name__], configs['activation']), 
    net_arch = configs['net_arch'],
    share_features_extractor=False
    )
    checkpoint_callback = CheckpointCallback(
    save_freq=configs['buffer_size'],
    save_path=save_path,
    name_prefix=run_name,
    save_replay_buffer=False,
    save_vecnormalize=True,
    )


    env      = meta_env(task_name,task_pos,configs['render'],configs['episode_length'],pos_emb_flag = configs['pos_emb_dim']>0,wandb_render = False,multi=configs['multi'],process = 'train')
    eval_env = meta_env(task_name,task_pos,configs['render'],configs['episode_length'],pos_emb_flag = configs['pos_emb_dim']>0,wandb_render = True ,multi=configs['multi'],process = 'valid')


    eval_callback = EvalCallback(eval_env, best_model_save_path="./eval_logs/"+run_name,
                             log_path=logs_dir+"/eval_logs/"+run_name, eval_freq=100000,
                             deterministic=True, render=False,
                             n_eval_episodes=10)
    
    callbacks = CallbackList([checkpoint_callback, eval_callback])

    
    run = wandb.init(
    project="Metaworld multi-task environment",
    name = run_name,
    config=configs,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=False,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
    )

    
    
    
    print('training on Task:',task_name, ' - ','with rendering' if configs['render'] else 'without rendering')
    if configs['load_from'] == 0:
        model = SAC("MlpPolicy",env,policy_kwargs=policy_kwargs, verbose=configs['verbose'],buffer_size=configs['buffer_size'],train_freq=configs['train_freq'],gradient_steps=configs["gradient_steps"],batch_size=configs['batch_size'],learning_rate=configs['lr'],tensorboard_log=f"runs/{run.id}")
    else:
        model = SAC.load(load_path,env, verbose=configs['verbose'],buffer_size=configs['buffer_size'],train_freq=configs['train_freq'],gradient_steps=configs["gradient_steps"],batch_size=configs['batch_size'],learning_rate=configs['lr'],tensorboard_log=f"runs/{run.id}")
    
    total_timesteps = configs['total_timesteps']

    
    model.learn(total_timesteps=total_timesteps, log_interval=configs['log_interval'],callback=callbacks)
       
    print()
    print('done training on Task:',task_name, ' - ','with rendering' if configs['render'] else 'without rendering',' with success ',env.success_counter)



main()
