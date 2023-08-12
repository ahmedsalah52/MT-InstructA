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
from collections import defaultdict
import cv2
from tqdm import tqdm
#import threading
import _thread

from stable_baselines3.common.callbacks import CheckpointCallback

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC
from meta_env import meta_env,meta_Callback,Custom_replay_buffer

from stable_baselines3.common.torch_layers import MlpExtractor,BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from typing import Callable

import sys
import optuna


class CustomMLP(BaseFeaturesExtractor):


    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__( observation_space = observation_space,features_dim = features_dim)
        n_input_channels = observation_space.shape[0]

        self.linear = nn.Sequential(nn.Linear(n_input_channels, features_dim),
                                    nn.LayerNorm(features_dim))

    def forward(self, observations: th.Tensor) -> th.Tensor:
        
        features = self.linear(observations)


        return features

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


def run_trial(configs):

    task_name  = sys.argv[1]  # "button-press-topdown-v2" #'basketball-v2' #'assembly-v2' "button-press-topdown-v2"#
    
    
    policy_kwargs = dict(
    features_extractor_class=getattr(sys.modules[__name__], configs['features_extractor_class']),
    features_extractor_kwargs=dict(features_dim=configs['features_dim']),
    activation_fn= getattr(sys.modules[__name__], configs['activation']), 
    net_arch = configs['net_arch'],
    share_features_extractor=False

    )
    checkpoint_callback = CheckpointCallback(
    save_freq=configs['buffer_size'],
    save_path="./logs/"+task_name,
    name_prefix=task_name,
    save_replay_buffer=True,
    save_vecnormalize=True,
    )
    env = meta_env(task_name,configs['render'])

    print('training on Task:',task_name, ' - ','with rendering' if configs['render'] else 'without rendering')
    
    model = SAC("MlpPolicy", env,policy_kwargs=policy_kwargs, verbose=configs['verbose'],buffer_size=configs['buffer_size'],train_freq=configs['train_freq'],gradient_steps=configs["gradient_steps"],batch_size=configs['batch_size'],learning_rate=configs['lr']) 
   
    model.learn(total_timesteps=configs['total_timesteps'], log_interval=configs['log_interval'],callback=checkpoint_callback)
       
    print()
    print('done training on Task:',task_name, ' - ','with rendering' if configs['render'] else 'without rendering',' with success ',env.success_counter)
    return env.total_rewards


def objective(trial):
    configs = json.load(open(os.path.join('training_configs','search.json')))

    # 2. Suggest values of the hyperparameters using a trial object.
    n_layers                  = trial.suggest_int('n_layers', 1, 5)
    layer_size                = trial.suggest_categorical('layer_size', [256, 512,1024])
    freq                      = trial.suggest_categorical('train_freq', [1,10,25,50])
    configs['train_freq']     = freq
    configs['gradient_steps'] = freq
    configs['net_arch']       = [layer_size] * n_layers
    print(configs)
    success = run_trial(configs)


    return success
def main():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)

    trial = study.best_trial

    print('Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))
    print("best_params",study.best_params)

main()
