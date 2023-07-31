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


from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC
from meta_env import meta_env,meta_Callback,Custom_replay_buffer

from stable_baselines3.common.torch_layers import MlpExtractor,BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


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

def main():

    task_name  = 'coffee-button-v2'
    
    policy_kwargs = dict(
    features_extractor_class=CustomMLP,
    features_extractor_kwargs=dict(features_dim=256),
    activation_fn= LeakyReLU, 
    net_arch = [256,256,256]
    )
    env = meta_env(task_name,True)


    model = SAC("MlpPolicy", env,policy_kwargs=policy_kwargs, verbose=1,buffer_size=10000,batch_size=256,learning_rate=3e-4,replay_buffer_class = Custom_replay_buffer) #,replay_buffer_class = HerReplayBuffer,replay_buffer_kwargs={ 'n_envs':10}
    #model = SAC.load("button-press-topdown-v2", env, verbose=1,buffer_size=10000,batch_size=512)
    # now save the replay buffer too
    #checkpoint_callback = meta_Callback(env=env,save_dir='logs/episodes')
    for i in tqdm(range(200)):
        model.learn(total_timesteps=10000, log_interval=5)
        model.save(os.path.join('trained_agents',task_name,str(i)))
        model.save_replay_buffer(os.path.join("buffers",task_name,str(i)))

main()