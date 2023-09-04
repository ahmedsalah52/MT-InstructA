import random
import numpy as np
from gymnasium import Env, spaces
import cv2

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
import pickle

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback,CheckpointCallback
import copy
from gymnasium.spaces import Dict, Box,Space
from stable_baselines3.common.buffers import ReplayBuffer
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS_multi
from metaworld.envs.build_random_envs import Multi_task_env

from typing import Any, Dict, Generator, List, Optional, Union


class Custom_replay_buffer(ReplayBuffer):
    def __init__(self, buffer_size: int, observation_space: Space, action_space: Space, device: str = "auto", n_envs: int = 1, optimize_memory_usage: bool = False, handle_timeout_termination: bool = True):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage, handle_timeout_termination)

        self.renders    = np.zeros((self.buffer_size, self.n_envs, 5,224,224,3) , dtype=np.uint8)
        self.file_order = np.zeros((self.buffer_size, self.n_envs, 1) , dtype=np.uint16)
        self.success    = np.zeros((self.buffer_size, self.n_envs, 1) , dtype=bool)




    def add(self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        self.renders[self.pos]     = np.array([info['images'] for info in infos])
        self.file_order[self.pos]  = np.array([info['file_order'] for info in infos])
        self.success[self.pos]     = np.array([bool(info['success']) for info in infos])
        super().add(obs,next_obs,action,reward,done,infos)

class meta_env(Env):
    def __init__(self,taskname,task_pos,save_images,episode_length = 200) -> None:
        super().__init__()
        class multi_task_V2(ALL_V2_ENVIRONMENTS_multi[taskname],Multi_task_env):
            def __init__(self,main_pos_index=None , task_variant = None) -> None:
                Multi_task_env.__init__(self)
                self.main_pos_index = main_pos_index
                self.task_variant = task_variant

            def reset_variant(self):
                ALL_V2_ENVIRONMENTS_multi[taskname].__init__(self)
                self._freeze_rand_vec = False
                self._set_task_called = True
                self._partially_observable = False #taskname not in ['assembly-v2', 'coffee-pull-v2', 'coffee-push-v2']

        self.taskname = taskname
        self.task_pos = task_pos
        self.env = multi_task_V2(main_pos_index=task_pos)
        self.env.reset_variant()  


        self.action_space = spaces.Box(-1, 1, shape=(4,)) #self.env.action_space
        #self.observation_space = Dict({'state' :self.env.observation_space_gymnasium() , 'render': Box(0, 255, shape=(5,224,224,3), dtype=np.uint8)})
        self.observation_space = self.env.observation_space_gymnasium()
        self.steps = 0
        self.episode_number = 0
        self.max_steps = episode_length
        self.end_episode = False
        self.current_episode = defaultdict(list)
        self.dump_states = True
        self.save_images = save_images
        self.success_counter = 0
        self.total_rewards = 0

    


    def reset(self,seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset_variant()  

        obs = self.env.reset()
        images = None
        #if self.dump_states:
        if self.save_images:
            images = self.get_visual_obs()

        #obs = {'state':obs,'render':images}
        
        self.steps = 0
        self.total_rewards = 0
        self.prev_reward = 0
        
        return obs ,  {'images':images,'file_order':self.env.file_order,'success':0.0} # Reset environment
        
    
    def render(self, mode='human'):

        return cv2.rotate(cv2.cvtColor(self.env.render(offscreen= True,camera_name='behindGripper'),cv2.COLOR_RGB2BGR), cv2.ROTATE_180)
    
    def close (self):
        pass

    def step(self,a):
        images = None
        obs, reward, done, info = self.env.step(a)
        self.steps +=1
        done = self.steps >= self.max_steps

        self.end_episode = done or (info['success']==1.0)
        if info['success']==1.0:
            self.success_counter+=1
            print(' success = ', self.success_counter)
        if self.save_images:
            images = self.get_visual_obs()

     
        info['images'] = images
        info['file_order'] = self.env.file_order

        #if done and not (info['success']==1.0): reward -= 50

        #reward += self.additional_reward(obs)
        

        #d_reward = reward - self.prev_reward
        #self.prev_reward = reward
        #self.total_rewards += d_reward
        return obs, reward, done ,(info['success']==1.0),info
    
    
    def additional_reward(self,obs):
        hand_pos = obs[:3]
        x = hand_pos[0]
        x_shift = self.env.x_shift

        delta_x = abs(x_shift - x)
        reward = (0.7 - delta_x)/0.7
        return reward

    def get_visual_obs(self):
        corner         = self.env.render(offscreen= True,camera_name='corner') # corner,2,3, corner2, topview, gripperPOV, behindGripper'
        corner2        = self.env.render(offscreen= True,camera_name='corner2')
        behindGripper  = self.env.render(offscreen= True,camera_name='behindGripper')
        corner3        = self.env.render(offscreen= True,camera_name='corner3')
        topview        = self.env.render(offscreen= True,camera_name='topview')
        
        images = [cv2.cvtColor(corner,cv2.COLOR_RGB2BGR),       
                cv2.cvtColor(corner2,cv2.COLOR_RGB2BGR),      
                cv2.cvtColor(behindGripper,cv2.COLOR_RGB2BGR),
                cv2.cvtColor(corner3,cv2.COLOR_RGB2BGR),      
                cv2.cvtColor(topview,cv2.COLOR_RGB2BGR)      
        ]


        input_imgs = []
        for image in images:
            input_imgs.append(cv2.resize(image,(224,224)))
        return np.array(input_imgs)
    

    def update_dict(self,a):
        self.current_episode['a'].append(copy.deepcopy(a))
        self.current_episode['obs'].append(copy.deepcopy(self.obs))
        self.current_episode['reward'].append(copy.deepcopy(self.reward))
        self.current_episode['done'].append(copy.deepcopy(self.done))
        self.current_episode['info'].append(copy.deepcopy(self.info))
        if self.save_images:
            self.current_episode['images'].append(copy.deepcopy(self.images))


class meta_Callback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose=0,env=None,save_dir=None):
        super(meta_Callback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.env = env
        self.save_dir = save_dir
    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        print('_on_training_start ')
        self.save_replay_buffer('_on_training_start ')

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        print('_on_rollout_start')



    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        """if self.env.end_episode:

            with open(os.path.join(self.save_dir,self.env.env.file_name+'-'+str(self.env.episode_number)+'-'+str(self.env.current_episode['info'][-1]['success']==1.0)+'.pkl'), 'wb') as fp:
                pickle.dump(self.env.current_episode, fp)
                print('Episode {}  {}  {}'.format(self.env.episode_number,self.env.env.file_name,self.env.current_episode['info'][-1]['success']))
            self.env.episode_number  +=1
            self.env.end_episode = False
            self.env.current_episode = defaultdict(list)
        """
        print('step')
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        print('_on_rollout_end')
        self.save_replay_buffer('_on_rollout_end ')

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        print('_on_training_end')