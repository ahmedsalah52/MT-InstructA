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
import wandb
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


class task_manager():
    def __init__(self,taskname,pos=None,variant=None,multi = True):
        self.env_args = {'main_pos_index':pos , 'task_variant':variant}
        self.task_name = taskname
        self.multi = multi
    def reset(self):
        if self.multi:
            ml1 = metaworld.ML_1_multi(self.task_name,self.env_args)
        else:
            ml1 = metaworld.ML_1_single(self.task_name)

        self.task_ = random.choice(ml1.train_tasks)
        self.env = ml1.my_env_s
        self.env.set_task(self.task_)  # Set task
        return self.env



class meta_env(Env):
    def __init__(self,taskname,task_pos,save_images,episode_length = 200,wandb_render = False,multi = True,process='None') -> None:
        super().__init__()
        
        self.taskname = taskname
        self.task_pos = task_pos
        self.task_man = task_manager(taskname=taskname,pos=task_pos,multi=multi)
        self.env = self.task_man.reset()
        

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
        self.render_mode = 'rgb_array'
        self.wandb_render = wandb_render
        self.multi = multi
        self.process = process

    def reset(self,seed=None, options=None):
        super().reset(seed=seed)
        self.env = self.task_man.reset()
        self.rendered_seq = []

        obs = self.env.reset()
        images = None
        if self.save_images:
            images = self.get_visual_obs()
        if self.wandb_render:
            self.rendered_seq.append(self.get_visual_obs_log())
        
        self.steps = 0
        self.total_rewards = 0
        self.prev_reward = 0
        
        obs = np.hstack((self.env.main_pos_index,obs))
        return obs ,  {'images':images,'file_order':self.env.file_order if self.multi else 0,'success':0.0} # Reset environment
        
    def get_visual_obs_log(self):
        behindGripper  = self.env.render(offscreen= True,camera_name='behindGripper')
        topview        = self.env.render(offscreen= True,camera_name='topview')
        topview        = cv2.rotate(topview, cv2.ROTATE_180)
        behindGripper  = cv2.rotate(behindGripper, cv2.ROTATE_180)
    
        conc_image  = cv2.hconcat([behindGripper,topview])
        conc_image  = cv2.resize(conc_image, (2*256,256))
        return conc_image

    def render(self, mode='human'):
        print('render________________________________________',self.env.current_task_variant)
        #super().render()
        wandb.log({"frame": wandb.Image( cv2.rotate(cv2.cvtColor(self.env.render(offscreen= True,camera_name='behindGripper'),cv2.COLOR_RGB2BGR), cv2.ROTATE_180))})
        pass
    def get_images(self):
        return self.get_visual_obs()
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
        if self.save_images:
            images = self.get_visual_obs()
        if self.wandb_render:
            self.rendered_seq.append(self.get_visual_obs_log())
            if self.end_episode:
                self.rendered_seq = np.array(self.rendered_seq, dtype=np.uint8)
                self.rendered_seq = self.rendered_seq.transpose(0,3, 1, 2)
                wandb.log({"video": wandb.Video(self.rendered_seq, fps=30)})
        if self.end_episode:
            wandb.log({self.process+" success counter": self.success_counter})

        
        
        info['images'] = images
        info['file_order'] = self.env.file_order if self.multi else 0

        obs = np.hstack((self.env.main_pos_index,obs))

            
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