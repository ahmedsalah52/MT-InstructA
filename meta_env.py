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
from gymnasium.spaces import  Box,Space
from stable_baselines3.common.buffers import ReplayBuffer
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS_multi
from metaworld.envs.build_random_envs import Multi_task_env
import wandb
from typing import Any, Dict, Generator, List, Optional, Union
import gymnasium
from collections import deque


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
    def __init__(self,taskname,pos=None,variant=None,multi = True,general_model=False):
        self.env_args = {'main_pos_index':pos , 'task_variant':{'variant':variant,'general_model':general_model}}
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
    def __init__(self,taskname,task_pos,save_images,variant=None,episode_length = 200,pos_emb_flag=False,wandb_render = False,multi = True,process='None',wandb_log = True,general_model = False,cams_ids=[0,1,2,3,4]) -> None:
        super().__init__()
        
        self.taskname = taskname
        self.task_pos = task_pos
        self.task_man = task_manager(taskname=taskname,pos=task_pos,variant=variant,multi=multi,general_model=general_model)
        self.env = self.task_man.reset()
        self.pos_emb_flag = pos_emb_flag

        self.action_space = spaces.Box(-1, 1, shape=(4,)) #self.env.action_space
        
        obs_space = self.env.observation_space
        if self.pos_emb_flag:
            self.observation_space = gymnasium.spaces.Box(
                                                        np.hstack((0,obs_space.low)),
                                                        np.hstack((3,obs_space.high)),
                                                        dtype=np.float64)
        else:
            self.observation_space = gymnasium.spaces.Box(
                                                        obs_space.low,
                                                        obs_space.high,
                                                        dtype=np.float64)
            
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
        self.wandb_log = wandb_log
        self.cams_ids = cams_ids
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
        
        if self.pos_emb_flag: obs = np.hstack((self.env.main_pos_index,obs))
        return obs ,  {'images':images,'file_order':self.env.file_order if self.multi else -1,'success':0.0,'is_success':False} # Reset environment
        
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
        #wandb.log({"frame": wandb.Image( cv2.rotate(cv2.cvtColor(self.env.render(offscreen= True,camera_name='behindGripper'),cv2.COLOR_RGB2BGR), cv2.ROTATE_180))})
        pass
    def get_images(self):
        return self.get_visual_obs()
    def close(self):
        self.env.close()

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
        if self.end_episode and self.wandb_log:
            wandb.log({self.process+" success counter": self.success_counter})        
        
        info['images'] = images
        info['file_order'] = self.env.file_order if self.multi else -1
        info['is_success'] = (info['success'] == 1.0)

        if self.pos_emb_flag: obs = np.hstack((self.env.main_pos_index,obs))
            
        return obs, reward, done ,(info['success']==1.0),info
    
    
  

    def get_visual_obs(self):
        cams = ['corner','corner2','behindGripper','corner3','topview']

        renders = [self.env.render(offscreen= True,camera_name=cam) for i,cam in enumerate(cams) if i in self.cams_ids]
        images = [cv2.resize(img,(224,224)) for img in renders]
        return np.array(images)
    

class sequence_metaenv(Env):
    def __init__(self,commands_dict,save_images,episode_length = 200,wandb_render = False,process='None',wandb_log = True,general_model = False,cams_ids=[0,1,2,3,4],max_seq_len=10,train=True):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.commands_dict = commands_dict
        self.save_images = save_images
        self.episode_length = episode_length
        self.wandb_render = wandb_render
        self.process = process
        self.wandb_log = wandb_log
        self.general_model = general_model
        self.cams_ids = cams_ids

        #random_task = np.random.choice(list(self.commands_dict.keys()))
        self.task_id = 0 #np.random.randint(len(self.commands_dict.keys()))
        task_name = list(self.commands_dict.keys())[self.task_id]
        self.command_id = np.random.randint(len(self.commands_dict[task_name]))

        self.env = meta_env(task_name,task_pos=None,save_images=self.save_images,variant=None,episode_length = self.episode_length,pos_emb_flag=False,wandb_render = self.wandb_render,multi = True,process='None',wandb_log = self.wandb_log,general_model = self.general_model,cams_ids=self.cams_ids)
        self.action_space = self.env.action_space

        observation_space = {#"hand_pos":Box(low=-1,high=1,shape=(max_seq_len,8),dtype=np.float32),
                             #"task_idx":Box(low=-1,high=len(commands_dict),shape=(1,),dtype=np.int32),
                             #"command_idx":Box(low=-1,high=500,shape=(1,),dtype=np.int32),
                             #"actions":Box(low=-1,high=1,shape=(max_seq_len,4),dtype=np.float32),
                            }
        if save_images:
            observation_space["images"] = Box(low=0,high=255,shape=(max_seq_len,len(cams_ids),224,224,3),dtype=np.uint8)
            self.images_list   = deque([np.zeros((len(cams_ids),224,224,3))]*max_seq_len,maxlen=self.max_seq_len)

        else:
            observation_space["obs"] = Box(low=-1,high=1,shape=(max_seq_len,39),dtype=np.float32)
            self.obs_list      = deque([np.zeros(39)]*max_seq_len,maxlen=self.max_seq_len)
        observation_space['command_dict_idx'] = Box(low=0,high=1                         ,shape=(max_seq_len,) ,dtype=np.int32)
        observation_space['task_idx']         = Box(low=0,high=len(commands_dict)        ,shape=(max_seq_len,) ,dtype=np.int32)
        observation_space['command_idx']      = Box(low=0,high=1000                      ,shape=(max_seq_len,) ,dtype=np.int32)
        observation_space['hand_pos']         = Box(low=-1,high=1                        ,shape=(max_seq_len,8),dtype=np.float32)
        self.observation_space = gymnasium.spaces.Dict(observation_space)
        

        self.actions_list  = deque([np.zeros(4)]*max_seq_len,maxlen=self.max_seq_len)
        self.hand_pos_list = deque([np.zeros(8)]*max_seq_len,maxlen=self.max_seq_len)
        self.command_dict_idx = 0 if train else 1

    def prepare_step(self,obs,images,aciton=np.zeros(4)):
        
        self.actions_list[-1] = aciton
        self.actions_list.append(np.zeros(4))
        self.hand_pos_list.append(np.concatenate((obs[0:4],obs[18:22]),axis = 0,dtype=np.float32))
       
        aciton   = np.stack(self.actions_list)
        hand_pos = np.stack(self.hand_pos_list)


        current_state = self.observation_space.sample()
        current_state["hand_pos"]         = hand_pos
        #current_state["actions"]          = aciton
        current_state["task_idx"]         = np.array([self.task_id],dtype=np.int32)   #task_id
        current_state["command_idx"]      = np.array([self.command_id],dtype=np.int32) #command_id
        current_state["command_dict_idx"] = np.array([self.command_dict_idx],dtype=np.int32)
        if self.save_images:
            self.images_list.append(images)
            images        = np.stack(self.images_list)
            current_state["images"] = images
        else:
            self.obs_list.append(obs)
            obs        = np.stack(self.obs_list)
            current_state["obs"] = obs
        
        return current_state 
    
 
    def reset(self,seed=None, options=None):
        super().reset(seed=seed)
        self.task_id = 0 #np.random.randint(len(self.commands_dict.keys()))
        task_name = list(self.commands_dict.keys())[self.task_id]
        self.command_id = np.random.randint(len(self.commands_dict[task_name]))
        
        self.env = meta_env(task_name,task_pos=None,save_images=self.save_images,variant=None,episode_length = self.episode_length,pos_emb_flag=False,wandb_render = self.wandb_render,multi = True,process='None',wandb_log = self.wandb_log,general_model = self.general_model,cams_ids=self.cams_ids)

        obs, first_info = self.env.reset()
        images = first_info['images']
        del first_info['images']
        return self.prepare_step(obs,images) , first_info

    def step(self,a):
        
        obs, reward, done ,success,info = self.env.step(a)
        images = info['images']
        del info['images']
        return self.prepare_step(obs,images,a), reward, done ,success,info
    def render(self, mode='human'):
        pass
  
    def close(self):
        self.env.close()
