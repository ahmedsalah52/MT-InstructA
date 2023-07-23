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


from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC

class meta_env(Env):
    def __init__(self,taskname) -> None:
        super().__init__()
        ml1 = metaworld.ML_1_multi(taskname) # Construct the benchmark, sampling tasks
        self.env = ml1.my_env_s
        task = random.choice(ml1.train_tasks)
        self.env.set_task(task)  # Set task
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
    def reset(self):
        return self.env.reset()  # Reset environment
        
    
    def render(self, mode='human'):
        print('render')
        return Image.fromarray(self.main_frame) 
    def close (self):
        pass



def get_episode(model,taskname):
    
    for i in range(200):
        hand_pos = policy._parse_obs(obs)['hand_pos'].astype(np.float32)
        #expert_a = policy.get_action(obs)
        a = model(obs)
        corner         = env.render(offscreen= True,camera_name='corner')# corner,2,3, corner2, topview, gripperPOV, behindGripper'
        corner2        = env.render(offscreen= True,camera_name='corner2')
        behindGripper  = env.render(offscreen= True,camera_name='behindGripper')
        corner3        = env.render(offscreen= True,camera_name='corner3')
        topview        = env.render(offscreen= True,camera_name='topview')
        

        images = [cv2.cvtColor(corner,cv2.COLOR_RGB2BGR),       
                cv2.cvtColor(corner2,cv2.COLOR_RGB2BGR),      
                cv2.cvtColor(behindGripper,cv2.COLOR_RGB2BGR),
                cv2.cvtColor(corner3,cv2.COLOR_RGB2BGR),      
                cv2.cvtColor(topview,cv2.COLOR_RGB2BGR)      
        ]
        episode_dict['images'] = process_imgs(images)
        episode_dict['action'] = a
        episode_dict['hand_pos'] = hand_pos
        episode_dict['action'] = a
        
        

        obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
        
        if info['success']:
            episode_dict['state'].append(1)
            break


    return episode_dict
