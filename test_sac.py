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
from meta_env import meta_env,meta_Callback  
    
    
def main():
    model = SAC.load("test")
    env = meta_env("button-press-topdown-v2")
    check_env(env)
    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        img = env.render()
        cv2.imshow('env',img)
        cv2.waitKey(1)
        print(action,reward,info['success'])
        if terminated or truncated:
            obs, info = env.reset()
            break
    cv2.destroyAllWindows()

main()