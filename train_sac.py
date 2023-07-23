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
from meta_env import meta_env

def main():

    env = meta_env("button-press-topdown-v2")


    check_env(env)

    model = SAC("MlpPolicy", env, verbose=1,buffer_size=1000,batch_size=32)
    model.learn(total_timesteps=10000, log_interval=4)
    #model.save("test")

    #del model # remove to demonstrate saving and loading

    model = SAC.load("test")

    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()



main()