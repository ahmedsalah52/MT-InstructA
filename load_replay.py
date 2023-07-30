
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

from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC
from meta_env import meta_env,meta_Callback
#from stable_baselines3.common.callbacks import CheckpointCallback
task_name  = 'button-press-topdown-v2' #'coffee-button-v2'# 'button-press-topdown-v2'  #'coffee-button-v2'
i = 19
env = meta_env(task_name,True)

model = SAC("MlpPolicy", env, verbose=1,buffer_size=10,batch_size=512,learning_rate=0.0001) #,replay_buffer_class = HerReplayBuffer,replay_buffer_kwargs={ 'n_envs':10}

model.load_replay_buffer(os.path.join("buffers",task_name,str(i)))
renders = model.replay_buffer.renders
succ = model.replay_buffer.success
print(succ.shape)
print(model.replay_buffer.file_order.shape)
print(renders.shape)
for i in range(len(renders)):
    cv2.imshow('test ',cv2.resize(renders[i][0,0,:,:,:],(512,512)))
    if succ[i,0,0]:
        print('success')
        key = cv2.waitKey(0)
    else:
        key = cv2.waitKey(1)
    if key == ord('q'): break

cv2.destroyAllWindows()