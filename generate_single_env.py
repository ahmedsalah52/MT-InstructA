import metaworld
import random
import time
import cv2
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS
from metaworld.policies import * #SawyerBasketballV2Policy
import csv
import numpy as np
from collections import defaultdict
import json
import os
tasks = ['assembly-v2', 'basketball-v2', 'bin-picking-v2', 'box-close-v2', 'button-press-topdown-v2', 'button-press-topdown-wall-v2', 'button-press-v2', 'button-press-wall-v2', 'coffee-button-v2', 'coffee-pull-v2', 'coffee-push-v2', 'dial-turn-v2', 'disassemble-v2', 'door-close-v2', 'door-lock-v2', 'door-open-v2', 'door-unlock-v2', 'hand-insert-v2', 'drawer-close-v2', 'drawer-open-v2', 'faucet-open-v2', 'faucet-close-v2', 'hammer-v2', 'handle-press-side-v2', 'handle-press-v2', 'handle-pull-side-v2', 'handle-pull-v2', 'lever-pi-v2', 'peg-insert-side-v2', 'pick-place-wall-v2', 'pick-out-of-hole-v2', 'reach-v2', 'push-back-v2', 'push-v2', 'pick-place-v2', 'plate-slide-v2', 'plate-slide-side-v2', 'plate-slide-back-v2', 'plate-slide-back-side-v2', 'peg-unplug-side-v2', 'soccer-v2', 'stick-push-v2', 'stick-pull-v2', 'push-wall-v2', 'reach-wall-v2', 'shelf-place-v2', 'sweep-into-v2', 'sweep-v2', 'window-open-v2', 'window-close-v2']

print(len(tasks))

images_dir = '/datasets/metaworld/images'
json_file = '/datasets/metaworld/single_env.json'
episode_num = 0
data_dict = {}
for i in range(1000):
    for taskname in ['button-press-topdown-v2']:
        sample_dir = os.path.join(images_dir,taskname,str(episode_num))
        os.system('mkdir '+sample_dir)

        ml1 = metaworld.ML_1_multi(taskname) # Construct the benchmark, sampling tasks
        #env = ml1.train_classes[task]()  # Create an environment with task `pick_place`
        env = ml1.my_env_s
        task = random.choice(ml1.train_tasks)
        env.set_task(task)  # Set task

        obs = env.reset()  # Reset environment
        policy = SawyerButtonPressTopdownV2Policy(env.main_env_pos)
        episode_dict = defaultdict(list)
        prev_action = np.zeros(env.action_space.shape)
        max_a = 0
        for i in range(200):
            a = policy.get_action(obs)
            a[a>0] = 1
            a[a<0] = -1
            obs, reward, done, info = env.step(a/2)  # Step the environoment with the sampled random action
            print(i,'action ',a,' reward ' ,round(reward,2),' state ',info['success'])
            corner         = env.render(offscreen= True,camera_name='corner')# corner,2,3, corner2, topview, gripperPOV, behindGripper'
            corner2        = env.render(offscreen= True,camera_name='corner2')
            behindGripper  = env.render(offscreen= True,camera_name='behindGripper')
            corner3        = env.render(offscreen= True,camera_name='corner3')
            topview        = env.render(offscreen= True,camera_name='topview')
            
            
            episode_dict['step'].append(i)
            episode_dict['prev_action'].append(list(prev_action))
            episode_dict['action'].append(list(a))
            episode_dict['reward'].append(reward)
            episode_dict['state'].append(info['success'])

            cv2.imwrite(sample_dir+'/'+str(i)+'_corner.png'       ,cv2.cvtColor(corner,cv2.COLOR_RGB2BGR))
            cv2.imwrite(sample_dir+'/'+str(i)+'_corner2.png'      ,cv2.cvtColor(corner2,cv2.COLOR_RGB2BGR))
            cv2.imwrite(sample_dir+'/'+str(i)+'_behindGripper.png',cv2.cvtColor(behindGripper,cv2.COLOR_RGB2BGR))
            cv2.imwrite(sample_dir+'/'+str(i)+'_corner3.png'      ,cv2.cvtColor(corner3,cv2.COLOR_RGB2BGR))
            cv2.imwrite(sample_dir+'/'+str(i)+'_topview.png'      ,cv2.cvtColor(topview,cv2.COLOR_RGB2BGR))
            
            prev_action    = a
        
        data_dict[episode_num] = {'task_name':taskname,'data':episode_dict}
        episode_num +=1

with open(json_file, "w") as write_file:
    json.dump(data_dict, write_file) 
