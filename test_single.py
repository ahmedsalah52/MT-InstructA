def additional_reward(x_shift,obs):
    hand_pos = obs[:3]
    x = hand_pos[0]

    delta_x = abs(x_shift - x)
    
    reward = (0.7 - delta_x)/0.7


    return reward

import metaworld
import random
import time
import cv2
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS_multi
from metaworld.policies import  SawyerAssemblyV2Policy,SawyerBasketballV2Policy
import numpy as np
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_coffee_push_v2 import  SawyerCoffeePushEnvV2
from metaworld.envs.build_random_envs import Multi_task_env

tasks =  ['assembly-v2', 'basketball-v2','box-close-v2', 'button-press-topdown-v2', 'button-press-topdown-wall-v2', 'button-press-v2', 'button-press-wall-v2', 'coffee-button-v2', 'coffee-pull-v2', 'coffee-push-v2', 'disassemble-v2', 'door-lock-v2', 'door-open-v2', 'door-unlock-v2','drawer-close-v2', 'drawer-open-v2', 'faucet-open-v2', 'faucet-close-v2','handle-press-v2',  'handle-pull-v2','lever-pull-v2', 'pick-place-wall-v2',  'reach-v2', 'push-back-v2', 'push-v2', 'pick-place-v2', 'plate-slide-v2',  'soccer-v2', 'push-wall-v2', 'reach-wall-v2', 'shelf-place-v2',    'window-open-v2', 'window-close-v2']

print(len(tasks))

task = 'button-press-topdown-v2' #'coffee-pull-v2'#,  #'coffee-button-v2'
#ml1 = metaworld.ML_1_multi(task) # Construct the benchmark, sampling tasks
#env = ml1.train_classes[task]()  # Create an environment with task `pick_place`
#env = ml1.my_env_s

class multi_task_V2(ALL_V2_ENVIRONMENTS_multi[task],Multi_task_env):
    def __init__(self,main_pos_index=None , task_variant = None) -> None:
        Multi_task_env.__init__(self)
        self.main_pos_index = main_pos_index
        self.task_variant = task_variant

    def reset_variant(self):
        ALL_V2_ENVIRONMENTS_multi[task].__init__(self)
        self._freeze_rand_vec = False
        self._set_task_called = True
        self._partially_observable = False# task not in ['assembly-v2', 'coffee-pull-v2', 'coffee-push-v2']

env = multi_task_V2(main_pos_index=1)
env.reset_variant()
 #
#env.set_task(ml1.train_tasks[0])  # Set task
obs = env.reset()  # Reset environment
x = y = z = g = 0
#policy = SawyerBasketballV2Policy(env.x_shift)
for i in range(500):
    #a = policy.get_action(obs)
    a  = np.array([x,y,z,g])
    obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
    read_obs = {
            'hand_pos': obs[:3],
            'gripper': obs[3],
            'mug_pos': obs[4:7],
            'goal_xy': obs[-3:-1],
        #    'unused_info_1': obs[7:-3],
        #    'unused_info_2': obs[-1],
        }
    #print(i,'-',reward,' - ',obs[:3] )
    print(i,'action ',a,' reward ' ,round(reward,2),' state ',info['success'],'pos ',env.main_pos_index)
    print(read_obs)

    x = y = z = g = 0
    #env.render() 
    corner         = env.render(offscreen= True,camera_name='corner')# corner,2,3, corner2, topview, gripperPOV, behindGripper'
    corner2        = env.render(offscreen= True,camera_name='corner2')
    behindGripper  = env.render(offscreen= True,camera_name='behindGripper')
    corner3        = env.render(offscreen= True,camera_name='corner3')
    topview        = env.render(offscreen= True,camera_name='topview')
    
    topview        = cv2.rotate(topview, cv2.ROTATE_180)
    behindGripper  = cv2.rotate(behindGripper, cv2.ROTATE_180)
    behindGripper  = cv2.resize(behindGripper, (256,256))
    all     = cv2.hconcat([corner,corner2,corner3,topview])

    behindGripper  = cv2.resize(behindGripper, (all.shape[1],int(behindGripper.shape[0] * all.shape[1]/all.shape[0])))

    final_frame = cv2.vconcat([all,behindGripper])
    final_frame = cv2.resize(final_frame,(1024,1024))
    final_frame = cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('show',final_frame)

    #print(a,reward)
    key = cv2.waitKey(0)
    if key & 0xFF == ord('w'):
        y = 1
    if key & 0xFF == ord('s'):
        y = -1
    if key & 0xFF == ord('d'):
        x = -1
    if key & 0xFF == ord('a'):
        x = 1
    if key & 0xFF == ord('i'):
        z = 1
    if key & 0xFF == ord('k'):
        z = -1
    if key & 0xFF == ord('n'):
        g = 1
    if key & 0xFF == ord('m'):
        g = -1
    if key & 0xFF == ord('q'):
        break
    #time.sleep(1/10)

cv2.destroyAllWindows()

env.close()