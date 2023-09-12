import metaworld
import random
import time
import cv2
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS
from metaworld.policies import * #SawyerBasketballV2Policy
import numpy as np
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS_multi
from metaworld.envs.build_random_envs import Multi_task_env

tasks = ['assembly-v2', 'basketball-v2', 'bin-picking-v2', 'box-close-v2', 'button-press-topdown-v2', 'button-press-topdown-wall-v2', 'button-press-v2', 'button-press-wall-v2', 'coffee-button-v2', 'coffee-pull-v2', 'coffee-push-v2', 'dial-turn-v2', 'disassemble-v2', 'door-close-v2', 'door-lock-v2', 'door-open-v2', 'door-unlock-v2', 'hand-insert-v2', 'drawer-close-v2', 'drawer-open-v2', 'faucet-open-v2', 'faucet-close-v2', 'hammer-v2', 'handle-press-side-v2', 'handle-press-v2', 'handle-pull-side-v2', 'handle-pull-v2', 'lever-pi-v2', 'peg-insert-side-v2', 'pick-place-wall-v2', 'pick-out-of-hole-v2', 'reach-v2', 'push-back-v2', 'push-v2', 'pick-place-v2', 'plate-slide-v2', 'plate-slide-side-v2', 'plate-slide-back-v2', 'plate-slide-back-side-v2', 'peg-unplug-side-v2', 'soccer-v2', 'stick-push-v2', 'stick-pull-v2', 'push-wall-v2', 'reach-wall-v2', 'shelf-place-v2', 'sweep-into-v2', 'sweep-v2', 'window-open-v2', 'window-close-v2']

print(len(tasks))

taskname = 'button-press-topdown-v2' #'coffee-pull-v2'#,  #'coffee-button-v2'

class multi_task_V2(ALL_V2_ENVIRONMENTS_multi[taskname],Multi_task_env):
    def __init__(self,main_pos_index=None , task_variant = None) -> None:
        Multi_task_env.__init__(self)
        self.main_pos_index = main_pos_index
        self.task_variant = task_variant
        self.reset_variant()  

    def reset_variant(self):
        ALL_V2_ENVIRONMENTS_multi[taskname].__init__(self)
        self._freeze_rand_vec = False
        self._set_task_called = True
        self._partially_observable = False #taskname not in ['assembly-v2', 'coffee-pull-v2', 'coffee-push-v2']
env = multi_task_V2(main_pos_index=1)


obs = env.reset()  # Reset environment
policy = SawyerButtonPressTopdownV2Policy()
for i in range(200):
    a = policy.get_action(obs)
    #a[a>1] = 1
    #a[a<-1] = -1
    
    obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
    print(i,'action ',a,' reward ' ,round(reward,2),' state ',info['success'])
    x = y = z = g = 0
    #env.render() 
    corner         = env.render(offscreen= True,camera_name='corner')# corner,2,3, corner2, topview, gripperPOV, behindGripper'
    corner2        = env.render(offscreen= True,camera_name='corner2')
    behindGripper  = env.render(offscreen= True,camera_name='behindGripper')
    corner3        = env.render(offscreen= True,camera_name='corner3')
    topview        = env.render(offscreen= True,camera_name='topview')
    
    topview        = cv2.rotate(topview, cv2.ROTATE_180)
    behindGripper  = cv2.rotate(behindGripper, cv2.ROTATE_180)

    all     = cv2.hconcat([corner,corner2,corner3,topview])

    behindGripper  = cv2.resize(behindGripper, (all.shape[1],int(behindGripper.shape[0] * all.shape[1]/all.shape[0])))

    final_frame = cv2.vconcat([all,behindGripper])
    final_frame = cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)
    final_frame = cv2.resize(final_frame, (final_frame.shape[1]//2,final_frame.shape[0]//2))
    cv2.imshow('show',final_frame)
    key = cv2.waitKey(100)
    if key == ord('q'): break

cv2.destroyAllWindows()

env.close()