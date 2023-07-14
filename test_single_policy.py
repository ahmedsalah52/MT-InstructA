import metaworld
import random
import time
import cv2
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS
from metaworld.policies import * #SawyerBasketballV2Policy
import numpy as np
tasks = ['assembly-v2', 'basketball-v2', 'bin-picking-v2', 'box-close-v2', 'button-press-topdown-v2', 'button-press-topdown-wall-v2', 'button-press-v2', 'button-press-wall-v2', 'coffee-button-v2', 'coffee-pull-v2', 'coffee-push-v2', 'dial-turn-v2', 'disassemble-v2', 'door-close-v2', 'door-lock-v2', 'door-open-v2', 'door-unlock-v2', 'hand-insert-v2', 'drawer-close-v2', 'drawer-open-v2', 'faucet-open-v2', 'faucet-close-v2', 'hammer-v2', 'handle-press-side-v2', 'handle-press-v2', 'handle-pull-side-v2', 'handle-pull-v2', 'lever-pi-v2', 'peg-insert-side-v2', 'pick-place-wall-v2', 'pick-out-of-hole-v2', 'reach-v2', 'push-back-v2', 'push-v2', 'pick-place-v2', 'plate-slide-v2', 'plate-slide-side-v2', 'plate-slide-back-v2', 'plate-slide-back-side-v2', 'peg-unplug-side-v2', 'soccer-v2', 'stick-push-v2', 'stick-pull-v2', 'push-wall-v2', 'reach-wall-v2', 'shelf-place-v2', 'sweep-into-v2', 'sweep-v2', 'window-open-v2', 'window-close-v2']

print(len(tasks))

task = 'button-press-topdown-v2'
ml1 = metaworld.ML_1_multi(task) # Construct the benchmark, sampling tasks
#env = ml1.train_classes[task]()  # Create an environment with task `pick_place`
env = ml1.my_env_s
print(ml1.train_tasks)
#task = random.choice(ml1.train_tasks)
task = ml1.train_tasks[0]

env.set_task(task)  # Set task

obs = env.reset()  # Reset environment
policy = SawyerButtonPressTopdownV2Policy(env.main_env_pos)
for i in range(200):
    a = policy.get_action(obs)
    a[a>0] = 1
    a[a<0] = -1
    
    obs, reward, done, info = env.step(a/2)  # Step the environoment with the sampled random action
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
    key = cv2.waitKey(0)
    if key == ord('q'): break

cv2.destroyAllWindows()

env.close()