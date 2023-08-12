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
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS
from metaworld.policies import  SawyerAssemblyV2Policy,SawyerBasketballV2Policy
import numpy as np
tasks =  ['assembly-v2', 'basketball-v2','box-close-v2', 'button-press-topdown-v2', 'button-press-topdown-wall-v2', 'button-press-v2', 'button-press-wall-v2', 'coffee-button-v2', 'coffee-pull-v2', 'coffee-push-v2', 'disassemble-v2', 'door-lock-v2', 'door-open-v2', 'door-unlock-v2','drawer-close-v2', 'drawer-open-v2', 'faucet-open-v2', 'faucet-close-v2','handle-press-v2',  'handle-pull-v2','lever-pull-v2', 'pick-place-wall-v2',  'reach-v2', 'push-back-v2', 'push-v2', 'pick-place-v2', 'plate-slide-v2',  'soccer-v2', 'push-wall-v2', 'reach-wall-v2', 'shelf-place-v2',    'window-open-v2', 'window-close-v2']

print(len(tasks))

task = 'door-lock-v2'
ml1 = metaworld.ML_1_multi(task) # Construct the benchmark, sampling tasks
#env = ml1.train_classes[task]()  # Create an environment with task `pick_place`
env = ml1.my_env_s
task = random.choice(ml1.train_tasks)
env.set_task(task)  # Set task

obs = env.reset()  # Reset environment
x = y = z = g = 0
#policy = SawyerBasketballV2Policy(env.x_shift)
for i in range(500):
    #a = policy.get_action(obs)
    a  = np.array([x,y,z,g])
    obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
    print(i,'-',reward,' - ',obs[:3] ,additional_reward(env.x_shift,obs))
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