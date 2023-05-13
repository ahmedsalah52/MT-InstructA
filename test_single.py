import metaworld
import random
import time
import cv2
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS

#['assembly-v2', 'basketball-v2', 'bin-picking-v2', 'box-close-v2', 'button-press-topdown-v2', 'button-press-topdown-wall-v2', 'button-press-v2', 'button-press-wall-v2', 'coffee-button-v2', 'coffee-pull-v2', 'coffee-push-v2', 'dial-turn-v2', 'disassemble-v2', 'door-close-v2', 'door-lock-v2', 'door-open-v2', 'door-unlock-v2', 'hand-insert-v2', 'drawer-close-v2', 'drawer-open-v2', 'faucet-open-v2', 'faucet-close-v2', 'hammer-v2', 'handle-press-side-v2', 'handle-press-v2', 'handle-pull-side-v2', 'handle-pull-v2', 'lever-pi-v2', 'peg-insert-side-v2', 'pick-place-wall-v2', 'pick-out-of-hole-v2', 'reach-v2', 'push-back-v2', 'push-v2', 'pick-place-v2', 'plate-slide-v2', 'plate-slide-side-v2', 'plate-slide-back-v2', 'plate-slide-back-side-v2', 'peg-unplug-side-v2', 'soccer-v2', 'stick-push-v2', 'stick-pull-v2', 'push-wall-v2', 'reach-wall-v2', 'shelf-place-v2', 'sweep-into-v2', 'sweep-v2', 'window-open-v2', 'window-close-v2']



task = 'sample'
ml1 = metaworld.ML_1_multi(task) # Construct the benchmark, sampling tasks
print('before')
#env = ml1.train_classes[task]()  # Create an environment with task `pick_place`
env = ml1.my_env_s
print('after')
task = random.choice(ml1.train_tasks)
print(task)
env.set_task(task)  # Set task

obs = env.reset()  # Reset environment
x = y = z = g = 0

for i in range(100):
    a = env.action_space.sample()  # Sample an action
    a[0] = x
    a[1] = y
    a[2] = z
    a[3] = g
    obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
    print(reward)
    x = y = z = g = 0
    #env.render() 
    corner         = env.render(offscreen= True,camera_name='corner')# corner,2,3, corner2, topview, gripperPOV, behindGripper'
    corner2        = env.render(offscreen= True,camera_name='corner2')
    behindGripper  = env.render(offscreen= True,camera_name='behindGripper')
    corner3        = env.render(offscreen= True,camera_name='corner3')
    topview        = env.render(offscreen= True,camera_name='topview')
    
    all     = cv2.hconcat([corner,corner2,corner3,cv2.flip(behindGripper, 0),topview])

    cv2.imshow('show',cv2.cvtColor(all, cv2.COLOR_RGB2BGR))
    #print(a,reward)
    key = cv2.waitKey(0)
    if key & 0xFF == ord('w'):
        y = 1
    if key & 0xFF == ord('s'):
        y = -1
    if key & 0xFF == ord('d'):
        x = 1
    if key & 0xFF == ord('a'):
        x = -1
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