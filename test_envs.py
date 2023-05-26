import metaworld
import random
import time
import cv2
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS
# removed 'stick-push-v2', 'stick-pull-v2','sweep-into-v2', 'sweep-v2',
#
tasks =  ['assembly-v2', 'basketball-v2', 'bin-picking-v2', 'box-close-v2', 'button-press-topdown-v2', 'button-press-topdown-wall-v2', 'button-press-v2', 'button-press-wall-v2', 'coffee-button-v2', 'coffee-pull-v2', 'coffee-push-v2', 'dial-turn-v2', 'disassemble-v2','door-close-v2', 'door-lock-v2', 'door-open-v2', 'door-unlock-v2','drawer-close-v2', 'drawer-open-v2', 'faucet-open-v2', 'faucet-close-v2','handle-press-v2',  'handle-pull-v2','lever-pull-v2', 'pick-place-wall-v2',  'reach-v2', 'push-back-v2', 'push-v2', 'pick-place-v2', 'plate-slide-v2',  'soccer-v2', 'push-wall-v2', 'reach-wall-v2', 'shelf-place-v2',    'window-open-v2', 'window-close-v2']
print(len(tasks))
counter = 1
for task in tasks:
    print(task)
    ml1 = metaworld.ML_1_multi(task) # Construct the benchmark, sampling tasks
    #env = ml1.train_classes[task]()  # Create an environment with task `pick_place`
    env = ml1.my_env_s
    task = random.choice(ml1.train_tasks)
    env.set_task(task)  # Set task

    obs = env.reset()  # Reset environment
    x = y = z = g = 0

    a = env.action_space.sample()  # Sample an action
    obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
    counter -=1 