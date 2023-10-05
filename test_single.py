import metaworld
import random
import cv2
import numpy as np
from  metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS


class task_manager():
    def __init__(self,taskname,pos=None,variant=None,multi = True,general_model=False):
        self.env_args = {'main_pos_index':pos , 'task_variant':{'variant':variant,'general_model':general_model}}
        self.task_name = taskname
        self.multi = multi

    def reset(self):
        if self.multi:
            ml1 = metaworld.ML_1_multi(self.task_name,self.env_args)
        else:
            ml1 = metaworld.ML_1_single(self.task_name)

        self.task_ = random.choice(ml1.train_tasks)
        self.env = ml1.my_env_s
        self.env.set_task(self.task_)  # Set task
        return self.env

tasks =  ['assembly-v2', 'box-close-v2', 'button-press-topdown-v2',  'button-press-v2', 'coffee-button-v2', 'coffee-pull-v2', 'coffee-push-v2', 'disassemble-v2', 'door-lock-v2', 'door-open-v2', 'door-unlock-v2','drawer-close-v2', 'drawer-open-v2', 'faucet-open-v2', 'faucet-close-v2','handle-press-v2',  'handle-pull-v2',  'soccer-v2',   'shelf-place-v2',    'window-open-v2', 'window-close-v2']
print(len(tasks))
full_tasks = ['assembly-v2', 'basketball-v2', 'bin-picking-v2', 'box-close-v2', 'button-press-topdown-v2', 'button-press-topdown-wall-v2', 'button-press-v2', 'button-press-wall-v2', 'coffee-button-v2', 'coffee-pull-v2', 'coffee-push-v2', 'dial-turn-v2', 'disassemble-v2', 'door-close-v2', 'door-lock-v2', 'door-open-v2', 'door-unlock-v2', 'hand-insert-v2', 'drawer-close-v2', 'drawer-open-v2', 'faucet-open-v2', 'faucet-close-v2', 'hammer-v2', 'handle-press-side-v2', 'handle-press-v2', 'handle-pull-side-v2', 'handle-pull-v2', 'lever-pull-v2', 'peg-insert-side-v2', 'pick-place-wall-v2', 'pick-out-of-hole-v2', 'reach-v2', 'push-back-v2', 'push-v2', 'pick-place-v2', 'plate-slide-v2', 'plate-slide-side-v2', 'plate-slide-back-v2', 'plate-slide-back-side-v2', 'peg-unplug-side-v2', 'soccer-v2', 'stick-push-v2', 'stick-pull-v2', 'push-wall-v2', 'reach-wall-v2', 'shelf-place-v2', 'sweep-into-v2', 'sweep-v2', 'window-open-v2', 'window-close-v2', 'sample']
dataset_tasks = ['button-press-topdown-v2', 'button-press-v2', 'door-lock-v2', 'door-unlock-v2', 'door-open-v2', 'door-close-v2', 'drawer-open-v2', 'drawer-close-v2', 'window-open-v2', 'window-close-v2', 'faucet-open-v2', 'faucet-close-v2', 'handle-press-v2', 'coffee-button-v2']
for taskname in dataset_tasks:#tasks:#ALL_V2_ENVIRONMENTS.keys():
    #taskname = 'sweep-v2' #'box-close-v2' #'soccer-v2'#'button-press-topdown-v2' #'door-lock-v2' 
    print(taskname)
    multi = True
    pos = 0
    variant = None # ['push_v2','button_press_topdown','door_lock']

    task_man = task_manager(taskname,pos=pos,variant=variant,multi=multi)


    env = task_man.reset()
    print(taskname)
    import torch
    #env.set_task(ml1.train_tasks[0])  # Set task
    obs = env.reset()  # Reset environment
    x = y = z = g = 0
    #policy = SawyerBasketballV2Policy(env.x_shift)
    for i in range(500):
        #a = policy.get_action(obs)
        a  = np.array([x,y,z,g])
        obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
        print(obs)
        print()
        print(np.concatenate((obs[0:4],obs[18:22]),axis =0))

        
        #print(i,'-',reward,' - ',obs[:3] )
        print(i,'action ',a,' reward ' ,round(reward,2),' state ',info['success'])
        #print(read_obs)

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
