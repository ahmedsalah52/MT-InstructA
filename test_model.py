

from torchvision import transforms
import itertools

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import clip
from train_utils.model import Transformer,Policy,AvgMeter
from train_utils.datasets import Metaworld_Dataset

import metaworld
import random
import time
import cv2
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS
from metaworld.policies import SawyerBasketballV2Policy
import numpy as np
from metaworld.policies import * #SawyerBasketballV2Policy

from PIL import Image
from train_utils.datasets import predict_action

class CFG():
    device = 'cpu' #"cuda" if torch.cuda.is_available() else "cpu"
    dataset_train_per = 0.8
    lr   = 1e-4
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 5
    batch_size = 24
    workers = 1
def predict_action_old(model,images,text,preprocess,tokenize):
    input_imgs = []
    for image in images:
        input_imgs.append(preprocess(Image.fromarray(image, "RGB")))

    input_imgs = torch.stack(input_imgs).unsqueeze(0).to(CFG.device)
    text = tokenize(text).to(CFG.device)
    print(input_imgs.shape,text.shape)
    batch = {'image':input_imgs,'caption':text}
    logits = model(batch)  
    actions = []
    for i in range(4):
        actions.append(torch.argmax(logits[i]).cpu().detach().numpy())
    
    return np.array(actions)


def main():

    seq_length = 385 #(5 imgs + 1 text encoded to * 512) = 384*8  + 1*8 pos emp
    emp_length = 8
    
    policy_head = Transformer(
    dim_model=8,
    num_heads=8,
    num_encoder_layers=4,
    dropout_p=0.1,
    seq_length = seq_length,
    emp_length = emp_length,
    num_actions= 4,
    variations_per_action = 3,
    device=CFG.device
    ).to(CFG.device) 
    clip_model , preprocess = clip.load("ViT-B/32", device=CFG.device)
    
    policy = Policy(language_img_model=clip_model,
                    policy_head=policy_head,
                    seq_length=385,
                    emp_length=8,
                    device=CFG.device
                    )
    policy.policy_head.load_state_dict(torch.load('best_19.pt'))
        
    task = 'button-press-topdown-v2'
    ml1 = metaworld.ML_1_multi(task) # Construct the benchmark, sampling tasks
    #env = ml1.train_classes[task]()  # Create an environment with task `pick_place`
    env = ml1.my_env_s
    task = random.choice(ml1.train_tasks)
    env.set_task(task)  # Set task

    obs = env.reset()  # Reset environment
    a = np.array([0,0,0,0])

    x = y = z = g = 0
    expert_policy = SawyerButtonPressTopdownV2Policy(env.main_env_pos)

    for i in range(500):
        hand_pos = expert_policy._parse_obs(obs)['hand_pos'].astype(np.float32)


        corner         = env.render(offscreen= True,camera_name='corner')# corner,2,3, corner2, topview, gripperPOV, behindGripper'
        corner2        = env.render(offscreen= True,camera_name='corner2')
        behindGripper  = env.render(offscreen= True,camera_name='behindGripper')
        corner3        = env.render(offscreen= True,camera_name='corner3')
        topview        = env.render(offscreen= True,camera_name='topview')
        
        #a = predict_action(policy,[corner,corner2,behindGripper,corner3,topview],'press the button',preprocess,clip.tokenize)
        a , _ = predict_action(policy,[corner,corner2,behindGripper,corner3,topview],'press the button',hand_pos,preprocess,clip.tokenize,CFG.device)


        topview        = cv2.rotate(topview, cv2.ROTATE_180)
        behindGripper  = cv2.rotate(behindGripper, cv2.ROTATE_180)
        behindGripper  = cv2.resize(behindGripper, (256,256))
        all     = cv2.hconcat([corner,corner2,corner3,topview])

        behindGripper  = cv2.resize(behindGripper, (all.shape[1],int(behindGripper.shape[0] * all.shape[1]/all.shape[0])))

        final_frame = cv2.vconcat([all,behindGripper])
        cv2.imshow('show',cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR))

        #print(a,reward)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        #time.sleep(1/10)
        obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
        print(a)
    cv2.destroyAllWindows()

    env.close()

main()