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



class Metaworld_Dataset:
    def __init__(self,json_file_dir,data_dir,images_transform,general_transform) -> None:
        self.csv_file = json_file_dir
        self.data_dir = data_dir
        self.images_transform = images_transform
        self.general_transform = general_transform
        with open(json_file_dir, "r") as read_file:
            data = json.load(read_file)
    


        self.data = data
        self.num_seqs = len(self.data.keys())
        self.seq_len = 200

        self.instructons = {'button-press-topdown-v2':['press the button']}
        self.max = 0


    def __len__(self):
        return (self.num_seqs * self.seq_len)
    
    def __getitem__(self,index):
        #idx = str(idx) 
        seq_num = index//self.seq_len
        idx     = (index - (seq_num*self.seq_len))

        data     = self.data[str(seq_num)]['data'] 
        taskname = self.data[str(seq_num)]['task_name']
        instruct = random.choice(self.instructons[taskname])

        ret = {}
       

        step        = data['step'][idx]
        hand_pos    = data['hand_pos'][idx]
        action      = data['action'][idx]
        reward      = data['reward'][idx]
        state       = data['state'][idx]
    
        images_dir = os.path.join(self.data_dir,'images',taskname,str(seq_num))
        
        images_dirs =  [images_dir+'/'+str(step)+'_corner.png'        ,       
                        images_dir+'/'+str(step)+'_corner2.png'     , 
                        images_dir+'/'+str(step)+'_behindGripper.png',
                        images_dir+'/'+str(step)+'_corner3.png'     , 
                        images_dir+'/'+str(step)+'_topview.png'     
        ]
        step_images = []
        for i in range(len(images_dirs)):
            image = Image.open(images_dirs[i])
            image = self.images_transform(image)
            step_images.append(image)

        action = torch.tensor(action)
        action -= action.min(0, keepdim=True)[0]
        action /= action.max(0, keepdim=True)[0]
        #action[0:3] = (action[0:3]+15)/30
        #action[action == -1] = 2


        ret['image']       = torch.stack(step_images)
        ret['action']      = action
        ret['state']       = torch.tensor(state)
        ret['hand_pos']    = torch.tensor(hand_pos)
        ret['reward']      = torch.tensor(reward)
        ret['caption']     = instruct
        
       
        return ret
    
def process_command(command,tokenizer):
    encoded_command = tokenizer(list([command]), padding=True, truncation=True, max_length=200)
    ret = {}
    for k,v in encoded_command.items():
        empty_tensor = np.zeros((12),dtype=int)
        empty_tensor[0:len(v[0])] =  v[0]
        v[0] =  list(empty_tensor)
        ret[k] =  v
    return ret

def prepare_batch(batch):
    batch['image'] = batch['image'].permute(2,0,1,3,4,5)
    

    return batch

def predict_action(model,images,encoded_command,hand_pos,device):
    with torch.no_grad():

        
        batch = {
                key: torch.tensor(values[0]).unsqueeze(0)
                for key, values in encoded_command.items()
            }


        batch['image']    = images.unsqueeze(0)
        batch['hand_pos'] = torch.tensor(hand_pos).reshape(1,-1)

        batch = {k:v.to(device) for k,v in batch.items()}
        logits  = model(batch)  
        actions = logits.cpu().detach().numpy()
        actions = actions.reshape(-1)
        actions = ((actions*2)-1)


    return actions 

def get_episode(model,taskname,images_transform,tokenizer,command,prob_expert_generate,steps_sampling_ratio,device):
    ml1 = metaworld.ML_1_multi(taskname) # Construct the benchmark, sampling tasks
    #env = ml1.train_classes[task]()  # Create an environment with task `pick_place`
    env = ml1.my_env_s
    task = random.choice(ml1.train_tasks)
    env.set_task(task)  # Set task
    obs = env.reset()  # Reset environment
    policy = SawyerButtonPressTopdownV2Policy(env.main_env_pos)
    episode_dict = defaultdict(list)
    print('Command:',command)
    print('env:',env.file_name)

    for i in range(200):
        hand_pos = policy._parse_obs(obs)['hand_pos'].astype(np.float32)
        expert_a = policy.get_action(obs)

        corner         = env.render(offscreen= True,camera_name='corner')# corner,2,3, corner2, topview, gripperPOV, behindGripper'
        corner2        = env.render(offscreen= True,camera_name='corner2')
        behindGripper  = env.render(offscreen= True,camera_name='behindGripper')
        corner3        = env.render(offscreen= True,camera_name='corner3')
        topview        = env.render(offscreen= True,camera_name='topview')
        

        images = [cv2.cvtColor(corner,cv2.COLOR_RGB2BGR),       
                cv2.cvtColor(corner2,cv2.COLOR_RGB2BGR),      
                cv2.cvtColor(behindGripper,cv2.COLOR_RGB2BGR),
                cv2.cvtColor(corner3,cv2.COLOR_RGB2BGR),      
                cv2.cvtColor(topview,cv2.COLOR_RGB2BGR)      
        ]


        input_imgs = []
        for image in images:
            input_imgs.append(images_transform(Image.fromarray(image, "RGB")))

        input_imgs = torch.stack(input_imgs)


        encoded_command = process_command(command, tokenizer)
        

        if random.random()<prob_expert_generate :
            #expert
            action_to_take = expert_a
        else:
            #model
            action_to_take  = predict_action(model,input_imgs,encoded_command,hand_pos,device)

        if random.random()<steps_sampling_ratio:

            
            #episode_dict['pred_a'].append(a)
            episode_dict['expert_a'].append(list(expert_a))
            episode_dict['image'].append(input_imgs)
            #episode_dict['step'].append(i)
            episode_dict['hand_pos'].append(hand_pos)
            episode_dict['caption'].append(command)
            episode_dict['encoded_command'].append(encoded_command)
            
            
            #episode_dict['reward'].append(reward)
            #episode_dict['state'].append(info['success'])

        obs, reward, done, info = env.step(action_to_take)  # Step the environoment with the sampled random action
        
        if info['success']:
            episode_dict['state'].append(1)
            break


    return episode_dict

class Metaworld_Dataset_live:
    def __init__(self,model,images_transform,tokenizer,prob_expert_generate,CFG) -> None:
        self.model  = model
        self.device = CFG.device
        self.images_transform = images_transform
        self.tokenizer = tokenizer
        self.prob_expert_generate = prob_expert_generate    
        self.steps_sampling_ratio = CFG.steps_sampling_ratio
        f = open(CFG.instructs_file_dir)
        self.instructons = json.load(f)
        self.tasks_names = list(self.instructons.keys())

        self.data  = self.get_data(model,CFG.episodes_per_model) # should be multithreading
        self.dones = float(sum(self.data['state']))


    def __len__(self):
        return len(self.data['expert_a'])
    
    def __getitem__(self,idx):
        #idx = str(idx) 

        #taskname = self.data['taskname'][idx]
        #instruct = random.choice(self.instructons[taskname][0:100])
        ret = {
                key: torch.tensor(values[0])
                for key, values in self.data['encoded_command'][idx].items()
            }

               
        #step        = data['step'][idx]
        hand_pos    = self.data['hand_pos'][idx]
        expert_a    = self.data['expert_a'][idx]
        #pred_a      = data['pred_a'][idx]
        #reward      = data['reward'][idx]
        #state       = data['state'][idx]
    
        

        expert_a = torch.tensor(expert_a)
        expert_a[expert_a>1]  = 1.0  
        expert_a[expert_a<1]  = -1.0 
        expert_a  += 1.0
        expert_a  /= 2.0
       

        ret['image']       = self.data['image'][idx]
        ret['expert_a']    = expert_a
        #ret['pred_a']      = pred_a
        #ret['state']       = self.data['state'][idx]
        ret['hand_pos']    = torch.tensor(hand_pos)
        #ret['reward']      = torch.tensor(reward)
        ret['caption']      = self.data['caption'][idx]
       
       
        return ret

    
    
    def get_data(self,model,episodes):
        
        
        '''
        ret = [get_episode(model,taskname,self.images_transform,self.clip,self.device) for episode_num in tqdm(range(episodes))]
        
        for episode_num in (range(episodes)):
            data_dict[episode_num] = {'task_name':taskname,'data':ret[episode_num]}
        #data_dict= {episode_num:{'task_name':taskname,'data':get_episode(model,taskname,self.images_transform,self.clip,command,self.device)}  for episode_num in tqdm(range(episodes))}
        '''

        data_dict = defaultdict(list)
        taskname = random.choice(self.tasks_names)

        
        for i in tqdm(range(episodes)):
            command = random.choice(self.instructons[taskname][0:100])
            
            episode_dict = get_episode(model,taskname,self.images_transform,self.tokenizer,command,self.prob_expert_generate,self.steps_sampling_ratio,self.device)
            
            for key, data in episode_dict.items():
                data_dict[key] += data

                    
        return data_dict


