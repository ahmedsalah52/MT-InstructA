import cv2
import numpy as np
from torch.utils.data import Dataset
import os
import json
from stable_baselines3 import SAC
import random
import torch
import uuid
import meta_env
from PIL import Image 
class temp_dataset(Dataset):
    def __init__(self):
        self.data = []
        for i in range(100):
            self.data.append(i)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        ret = {}
        ret['images']         = torch.zeros((5,3,224,224)).to(torch.float32)
        ret['hand_pos']       = torch.zeros(8).to(torch.float32)
        ret['action']         = torch.zeros(4).to(torch.float32)
        
        ret['instruction']    = "empty instruction"

        return ret


class MW_dataset(Dataset):
    def __init__(self,preprocess,dataset_dict_dir,dataset_dir,tasks_commands,total_data_len):
        self.data_dict = json.load(open(dataset_dict_dir))
        self.dataset_dir = dataset_dir
        self.tasks_commands = tasks_commands
        self.total_data_len = total_data_len
        self.preprocess = preprocess
        self.load_data()
    def load_data(self):
        self.tasks = list(self.data_dict.keys())
        self.data = []
        for task in self.tasks:
            for epi in range(len(self.data_dict[task])):
                for s in range(len(self.data_dict[task][epi])):
                    step = self.data_dict[task][epi][s]
                    step['instruction'] = random.choice(self.tasks_commands[task])
                    self.data.append(step)
        
        #random.shuffle(self.data)
        #self.data = self.data[0:self.total_data_len]
    def get_stats(self):
        table=[]
        total_success_rate = 0
        for task_name , episodes in self.data_dict.items():
            task_success = 0
            for episode in episodes:
                task_success += episode[-1]['success']
            
            task_success_rate = float(task_success) / len(episodes)
            total_success_rate += task_success_rate
            table.append([task_name,task_success_rate])
        table.append(['total_success_rate',total_success_rate/len(self.data_dict.items())])
        return table
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        step_data = self.data[idx]
        images_dir = step_data['images_dir']
        images = [self.preprocess(Image.open(os.path.join(self.dataset_dir,dir))) for dir in images_dir]
        ret = {}
        ret['images']   = torch.stack(images)
        ret['hand_pos'] = torch.tensor(np.concatenate((step_data['obs'][0:4],step_data['obs'][18:22]),axis =0)).to(torch.float32)
        ret['action']      = torch.tensor(step_data['action'])
        ret['instruction'] = step_data['instruction']
        return ret
    
class Generate_data():
    def __init__(self,meta_env,data_dir,agents_dir,tasks,total_num_steps,agents_dict_dir,agent_levels):
        self.agent_levels = agent_levels
        self.data_dir = data_dir
        self.agents_dir = agents_dir
        self.tasks = tasks
        self.total_num_steps = total_num_steps
        self.max_task_steps = total_num_steps//len(tasks)
        self.task_poses = ['Right','Mid','Left']
        self.agents_dict = json.load(open(agents_dict_dir))
        self.meta_env = meta_env
        self.agents_levels_step = 10000
    def generate_data(self):
        data_dict = {}
        for task in self.tasks:
            data_dict[task] = self.generate_task_data(task)

        return data_dict

    def generate_task_data(self,task):
        task_data = []
        for agent_level in range(self.agent_levels):
            for pos in [0,1,2]:
                env   = self.meta_env(task,pos,save_images=True,process = 'train',wandb_log = False,general_model=True)
                agent = self.get_agent(env,task,pos,agent_level)
                task_data += self.generate_pos_data(env,agent,task,pos)
            
        return task_data
    

    def get_agent(self,env,taskname,pos,agent_level):
        run_name   = f'{taskname}_{self.task_poses[pos]}_ID{self.agents_dict[taskname][str(pos)]["id"]}'
        best_model = self.agents_dict[taskname][str(pos)]['best_model']
        load_from  = int( (best_model//(10000 *self.agent_levels)) * (agent_level+1) * 10000)
        print(f"loading agent for task {taskname} step {load_from}")
        load_path = os.path.join(self.agents_dir,run_name, f"{run_name}_{load_from}_steps")

        return SAC.load(load_path,env)


    def generate_pos_data(self,env,agent,task,pos):
        max_steps = self.max_task_steps//(3*self.agent_levels)

        episodes = []
        total_steps = 0
        id_num = 0
        while total_steps < max_steps:
            id_num += 1
            step_num = 0
            success = False
            done = False
            prev_obs , info = env.reset()
            prev_images_obs = self.save_images(info['images'],task,pos,id_num,step_num)
            episode = [] 
            while 1:
                a , _states = agent.predict(prev_obs, deterministic=True)
                obs, reward, done,success, info = env.step(a) 
                episode.append({'obs':prev_obs.tolist(),'action':a.tolist(),'reward':reward,'success':success,'images_dir':prev_images_obs})
                
                if (success or done): break 
                prev_obs = obs
                prev_images_obs = self.save_images(info['images'],task,pos,id_num,step_num)
                step_num+=1

            total_steps+=step_num
            episodes.append(episode[:])
        return episodes   
    def save_images(self,images,taskname,pos,id_num,step_num):
        ret = []
        for i in range(len(images)):
            ret_dir = os.path.join(taskname,str(pos))
            img_name = f'{id_num}_{step_num}_{i}.jpg'

            save_dir = os.path.join(self.data_dir,ret_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            img_dir = os.path.join(save_dir,img_name)
            cv2.imwrite(img_dir,cv2.cvtColor(images[i],cv2.COLOR_RGB2BGR))
            
            ret.append(os.path.join(ret_dir,img_name))
            
        return ret
    
