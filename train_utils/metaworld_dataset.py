import cv2
import numpy as np
from torch.utils.data import Dataset
import os
import json
from stable_baselines3 import SAC
import random
import torch
import uuid

     
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
    def __init__(self,tasks_commands,preprocess,total_data_len):
        self.tasks_commands =  tasks_commands

        #self.load_data(dataset_dict)
        self.preprocess = preprocess
        self.total_data_len = total_data_len
    
    def load_data(self,dataset_dict):
        self.data_dict = dataset_dict
        self.tasks = list(self.data_dict.keys())
        self.data = []
        for task in self.tasks:
            for epi in range(len(self.data_dict[task])):
                for s in range(len(self.data_dict[task][epi])):
                    step = self.data_dict[task][epi][s]
                    step['instruction'] = random.choice(self.tasks_commands[task])
                    self.data.append(step)
        
        random.shuffle(self.data)
        self.data = self.data[0:self.total_data_len]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        step_data = self.data[idx]
        images_dir = step_data['images_dir']
        images = [self.preprocess(cv2.imread(dir)) for dir in images_dir]
        ret = {}
        ret['images']   = torch.stack(images)
        ret['hand_pos'] = torch.tensor(np.concatenate((step_data['obs'][0:4],step_data['obs'][18:22]),axis =0)).to(torch.float32)
        ret['action']      = torch.tensor(step_data['action'])
        ret['instruction'] = step_data['instruction']
        return ret
    
class Generate_data():
    def __init__(self,meta_env,data_dir,agents_dir,tasks,agents_level,total_num_steps,agents_dict_dir):
        self.data_dir = data_dir
        self.agents_dir = agents_dir
        self.tasks = tasks
        self.agents_level = agents_level
        self.total_num_steps = total_num_steps
        self.max_task_steps = total_num_steps//len(tasks)
        self.task_poses = ['Right','Mid','Left']
        self.agents_dict = json.load(open(agents_dict_dir))
        self.meta_env = meta_env
    def generate_data(self,device):
        self.agents_level += 100000
        data_dict = {}
        for task in self.tasks:
            data_dict[task] = self.generate_task_data(task,device)

        return data_dict

    def generate_task_data(self,task,device):
        task_data = []
        for pos in [0,1,2]:
            env   = self.meta_env(task,pos,save_images=True,process = 'train',wandb_log = False,general_model=True)
            agent = self.get_agent(env,task,pos,device)
            task_data += self.generate_pos_data(env,agent,task,pos,device)
        
        return task_data
    

    def get_agent(self,env,taskname,pos,device):
        run_name   = f'{taskname}_{self.task_poses[pos]}_ID{self.agents_dict[taskname][str(pos)]["id"]}'
        load_from = min(self.agents_level,self.agents_dict[taskname][str(pos)]['best_model'])
        load_path = os.path.join(self.agents_dir,run_name, f"{run_name}_{load_from}_steps")

        return SAC.load(load_path,env,device=device)


    def generate_pos_data(self,env,agent,task,pos,device):
        max_steps = self.max_task_steps//3

        episodes = []
        total_steps = 0
        id_num = 0
        while total_steps < max_steps:
            id_num += 1
            step_num = 0
            success = False
            done = False
            prev_obs , info = env.reset()
            prev_images_obs = self.save_images(info['images'],task,pos,id_num,step_num,device)
            episode = [] 
            while 1:
                a , _states= agent.predict(prev_obs, deterministic=True)
                obs, reward, done,success, info = env.step(a) 
                episode.append({'obs':prev_obs,'action':a,'reward':reward,'success':success,'images_dir':prev_images_obs})
                
                if (success or done): break 
                prev_obs = obs
                prev_images_obs = self.save_images(info['images'],task,pos,id_num,step_num,device)
                step_num+=1

            total_steps+=step_num
            episodes.append(episode[:])
        return episodes   
    def save_images(self,images,taskname,pos,id_num,step_num,device):
        ret = []
        for i in range(len(images)):
            save_dir = os.path.join(self.data_dir,str(device),taskname,str(pos))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            img_dir = os.path.join(save_dir,f'{id_num}_{step_num}_{i}.jpg')
            cv2.imwrite(img_dir,cv2.cvtColor(images[i],cv2.COLOR_RGB2BGR))
            ret.append(img_dir)
            
        return ret
    

class generator_manager():
    def __init__(self,args,meta_env,preprocess):
        self.tasks_commands = json.load(open(args.tasks_commands_dir))
        self.train_data_generator = Generate_data(meta_env,os.path.join(args.data_dir,'train'),args.agents_dir,args.tasks,args.init_agents_level,args.train_data_total_steps,args.agents_dict_dir)
        self.train_dataset = MW_dataset(self.tasks_commands,preprocess,total_data_len=args.train_data_total_steps)
        
        self.valid_data_generator = Generate_data(meta_env,os.path.join(args.data_dir,'valid'),args.agents_dir,args.tasks,args.init_agents_level,args.valid_data_total_steps,args.agents_dict_dir)
        self.valid_dataset = MW_dataset(self.tasks_commands,preprocess,total_data_len=args.valid_data_total_steps)

        self.batch_size  = args.batch_size
        self.num_workers = args.num_workers
        self.preprocess = preprocess    

    def get_train_dataloader(self,device):
        dataset_dict = self.train_data_generator.generate_data(device)
        self.train_dataset.load_data(dataset_dict)
        train_dataloader = torch.utils.data.DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True,num_workers = self.num_workers)
        return train_dataloader
    

    def get_valid_dataloader(self,device):
        dataset_dict = self.valid_data_generator.generate_data(device)
        self.valid_dataset.load_data(dataset_dict)
        val_dataloader = torch.utils.data.DataLoader(self.valid_dataset,batch_size=self.batch_size,shuffle=False,num_workers = self.num_workers)
        return val_dataloader 


