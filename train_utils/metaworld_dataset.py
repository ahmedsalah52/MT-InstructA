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
from collections import defaultdict
from tqdm import tqdm
class temp_dataset(Dataset):
    def __init__(self,seq_len=1,seq_overlap=10):
        self.data = []
        for i in range(100):
            self.data.append(i)
        self.sequence = seq_len>1
        self.seq_len = seq_len
        self.seq_overlap = seq_overlap 
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        if not self.sequence:
            return self.prepare_step(None)
        
        rets = defaultdict(list)
        for i in range(self.seq_len):
            step = self.prepare_step(None)
            for k,v in step.items():
                rets[k].append(v)
        
        #rets = {k : torch.tensor(v) if k != 'instruction' else v  for k,v in rets.items()}

        return rets
    def prepare_step(self,step_data):
        ret = {}
        ret['images']         = torch.zeros((5,3,224,224)).to(torch.float32)
        ret['hand_pos']       = torch.zeros(8).to(torch.float32)
        ret['action']         = torch.zeros(4).to(torch.float32)
        
        ret['instruction']    = "empty instruction"

        return ret

def get_stats(data_dict):
    table=[]
    total_success_rate = 0
    avg_length = 0
    max_length = 0
    min_length = np.inf
    for task_name , episodes in data_dict.items():
        task_success = 0
        for episode in episodes:
            task_success += episode[-1]['success']
            avg_length += len(episode)
            max_length = max(max_length,len(episode))
            min_length = min(min_length,len(episode))
        avg_length = float(avg_length)/len(episodes)
        task_success_rate = float(task_success) / len(episodes)
        total_success_rate += task_success_rate
        table.append([task_name,task_success_rate])
    table.append(['total_success_rate',total_success_rate/len(data_dict.items())])
    table.append(['avg_length',avg_length])
    table.append(['max_length',max_length])
    table.append(['min_length',min_length])

    return table

class MW_dataset(Dataset):
    def __init__(self,preprocess,dataset_dict_dir,dataset_dir,tasks_commands,total_data_len,seq_len=1,seq_overlap=10):
        self.data_dict = json.load(open(dataset_dict_dir))
        self.dataset_dir = dataset_dir
        self.tasks_commands = tasks_commands
        self.total_data_len = total_data_len
        self.preprocess = preprocess
        self.sequence = seq_len>1
        self.seq_len = seq_len
        self.seq_overlap = seq_overlap 
        if self.sequence: self.load_data_sequence()
        else: self.load_data_single()
    def load_data_single(self):
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
    def load_data_sequence(self):
        self.tasks = list(self.data_dict.keys())
        self.data = []
        for task in self.tasks:
            print('preparing task:',task)
            for epi in tqdm(range(len(self.data_dict[task]))):
            #for epi in range(len(self.data_dict[task])):
                episode = []
                for s in range(len(self.data_dict[task][epi])):
                    step = self.data_dict[task][epi][s]
                    step['instruction'] = random.choice(self.tasks_commands[task])
                    episode.append(step)
                self.data += self.get_seqs(episode[:])

        print('seq data preparation done with length',len(self.data))
    def get_seqs(self,episode):
        seqs = []
        i  = 0
        done = False
        while not done:
            start = i
            end = i + self.seq_len
            if end >= len(episode)-1:
                done = True
                sublist = episode[-self.seq_len:]
            else:
                sublist = episode[start:end]
                i = end - self.seq_overlap

            seqs.append(sublist)
        return seqs
    def get_overlap(self):

        rand = random.randint(0,self.seq_overlap)
        return rand + self.seq_overlap//2
    
    def get_stats(self):
        return get_stats(self.data_dict)
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        if not self.sequence:
            return self.prepare_step(self.data[idx])
        
        sequence_steps = self.data[idx]
        rets = defaultdict(list)
        for step in sequence_steps:
            step = self.prepare_step(step)
            for k,v in step.items():
                rets[k].append(v)
        
        #rets = {k : torch.tensor(v) if k != 'instruction' else v  for k,v in rets.items()}

        return rets
    def prepare_step(self,step_data):
        images_dir = step_data['images_dir']
        images = [self.preprocess(Image.open(os.path.join(self.dataset_dir,dir))) for dir in images_dir]
        ret = {}
        ret['images']   = torch.stack(images)
        ret['hand_pos'] = torch.tensor(np.concatenate((step_data['obs'][0:4],step_data['obs'][18:22]),axis =0)).to(torch.float32)
        ret['action']      = torch.tensor(step_data['action'])
        ret['instruction'] = step_data['instruction']
        return ret

def split_dict(dict_of_lists, split_ratio=0.8,seed=42):
    """
    Split a dictionary of lists into training and validation dictionaries with the same keys.
    
    :param dict_of_lists: Input dictionary with keys and lists.
    :param split_ratio: The ratio of data to be allocated for training (default is 0.8).
    :return: A tuple of two dictionaries - training_dict and validation_dict.
    """
    if not isinstance(dict_of_lists, dict):
        raise ValueError("Input must be a dictionary of lists")
    
    if not (0 <= split_ratio <= 1):
        raise ValueError("Split ratio must be between 0 and 1")
    random.seed(seed)

    training_dict = {}
    validation_dict = {}

    for key, value in dict_of_lists.items():
        if not isinstance(value, list):
            raise ValueError(f"Value for key '{key}' must be a list")
        
        # Determine the split index
        split_index = int(len(value) * split_ratio)
        
        # Shuffle the list before splitting to ensure randomness
        random.shuffle(value)
        
        # Split the list into training and validation
        training_data   = value[:split_index]
        validation_data = value[split_index:]
        
        # Update the training and validation dictionaries
        training_dict[key] = training_data
        validation_dict[key] = validation_data
    
    return training_dict, validation_dict


class Generate_data():
    def __init__(self,meta_env,data_dir,agents_dir,tasks,total_num_steps,agents_dict_dir,agent_levels,poses,with_imgs):
        self.agent_levels = agent_levels
        self.data_dir = data_dir
        self.agents_dir = agents_dir
        self.tasks = tasks
        self.total_num_steps = total_num_steps
        self.max_task_steps = total_num_steps//len(tasks)
        self.task_poses = ['Right','Mid','Left']
        self.poses = poses
        self.agents_dict = json.load(open(agents_dict_dir))
        self.meta_env = meta_env
        self.agents_levels_step = 10000
        self.with_imgs = with_imgs

    def generate_data(self):
        data_dict = {}
        for task in self.tasks:
            data_dict[task] = self.generate_task_data(task)

        return data_dict

    def generate_task_data(self,task):
        task_data = []
        for agent_level in range(self.agent_levels):
            for pos in self.poses:
                env   = self.meta_env(task,pos,save_images=self.with_imgs,process = 'train',wandb_log = False,general_model=True)
                agent = self.get_agent(env,task,pos,agent_level)
                task_data += self.generate_pos_data(env,agent,task,pos,agent_level)
            
        return task_data
    

    def get_agent(self,env,taskname,pos,agent_level):
        run_name   = f'{taskname}_{self.task_poses[pos]}_ID{self.agents_dict[taskname][str(pos)]["id"]}'
        best_model = self.agents_dict[taskname][str(pos)]['best_model']
        #load_from  = int( (best_model//(10000 *self.agent_levels)) * (agent_level+1) * 10000)
        load_from  = best_model - int( (best_model//(10000 *self.agent_levels)) * (agent_level) * 10000)

        print(f"loading agent lvl{agent_level} for task {taskname} pos {pos} step {load_from} with best model step {best_model}")
        load_path = os.path.join(self.agents_dir,run_name, f"{run_name}_{load_from}_steps")

        return SAC.load(load_path,env)


    def generate_pos_data(self,env,agent,task,pos,agent_level):
        max_steps = self.max_task_steps//(len(self.poses)*self.agent_levels)
        prev_images_obs = None

        episodes = []
        total_steps = 0
        id_num = 0
        while total_steps < max_steps:
            id_num += 1
            step_num = 0
            success = False
            done = False
            prev_obs , info = env.reset()
            
            episode = [] 
            while 1:
                if self.with_imgs: prev_images_obs = self.save_images(info['images'],task,pos,id_num,step_num,agent_level)
                a , _states = agent.predict(prev_obs, deterministic=True)
                obs, reward, done,success, info = env.step(a) 
                episode.append({'obs':prev_obs.tolist(),'action':a.tolist(),'reward':reward,'success':success,'pos':self.task_poses[pos],'images_dir':prev_images_obs})
                
                if (success or done): break 
                prev_obs = obs
                step_num+=1

            total_steps+=step_num
            episodes.append(episode[:])
        return episodes   
    def save_images(self,images,taskname,pos,id_num,step_num,agent_level):
        ret = []
        for i in range(len(images)):
            ret_dir = os.path.join(taskname,str(pos))
            img_name = f'lvl{agent_level}_id{id_num}_step{step_num}_{i}.jpg'

            save_dir = os.path.join(self.data_dir,ret_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            img_dir = os.path.join(save_dir,img_name)
            cv2.imwrite(img_dir,cv2.cvtColor(images[i],cv2.COLOR_RGB2BGR))
            
            ret.append(os.path.join(ret_dir,img_name))
            
        return ret
    
