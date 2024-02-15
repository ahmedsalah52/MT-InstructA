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


# Define a custom sampler with weighted sampling
class WeightedRandomSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, weights, num_samples, replacement=True):
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement))

    def __len__(self):
        return self.num_samples


class temp_dataset(Dataset):
    def __init__(self,seq_len=1,seq_overlap=10,cams=[0,1,2,3,4],with_imgs=True):
        self.data = []
        for i in range(100):
            self.data.append(i)
        self.sequence = seq_len>1
        self.seq_len = seq_len
        self.seq_overlap = seq_overlap 
        self.cams= cams
        self.with_imgs = with_imgs
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
        if self.with_imgs:
            ret['images']         = torch.zeros((len(self.cams),3,224,224)).to(torch.float32)
        ret['hand_pos']       = torch.zeros(8).to(torch.float32)
        ret['action']         = torch.zeros(4).to(torch.float32)
        ret['timesteps']      = 1
        ret['reward']         = 8.0
        ret['return_to_go']   = 800.0
        ret['instruction']    = "empty instruction"
        ret['obs'] = torch.zeros(39).to(torch.float32)
        ret['attention_mask'] = 1
        ret['task_id'] = torch.tensor([0],dtype=torch.int)
        return ret

"""def get_stats(data_dict):
    table=[]
 
    stats_dict = defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    poses = ['Left','Mid','Right']
    for task_name , episodes in data_dict.items():
        for episode in episodes:
            stats_dict[task_name][episode[-1]['pos']]['success'].append(episode[-1]['success']) 
            
            stats_dict[task_name][episode[-1]['pos']]['length'].append(len(episode)) 
            
        poses_sr = [np.mean(stats_dict[task_name][pos]['success']) for pos in poses]
        poses_ln = [np.mean(stats_dict[task_name][pos]['length']) for pos in poses]
           
        table.append([task_name,*poses_sr,np.mean(poses_sr),*poses_ln,np.mean(poses_ln)])

    total_sr_left   = np.mean([row[1] for row in table])
    total_sr_mid    = np.mean([row[2] for row in table])
    total_sr_right  = np.mean([row[3] for row in table])
    total_sr        = np.mean([row[4] for row in table])
    total_len_left  = np.mean([row[5] for row in table])
    total_len_mid   = np.mean([row[6] for row in table])
    total_len_right = np.mean([row[7] for row in table])
    total_len       = np.mean([row[8] for row in table])
    table.append(['total',total_sr_left,total_sr_mid,total_sr_right,total_sr,total_len_left,total_len_mid,total_len_right,total_len])
   

    return table
    wandb_logger.log_table(key=f"Dataset Success Rate",  columns=['Task name','Left SR','Middle SR','Right SR','Total SR','Left Len','Middle Len','Right Len','Total Len'],data=stats_table)

"""
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
    def __init__(self,preprocess,dataset_dict_dir,dataset_dir,tasks_commands,total_data_len,seq_len=1,seq_overlap=10,cams=[0,1,2,3,4],with_imgs=True):
        self.data_dict = json.load(open(dataset_dict_dir))
        self.dataset_dir = dataset_dir
        self.tasks_commands = tasks_commands
        self.total_data_len = total_data_len
        self.preprocess = preprocess
        self.sequence = seq_len>1
        self.seq_len = seq_len
        self.seq_overlap = seq_overlap 
        self.cams = cams
        self.max_return_to_go = defaultdict(lambda:0)
        self.with_imgs = with_imgs
        self.data_specs = {}
        self.load_data()
        print('seq' if self.sequence else 'single step'+' data preparation done with length',len(self.data))
        print('state mean:',np.mean(self.obs_state_mean), 'obs std:',np.mean(self.obs_state_std))
        
    def load_data(self):
        self.tasks = list(self.tasks_commands.keys())
        self.data = []
        all_obs = []
        traj_lens = []
        for i,task in enumerate(self.tasks):
            print('preparing task:',task)
            for epi in tqdm(range(len(self.data_dict[task]))):
                episode = []
                return_to_go = sum([self.data_dict[task][epi][s]['reward'] for s in range(len(self.data_dict[task][epi]))])
                self.max_return_to_go[task] = max(self.max_return_to_go[task],return_to_go)
                traj_lens.append(len(self.data_dict[task][epi]))
                success = self.data_dict[task][epi][-1]['success']
                for s in range(len(self.data_dict[task][epi])):
                    step = self.data_dict[task][epi][s]
                    step['success'] = success
                    step['task_id'] = i
                    step['timesteps'] = s
                    step['return_to_go'] = return_to_go
                    return_to_go -= step['reward']
                    episode.append(step)
                    all_obs.append(step['obs'])
                if self.sequence:
                    self.data.append(episode) 
                    
                else:
                    self.data += episode
        
     
        traj_lens = np.array(traj_lens,dtype=np.float32)
        p_sample = traj_lens / np.sum(traj_lens)
        states = np.concatenate(all_obs, axis=0)
        self.obs_state_mean, self.obs_state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        self.data_specs['obs_state_mean'] = self.obs_state_mean
        self.data_specs['obs_state_std']  = self.obs_state_std
        self.data_specs['p_sample']       = p_sample
        self.data_specs['max_return_to_go'] = self.max_return_to_go
      
    
    def get_stats(self):
        
        table = get_stats(self.data_dict)
        #table.append(['max_return_to_go',self.max_return_to_go])
        return table
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        if not self.sequence:
            return self.prepare_step(self.data[idx])
        if idx > len(self.data) - 1:
            print(f'{idx} idx not found')
            idx = random.randint(0, len(self.data) - 1)
        sequence_steps = self.data[idx]
        episode_length = len(sequence_steps)
        #get the sequence with length from 1 to self.seq_len
        start = random.randint(0, episode_length - 1)
        end = min(start + self.seq_len , episode_length)
        actual_seq_len = end - start
        rets = defaultdict(list)
          
        for step in sequence_steps[start:end]:
            step = self.prepare_step(step)
            for k,v in step.items():
                rets[k].append(v)

        for i in range(self.seq_len - actual_seq_len):
            step = self.prepare_padding_step(None)
            for k,v in step.items():
                rets[k].append(v)
      
        return rets
    def prepare_step(self,step_data):
        ret = {}
        if self.with_imgs:
            images_dir = step_data['images_dir']
            images = [self.preprocess(Image.open(os.path.join(self.dataset_dir,dir))) for dir in images_dir if int(dir.split('_')[-1].split('.')[0]) in self.cams]
            ret['images']   = torch.stack(images)
        task_name           = self.tasks[step_data['task_id']]
        ret['hand_pos']     = torch.tensor(np.concatenate((step_data['obs'][0:4],step_data['obs'][18:22]),axis =0)).to(torch.float32)
        ret['obs']          = torch.tensor(step_data['obs']).to(torch.float32)
        ret['action']       = torch.tensor(step_data['action'])
        ret['task_id']      = torch.tensor([step_data['task_id']],dtype=torch.int)
        ret['timesteps']    = step_data['timesteps']
        ret['reward']       = step_data['reward'] / self.max_return_to_go[task_name]
        ret['return_to_go'] = step_data['return_to_go'] / self.max_return_to_go[task_name]
        ret['attention_mask'] = 1
        if step_data['success']:
            ret['instruction']  = random.choice(self.tasks_commands[task_name])
        else:
            ret['instruction']  = "do a stupid thing with "+task_name.remove('-v2').replace('-',' ')
        return ret
    def prepare_padding_step(self,step_data):
        ret = {}
        if self.with_imgs:
            ret['images']         = torch.zeros((len(self.cams),3,224,224)).to(torch.float32)
        ret['hand_pos']       = torch.zeros(8).to(torch.float32)
        ret['action']         = torch.zeros(4).to(torch.float32)
        ret['timesteps']      = 0
        ret['reward']         = 0
        ret['return_to_go']   = 0
        ret['instruction']    = ""
        ret['obs'] = torch.zeros(39).to(torch.float32)
        ret['attention_mask'] = 0
        ret['task_id'] = torch.tensor([0],dtype=torch.int)
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
    
