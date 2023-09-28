import cv2
import numpy as np
from torch.utils.data import Dataset
import os
import json
from stable_baselines3 import SAC


        
class MW_dataset(Dataset):
    def __init__(self,data_dir,tasks):
        self.tasks = tasks
        self.data_dir = data_dir
        self.data = []
        
        self.data_dict = self.get_data_dict()

    
    def get_agent(self,taskname,pos):
        pass
    def get_episodes(self,taskname,pos):
        
        return []
    
class Generate_data():
    def __init__(self,meta_env,data_dir,agents_dir,tasks,total_num_steps,agents_level,agents_dict_dir):
        self.data_dir = data_dir
        self.agents_dir = agents_dir
        self.tasks = tasks
        self.total_num_steps = total_num_steps
        self.max_task_steps = total_num_steps//len(tasks)
        self.agents_level = agents_level
        self.task_poses = ['Right','Mid','Left']
        self.agents_dict = json.load(open(agents_dict_dir))
        self.meta_env = meta_env
    def generate_data(self):
        data_dict = {}
        for task in self.tasks:
            data_dict[task] = self.generate_task_data(task)

        return data_dict

    def generate_task_data(self,task):
        task_data = []
        for pos in [0,1,2]:
            env   = self.meta_env(task,pos,save_images=True,process = 'train',wandb_log = False)
            agent = self.get_agent(env,task,pos)
            task_data.append(self.generate_pos_data(env,agent,task,pos))
        
        return task_data
    

    def get_agent(self,env,taskname,pos):
        run_name   = f'{taskname}_{self.task_poses[pos]}_ID{self.agents_dict[taskname][str(pos)]["id"]}'
        load_from = min(self.agents_level,self.agents_dict[taskname][str(pos)]['best_model'])
        load_path = os.path.join(self.agents_dir,run_name, f"{run_name}_{load_from}_steps")

        return SAC.load(load_path,env)


    def generate_pos_data(self,env,agent,task,pos):
        max_steps = self.max_task_steps//3

        episodes = []
        total_steps = 0
        id_num = 0
        while total_steps < max_steps:
            id_num += 1
            success = False
            done = False
            obs , info = env.reset()
            step_num = 0
            episode = [{'obs':obs,'action':None,'reward':None,'success':False,'file_order':info['file_order'],'images':self.save_images(info['images'],task,pos,id_num,step_num)}]
            while not (success or done):
                step_num+=1
                a , _states= agent.predict(obs, deterministic=True)
                obs, reward, done,success, info = env.step(a)  # Step the environoment with the sampled random action
                episode.append({'obs':obs,'action':a,'reward':reward,'success':success,'images':self.save_images(info['images'],task,pos,id_num,step_num)})
            
            total_steps+=step_num
            episodes.append(episode[:])
        return episodes   
    def save_images(self,images,taskname,pos,id_num,step_num):
        ret = []
        for i in range(len(images)):
            save_dir = os.path.join(self.data_dir,taskname,str(pos))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            img_dir = os.path.join(save_dir,f'{id_num}_{step_num}_{i}.jpg')
            cv2.imwrite(img_dir,cv2.cvtColor(images[i],cv2.COLOR_RGB2BGR))
            ret.append(img_dir)
            
        return ret
    
