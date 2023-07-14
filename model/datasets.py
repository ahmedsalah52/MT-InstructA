import csv
from PIL import Image
import os
import json
import numpy as np
import torch
import random
class Metaworld_Dataset:
    def __init__(self,json_file_dir,data_dir,images_transform,general_transform,tokenizer) -> None:
        self.csv_file = json_file_dir
        self.data_dir = data_dir
        self.images_transform = images_transform
        self.general_transform = general_transform
        with open(json_file_dir, "r") as read_file:
            data = json.load(read_file)
    
        self.data = data
        self.instructons = {'button-press-topdown-v2':['press the button']}
        self.tokenizer  = tokenizer
        self.taskname      = self.data[str(0)]['task_name']
        self.data     = self.data[str(0)]['data'] 
        self.instruct = 'press the button'
        self.max = 0
    def __len__(self):
        return len(self.data['step'])
    
    def __getitem__(self,idx):
        #idx = str(idx) 
        self.encoded_captions = self.tokenizer([self.instruct], padding=True, truncation=True, max_length=CFG.max_length)
        
        
        ret = {
            key: torch.tensor(values[0])
            for key, values in self.encoded_captions.items()
        }

        step        = self.data['step'][idx]
        prev_action = self.data['prev_action'][idx]
        action      = self.data['action'][idx]
        reward      = self.data['reward'][idx]
        state       = self.data['state'][idx]
    
        images_dir = os.path.join(self.data_dir,'images',self.taskname,str(0))
        
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
        #action[0:3] = (action[0:3]+5)/10
        action[action == -1] = 2


        ret['image']       = torch.stack(step_images)
        ret['action']      = action
        ret['state']       = torch.tensor(state)
        ret['prev_action'] = torch.tensor(prev_action)
        ret['reward']      = torch.tensor(reward)
        ret['caption']     = self.instruct
        
        if abs(ret['action'].max()) > self.max:
            self.max = ret['action'].max()
        return ret
    

def prepare_batch(batch):
    batch['image'] = batch['image'].permute(2,0,1,3,4,5)
    

    return batch

