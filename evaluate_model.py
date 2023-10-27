import torch
from train_utils.tl_model import TL_model
from train_utils.args import  parser 

from meta_env import meta_env,task_manager
from train_utils.metaworld_dataset import split_dict

import cv2
from PIL import Image
import numpy as np
import random
import os
import shutil
import json

def main():
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks_commands = json.load(open(args.tasks_commands_dir))
    tasks_commands = {k:list(set(v)) for k,v in tasks_commands.items() if k in args.tasks}
    _,val_tasks_commands = split_dict(tasks_commands,args.commands_split_ratio,seed=args.seed)
    
    model = TL_model.load_from_checkpoint(args.load_checkpoint_path,args=args,tasks_commands=val_tasks_commands,env=meta_env,wandb_logger=None,seed=None)
    model.eval()
    model.evaluate_model()

if __name__ == "__main__":
    main()
    