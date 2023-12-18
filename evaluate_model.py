import torch
from train_utils.tl_model import TL_model
from train_utils.args import  parser ,process_args

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
    args = process_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks_commands = json.load(open(args.tasks_commands_dir))
    tasks_commands = {k:list(set(tasks_commands[k])) for k in args.tasks} #the commands dict should have the same order as args.tasks list
   
    _,val_tasks_commands = split_dict(tasks_commands,args.commands_split_ratio,seed=args.seed)
    
    #assign general seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    

    model = TL_model.load_from_checkpoint(args.load_checkpoint_path,args=args,tasks_commands=val_tasks_commands,env=meta_env,wandb_logger=None,seed=None)
    model.eval()
    #train_dataset = MW_dataset(model.preprocess,os.path.join(data_dir,'dataset_dict.json'),os.path.join(data_dir,'data'),train_tasks_commands,total_data_len=args.train_data_total_steps,seq_len=args.seq_len,seq_overlap=args.seq_overlap,cams = args.cams,with_imgs='obs' not in args.model)
    #stats_table = train_dataset.get_stats()
    #model.model.set_dataset_specs(train_dataset.data_specs)

    dataset_rtg_dict = { 'max_return_to_go': {'button-press-topdown-v2': 1177.6984059210347, 'button-press-v2': 1289.018530097603, 'door-lock-v2': 1141.5641889309939, 'door-open-v2': 892.0785433900224, 'drawer-open-v2': 1487.7217649712063, 'window-open-v2': 1261.3283346301089, 'faucet-open-v2': 1140.6936299207996, 'faucet-close-v2': 1418.5725616135462, 'handle-press-v2': 1646.7226295389921, 'coffee-button-v2': 542.8299921921665}}
    model.model.set_dataset_specs(dataset_rtg_dict)

    success_rate = model.evaluate_model()
    to_log = f'model {args.load_checkpoint_path} with success rate {success_rate} - {args.run_name}'
    print('-'*50)
    print(to_log)
    print('-'*50)
    with open(os.path.join(os.path.split(args.load_checkpoint_path)[0],'logs.txt'),'a') as f:
        f.write(to_log+'\n')


if __name__ == "__main__":
    main()
    