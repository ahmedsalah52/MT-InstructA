from train_utils.metaworld_dataset import Generate_data
from train_utils.args import  parser 
import json
from meta_env import meta_env
import os
import wandb
args = parser.parse_args()





generator = Generate_data(meta_env,os.path.join(args.project_dir,args.data_dir,'data'),args.agents_dir,args.tasks,args.train_data_total_steps,args.agents_dict_dir,args.agent_levels,with_imgs=args.with_imgs)
data_dict  = generator.generate_data()


with open(os.path.join(args.project_dir,args.data_dir,'dataset_dict.json'), 'w') as f:
    json.dump(data_dict, f)


