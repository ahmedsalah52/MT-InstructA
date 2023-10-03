from train_utils.metaworld_dataset import Generate_data
from train_utils.args import  parser 
import json
import meta_env
import os
import wandb
args = parser.parse_args()





generator = Generate_data(meta_env,os.path.join(args.project_dir,args.data_dir),args.agents_dir,args.tasks,args.train_data_total_steps,args.agents_dict_dir)
data_dict  = generator.generate_data()


with open(os.path.join(args.project_dir,args.dataset_dict_dir), 'w') as f:
    json.dump(data_dict, f)


