import torch
from train_utils.tl_model import TL_model,load_checkpoint,freeze_layers
from train_utils.args import  parser ,process_args
import os

import json
from collections import defaultdict
from PIL import Image
import tqdm
def main():
    args = parser.parse_args()
    args = process_args(args)
    data_dir = os.path.join(args.project_dir,'data',args.dataset)
    batch_size = args.batch_size

   
    tasks_commands = json.load(open(args.tasks_commands_dir))
    
    dict_dir , images_dir = os.path.join(data_dir,'dataset_dict.json'),os.path.join(data_dir,'data')
    model = TL_model(args=args,tasks_commands=None,env=None,wandb_logger=None,seed=args.seed)
    model = load_checkpoint(model,args.load_weights)
    model.eval()
    
    data_dict = json.load(open(dict_dir))
    encoded_data_dict = defaultdict(list)
    #init bar with no max value but shows progress
    print("encoding data")
    bar = tqdm.tqdm(total=0)
    for task , episodes in data_dict.items():
        if task not in args.tasks: continue
        for episode in episodes:
            images = []
            for step in episode:
                images.append(torch.stack([model.preprocess(Image.open(img_dir)) for img_dir in step['images_dir']]))
            
            batch = torch.stack(images).to(model.device)
            images_emps ,_,_= model.backbone({'images':batch},cat=False,vision=True,command=False,pos=False)
            images_emps = images_emps.detach().cpu().numpy()
            for step , step_emps in zip(episode,images_emps):
                step['imgs_emps'] = step_emps
                encoded_data_dict[task].append(step)

            bar.update(1)

    bar.close()
    with open(os.path.join(data_dir,'encoded_dataset_dict.json'),'w') as f:
        json.dump(encoded_data_dict,f)

if __name__ == "__main__":
    main()
    