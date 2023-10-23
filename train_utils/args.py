import argparse

parser = argparse.ArgumentParser(
                    prog='Metaworld multitasks',
                    description='What the program does',
                    epilog='Text at the bottom of help')
#project
parser.add_argument('--project_name', type=str, default='Metaworld_General_Model')
parser.add_argument('--run_name', type=str, default='run_1')
parser.add_argument('--logs_dir', type=str, default='logs')
parser.add_argument('--project_dir', type=str, default='/system/user/publicdata/mansour_datasets/metaworld/')



#data generation args
parser.add_argument('--train_data_total_steps', type=int, default=100000)
parser.add_argument('--agents_dir', type=str, default='/system/user/publicdata/mansour_datasets/metaworld/logs/')
parser.add_argument('--agent_levels', type=int, default=1,help='how many levels of agents to generate data for, if 1 then only the best agent will be used')
parser.add_argument('--with_imgs', action='store_true',help='to render and save images or export the dict file only')

#shared args
parser.add_argument('--data_dir', type=str, default='generated_data')
parser.add_argument('--tasks', type=list, default= ['button-press-topdown-v2', 'button-press-v2', 'door-lock-v2', 'door-open-v2',  'drawer-open-v2', 'window-open-v2', 'faucet-open-v2', 'faucet-close-v2', 'handle-press-v2', 'coffee-button-v2'])
parser.add_argument('--poses', type=list, default= [0,1,2])
parser.add_argument('--agents_dict_dir', type=str, default='configs/general_model_configs/agents_dict.json')



#model                
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--backbone', type=str, default='open_ai_clip',choices=['open_ai_clip','simple_clip'])
parser.add_argument('--neck', type=str, default='transformer',choices=[None,'transformer'])
parser.add_argument('--head', type=str, default='fc',choices=['fc'])
parser.add_argument('--pos_emp', type=int, default=128)



#simple clip backbone params
parser.add_argument('--image_model_name', type=str, default='resnet50')
parser.add_argument('--text_model_name', type=str, default='distilbert-base-uncased')
parser.add_argument('--text_model_pretrained', type=bool, default=True)
parser.add_argument('--text_model_trainable', type=bool, default=True)
parser.add_argument('--image_model_pretrained', type=bool, default=True)
parser.add_argument('--image_model_trainable', type=bool, default=True)
parser.add_argument('--text_model_max_length', type=int, default=20)
parser.add_argument('--img_model_lr', type=float, default=1e-5)
parser.add_argument('--txt_model_lr', type=float, default=1e-5)


#open ai clip backbone params
parser.add_argument('--loss_fun', type=str, default='mse')
parser.add_argument('--clip_lr', type=float, default=1e-6)
parser.add_argument('--head_lr', type=float, default=1e-4)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--op_image_model_name', type=str, default='ViT-B/32',choices=
['RN50',
 'RN101',
 'RN50x4',
 'RN50x16',
 'RN50x64',
 'ViT-B/32',
 'ViT-B/16',
 'ViT-L/14',
 'ViT-L/14@336px'])

#neck params
parser.add_argument('--n_heads'     , type=int, default=8)
parser.add_argument('--att_head_emp', type=int, default=128)
parser.add_argument('--neck_layers' , type=int, default=1)

#head params


#training
parser.add_argument('--tasks_commands_dir', type=str, default='configs/general_model_configs/commands.json')
parser.add_argument('--commands_split_ratio', type=float,default=0.8,help='how much of the commands to use for training')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=12)
parser.add_argument('--num_epochs', type=int, default=60)
parser.add_argument('--generate_data_every', type=int, default=2)
parser.add_argument('--check_val_every_n_epoch', type=int, default=2)
parser.add_argument('--evaluate_every', type=int, default=1)
parser.add_argument('--evaluation_episodes',help='number of episodes per task per position', type=int, default=1)
parser.add_argument('--load_checkpoint_path', type=str, default=None)
parser.add_argument('--n_gpus', type=int, default=1)


#testing
parser.add_argument('--save_video', action='store_true')
parser.add_argument('--video_exp_name', type=str, default='video')
parser.add_argument('--video_dir', type=str, default='video_results')
parser.add_argument('--video_res', type=tuple, default=(1920,1080))