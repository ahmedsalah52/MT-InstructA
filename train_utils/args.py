import argparse

parser = argparse.ArgumentParser(
                    prog='Metaworld multitasks',
                    description='What the program does',
                    epilog='Text at the bottom of help')
#project
parser.add_argument('--project_name', type=str, default='general_model')
parser.add_argument('--run_name', type=str, default='run_1')
parser.add_argument('--logs_dir', type=str, default='logs')
parser.add_argument('--project_dir', type=str, default='/system/user/publicdata/mansour_datasets/metaworld/')
parser.add_argument('--run_notes', type=str, default='')



#data generation args
parser.add_argument('--train_data_total_steps', type=int, default=100000)
parser.add_argument('--agents_dir', type=str, default='/system/user/publicdata/mansour_datasets/metaworld/logs/')
parser.add_argument('--agent_levels', type=int, default=1,help='how many levels of agents to generate data for, if 1 then only the best agent will be used')
parser.add_argument('--with_imgs', action='store_true',help='to render and save images or export the dict file only')

#shared args
parser.add_argument('--dataset', type=str, default='generated_data')
parser.add_argument('--tasks', type=str, default= "button-press-topdown-v2,button-press-v2,door-lock-v2,door-open-v2,drawer-open-v2,window-open-v2,faucet-open-v2,faucet-close-v2,handle-press-v2,coffee-button-v2")
parser.add_argument('--poses', type=list, default= [0,1,2])
parser.add_argument('--agents_dict_dir', type=str, default='configs/general_model_configs/agents_dict.json')

#data preprocessing args
parser.add_argument('--cams',type=str,default='0,1,2,3,4')
parser.add_argument('--seq_len',type=int,default=1)
parser.add_argument('--seq_overlap',type=int,default=2,help='overlap between sequences')



#model       
#arch         
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--model', type=str, default='base',choices=['base','GAN','seq','dt','dt_obs'])
parser.add_argument('--backbone', type=str, default='open_ai_clip',choices=['open_ai_clip','simple_clip'])
parser.add_argument('--neck', type=str, default=None,choices=[None,'transformer','cross_attention','film'])
parser.add_argument('--head', type=str, default='fc',choices=['fc'])

#params
parser.add_argument('--action_dim', type=int, default=4)
parser.add_argument('--pos_dim', type=int, default=8)
parser.add_argument('--pos_emp', type=int, default=128)
parser.add_argument('--imgs_emps', type=int, default=512,help='the size of images embeddings')
parser.add_argument('--instuction_emps', type=int, default=512,help='the size of instruction embeddings')
parser.add_argument('--freeze_modules', type=str, default=None)
parser.add_argument('--freeze_except', type=str, default=None)
parser.add_argument('--max_ep_len',type=int,default=200)
parser.add_argument('--debugging_mode', action='store_true',help='by raising this flag, a smaller fake dataset will be used for debugging the pipline')

#GAN params
parser.add_argument('--action_emp', type=int, default=128)
parser.add_argument('--noise_len' , type=int, default=128)


#Decision Transformer params
parser.add_argument('--dt_embed_dim', type=int, default=128)
parser.add_argument('--dt_n_layer', type=int, default=3)
parser.add_argument('--dt_n_head', type=int, default=1)
parser.add_argument('--dt_activation_function', type=str, default='relu')
parser.add_argument('--dt_dropout', type=float, default=0.1)
parser.add_argument('--prompt_scale', type=float, default=164.6723)
parser.add_argument('--prompt', type=str, default='return_to_go',choices=['reward','return_to_go'])
parser.add_argument('--target_rtg', type=float, default=2000.0)
parser.add_argument('--use_env_reward', action='store_true',help='by raising this flag, the return to go will be calulated using the env reward instead of the model predicted reward')

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
parser.add_argument('--clip_lr', type=float, default=1e-6)
parser.add_argument('--head_lr', type=float, default=1e-4)
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

#neck params  --n_heads 16 --att_head_emp 16 --neck_layers 2 --neck_max_len 200
parser.add_argument('--n_heads'     , type=int, default=16)
parser.add_argument('--emp_size'    , type=int, default=16)
parser.add_argument('--neck_layers' , type=int, default=2)
parser.add_argument('--neck_dropout', type=int, default=0.2)
parser.add_argument('--neck_max_len', type=int, default=32)

parser.add_argument('--instruct_dropout', type=float, default=0) #only for the cross attention
parser.add_argument('--film_N_blocks', type=int, default=4)

#head params
parser.add_argument('--act_fun', type=str, default=None,choices=[None,'tanh'])


#loss funcitons
parser.add_argument('--loss_fun', type=str, default='mse',choices=['mse','relative_mse'])
parser.add_argument('--mag_weight', type=float, default=0.01,help='weight of the magnitude loss in the relative loss funtion')


#training
parser.add_argument('--tasks_commands_dir', type=str, default='configs/general_model_configs/commands.json')
parser.add_argument('--commands_split_ratio', type=float,default=0.8,help='how much of the commands to use for training')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=12)
parser.add_argument('--num_epochs', type=int, default=60)
parser.add_argument('--gradient_clip_val', type=float, default=0.25)
parser.add_argument('--checkpoint_every', type=int, default=1)
parser.add_argument('--evaluation_episodes',type=int, default=1,help='number of episodes per task per position')
parser.add_argument('--load_checkpoint_path', type=str, default=None)
parser.add_argument('--load_weights', type=str, default=None)
parser.add_argument('--n_gpus', type=int, default=1)
parser.add_argument('--schedular_step', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--opt_patience', type=int, default=1)
parser.add_argument('--trainloss_checkpoint',action='store_true')


#testing
parser.add_argument('--video_exp_name', type=str, default=None)
parser.add_argument('--video_dir', type=str, default='video_results')
parser.add_argument('--video_res', type=tuple, default=(1920,1080))
parser.add_argument('--vis_embeddings', action='store_true')


def process_args(args):
    args.cams  = [int(c) for c in args.cams.split(',')]
    args.tasks = [c for c in args.tasks.split(',')]

    return args