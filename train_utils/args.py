import argparse

parser = argparse.ArgumentParser(
                    prog='Metaworld multitasks',
                    description='What the program does',
                    epilog='Text at the bottom of help')
#project
parser.add_argument('--project_name', type=str, default='metaworld_general_model')
parser.add_argument('--run_name', type=str, default='run_1')
parser.add_argument('--logs_dir', type=str, default='/system/user/publicdata/mansour_datasets/metaworld/all_logs')

#model                
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--model_name', type=str, default='clip')

#clip params
parser.add_argument('--image_model_name', type=str, default='resnet50')
parser.add_argument('--text_model_name', type=str, default='distilbert-base-uncased')
parser.add_argument('--loss_fun', type=str, default='mse')
parser.add_argument('--img_model_lr', type=float, default=1e-5)
parser.add_argument('--txt_model_lr', type=float, default=1e-5)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--text_model_pretrained', type=bool, default=True)
parser.add_argument('--text_model_trainable', type=bool, default=True)
parser.add_argument('--image_model_pretrained', type=bool, default=True)
parser.add_argument('--image_model_trainable', type=bool, default=True)
parser.add_argument('--text_model_max_length', type=int, default=20)

#data
parser.add_argument('--train_data_total_steps', type=int, default=100000)
parser.add_argument('--valid_data_total_steps', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--init_agents_level', type=int, default=100000)
parser.add_argument('--data_dir', type=str, default='/system/user/publicdata/mansour_datasets/metaworld/generated_data/')
parser.add_argument('--agents_dir', type=str, default='/system/user/publicdata/mansour_datasets/metaworld/logs/')
parser.add_argument('--tasks', type=list, default=['button-press-topdown-v2'])
parser.add_argument('--agents_dict_dir', type=str, default='training_configs/agents_dict.json')
parser.add_argument('--tasks_commands_dir', type=str, default='training_configs/commands.json')

#training
parser.add_argument('--num_epochs', type=int, default=40)
parser.add_argument('--generate_data_every', type=int, default=2)
parser.add_argument('--check_val_every_n_epoch', type=int, default=2)
parser.add_argument('--evaluate_every', type=int, default=2)
parser.add_argument('--evaluation_episodes',help='number of episodes per task per position', type=int, default=1)
parser.add_argument('--load_checkpoint_path', type=str, default=None)
