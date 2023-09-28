import argparse

parser = argparse.ArgumentParser(
                    prog='Metaworld multitasks',
                    description='What the program does',
                    epilog='Text at the bottom of help')
#project
parser.add_argument('--project_name', type=str, default='metaworld_general_model')
parser.add_argument('--run_name', type=str, default='run_1')

#model                
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
parser.add_argument('--data_size', type=int, default=50000)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--validation_episodes', type=int, default=10)


#training
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
parser.add_argument('--checkpoint_path', type=str, default=None)


