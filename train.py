import torch
from train_utils.model import base_model
from train_utils.args import  parser 
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from meta_env import meta_env
import os
from pytorch_lightning.callbacks import ModelCheckpoint
from train_utils.metaworld_dataset import MW_dataset
import json

def main():
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.project_dir,args.project_name,args.run_name)):
        os.makedirs(os.path.join(args.project_dir,args.project_name,args.run_name,'checkpoints'))
        os.makedirs(os.path.join(args.project_dir,args.project_name,args.run_name,args.logs_dir))

    os.environ["WANDB_DIR"] = os.path.join(args.project_dir,args.project_name,args.run_name,args.logs_dir)
    os.environ["WANDB_CACHE_DIR"] = os.path.join(args.project_dir,args.project_name,args.run_name,args.logs_dir)

    wandb_logger = WandbLogger( 
    project= args.project_name,
    name   = args.run_name)
    print("checkpoints dir:",os.path.join(args.project_dir,args.project_name,args.run_name,'checkpoints'))
    succ_rate_checkpoint_callback = ModelCheckpoint(
        dirpath = os.path.join(args.project_dir,args.project_name,args.run_name,'checkpoints'),
        filename= '{epoch}-{success_rate:.2f}',
        monitor="success_rate",  # Monitor validation loss
        mode="max",  # "min" if you want to save the lowest validation loss
        save_top_k=5,  # Save only the best model
        save_last=True,  # Save the last model as well
        every_n_epochs=args.evaluate_every
        )
    training_checkpoint_callback = ModelCheckpoint(
        dirpath   = 'checkpoints/',
        save_last=True
    )
    
    tasks_commands = json.load(open(args.tasks_commands_dir))
    model = base_model(args=args,tasks_commands=tasks_commands,env=meta_env,wandb_logger=wandb_logger,seed=args.seed)

    train_dataset = MW_dataset(model.preprocess,os.path.join(args.project_dir,args.dataset_dict_dir),os.path.join(args.project_dir,args.data_dir),tasks_commands,total_data_len=args.train_data_total_steps)
    stats_table = train_dataset.get_stats()
    wandb_logger.log_table(key=f"Dataset Success Rate",  columns=['Task name','Success Rate'],data=stats_table)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers = args.num_workers)


    trainer = Trainer(callbacks=[training_checkpoint_callback],logger = wandb_logger,max_epochs=args.num_epochs,check_val_every_n_epoch=args.check_val_every_n_epoch,strategy='ddp_find_unused_parameters_true')#,reload_dataloaders_every_n_epochs=args.generate_data_every,use_distributed_sampler=False)
    trainer.fit(model,train_dataloader,ckpt_path= args.load_checkpoint_path)



if __name__ == "__main__":
    main()
    