import torch
from train_utils.model import base_model
from train_utils.args import  parser 
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from train_utils.metaworld_dataset import generator_manager,temp_dataset
from meta_env import meta_env
import os
from pytorch_lightning.callbacks import ModelCheckpoint
import json
from train_utils.metaworld_dataset import MW_dataset


def main():
    args = parser.parse_args()
    wandb_logger = WandbLogger( 
    project= args.project_name,
    name   = args.run_name)
   
    """ wandb.init(
    project=args.project_name,
    name = args.run_name
    ) """
    
    
    os.environ["WANDB_DIR"] = os.path.join(args.project_dir,args.project_name,args.run_name)

    
    succ_rate_checkpoint_callback = ModelCheckpoint(
        dirpath = os.path.join(args.logs_dir,args.project_name,args.run_name),
        filename= '{epoch}-{success_rate:.2f}',
        monitor="success_rate",  # Monitor validation loss
        mode="min",  # "min" if you want to save the lowest validation loss
        save_top_k=1,  # Save only the best model
        save_last=True,  # Save the last model as well
        every_n_epochs=args.evaluate_every
        )
    
    #data_generator.get_train_dataloader(model.device)
    #val_dataloaders=data_generator.get_valid_dataloader(model.device)


    train_dataset = MW_dataset(args.dataset_dict_dir,args.tasks_commands_dir,total_data_len=args.train_data_total_steps)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers = args.num_workers)

    #data_generator = generator_manager(args=args,meta_env = meta_env ,preprocess=preprocess)
    model = base_model(args=args,tasks_commands=train_dataset.tasks_commands,env=meta_env,wandb_logger=wandb_logger,seed=args.seed)

    trainer = Trainer(callbacks=succ_rate_checkpoint_callback,logger = wandb_logger,max_epochs=args.num_epochs,check_val_every_n_epoch=args.check_val_every_n_epoch)#,reload_dataloaders_every_n_epochs=args.generate_data_every,use_distributed_sampler=False)
    trainer.fit(model,train_dataloader,ckpt_path= args.load_checkpoint_path)



if __name__ == "__main__":
    main()
    