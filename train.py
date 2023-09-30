import torch
from train_utils.model import base_model
from train_utils.args import  parser 
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from train_utils.metaworld_dataset import generator_manager,temp_dataset
from meta_env import meta_env
import os
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

def main():
    args = parser.parse_args()
    wandb_logger = WandbLogger( 
    project= args.project_name,
    name   = args.run_name)
   
    wandb.init(
    project=args.project_name,
    name = args.run_name
    ) 
    preprocess = transforms.Compose([
            transforms.ToTensor(), 
            #transforms.Resize((224,224)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])      
    
    os.environ["WANDB_DIR"] = os.path.join(args.logs_dir,args.project_name,args.run_name)

    
    valid_checkpoint_callback = ModelCheckpoint(
        dirpath = os.path.join(args.logs_dir,args.project_name,args.run_name),
        filename= '{epoch}-{val_loss:.3f}',
        monitor="val_loss",  # Monitor validation loss
        mode="min",  # "min" if you want to save the lowest validation loss
        save_top_k=1,  # Save only the best model
        save_last=True,  # Save the last model as well
        every_n_epochs=args.check_val_every_n_epoch
        )
    succ_rate_checkpoint_callback = ModelCheckpoint(
        dirpath = os.path.join(args.logs_dir,args.project_name,args.run_name),
        filename= '{epoch}-{success_rate:.2f}',
        monitor="success_rate",  # Monitor validation loss
        mode="min",  # "min" if you want to save the lowest validation loss
        save_top_k=1,  # Save only the best model
        save_last=False,  # Save the last model as well
        every_n_epochs=args.evaluate_every
        )
    
    #data_generator.get_train_dataloader(model.device)
    #val_dataloaders=data_generator.get_valid_dataloader(model.device)

    empty_dataset = temp_dataset()
    temp_dataloader = torch.utils.data.DataLoader(empty_dataset,batch_size=1,shuffle=False)
   
    data_generator = generator_manager(args=args,meta_env = meta_env ,preprocess=preprocess)
    model = base_model(args=args,generator=data_generator,env=meta_env,seed=args.seed)

    trainer = Trainer(callbacks=[succ_rate_checkpoint_callback,valid_checkpoint_callback],logger = wandb_logger,max_epochs=args.num_epochs,check_val_every_n_epoch=args.check_val_every_n_epoch,reload_dataloaders_every_n_epochs=1)
    trainer.fit(model,train_dataloaders=temp_dataloader ,val_dataloaders=temp_dataloader,ckpt_path= args.load_checkpoint_path)



if __name__ == "__main__":
    main()
    