import torch
from train_utils.model import base_model
from train_utils.args import  parser 
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer


def main():
    args = parser.parse_args()

    wandb_logger = WandbLogger( 
    project= args.project_name,
    name   = args.run_name)
    trainer = Trainer(logger=wandb_logger)


    train_dataloaders = None
    val_dataloaders = None
    model = base_model(args=args)
    trainer = Trainer(logger = wandb_logger,max_epochs=args.num_epochs,check_val_every_n_epoch=args.check_val_every_n_epoch)
    trainer.fit(model, train_dataloaders,val_dataloaders,ckpt_path= args.checkpoint_path)



if __name__ == "__main__":
    main()