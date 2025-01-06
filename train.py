import torch
from train_utils.tl_model import TL_model,load_checkpoint,freeze_layers
from train_utils.args import  parser ,process_args
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from meta_env import meta_env
import os
from pytorch_lightning.callbacks import ModelCheckpoint ,LearningRateMonitor
from train_utils.metaworld_dataset import MW_dataset,split_dict,temp_dataset,WeightedRandomSampler
import json

def main():
    args = parser.parse_args()
    args = process_args(args)
    data_dir = os.path.join(args.project_dir,'data',args.dataset)
    checkpoints_dir = os.path.join(args.project_dir,args.project_name,args.run_name,'checkpoints')

    if not args.debugging_mode:
        if not os.path.exists(os.path.join(args.project_dir,args.project_name,args.run_name)):
            os.makedirs(os.path.join(args.project_dir,args.project_name,args.run_name,args.logs_dir))

        os.environ["WANDB_DIR"]       = os.path.join(args.project_dir,args.project_name,args.run_name,args.logs_dir)
        os.environ["WANDB_CACHE_DIR"] = os.path.join(args.project_dir,args.project_name,args.run_name,args.logs_dir)

   
    wandb_logger = WandbLogger( 
    project= args.project_name,
    name   = args.run_name)
    wandb_logger.experiment.notes = args.run_notes

    succ_rate_checkpoint_callback = ModelCheckpoint(
        dirpath = checkpoints_dir,
        filename= '{epoch}-{success_rate:.2f}',
        monitor="success_rate",  # Monitor validation loss
        mode="max",  # "min" if you want to save the lowest validation loss
        save_top_k=10,  # Save only the best model
        save_last=True,  # Save the last model as well
        every_n_epochs=args.checkpoint_every,
        save_on_train_epoch_end=True
        )
    train_loss_checkpoint_callback = ModelCheckpoint(
        dirpath = checkpoints_dir,
        filename= '{epoch}-{train_loss:.2f}',
        monitor="train_loss",  # Monitor validation loss
        mode="min",  # "min" if you want to save the lowest validation loss
        save_top_k=10,  # Save only the best model
        save_last=True,  # Save the last model as well
        every_n_epochs=args.checkpoint_every,
        save_on_train_epoch_end=True
        )
    checkpoint_callback = train_loss_checkpoint_callback if args.trainloss_checkpoint else succ_rate_checkpoint_callback
    
    lr_logger_callback  = LearningRateMonitor(logging_interval='step')

    tasks_commands = json.load(open(args.tasks_commands_dir))
    tasks_commands = {k:list(set(tasks_commands[k])) for k in args.tasks} #the commands dict should have the same order as args.tasks list
   
    train_tasks_commands,val_tasks_commands = split_dict(tasks_commands,args.commands_split_ratio,seed=args.seed)
   
    model = TL_model(args=args,tasks_commands=val_tasks_commands,env=meta_env,wandb_logger=wandb_logger,seed=args.seed)
    model = load_checkpoint(model,args.load_weights)
    model = freeze_layers(model , args)
   
    sampler = None
    shaffle = True
    if args.debugging_mode:
        train_dataset = temp_dataset(seq_len=args.seq_len,seq_overlap=args.seq_overlap,cams = args.cams,with_imgs='obs' not in args.model)
    else:
        train_dataset = MW_dataset(model.preprocess,os.path.join(data_dir,'dataset_dict.json'),os.path.join(data_dir,'data'),train_tasks_commands,total_data_len=args.train_data_total_steps,seq_len=args.seq_len,seq_overlap=args.seq_overlap,cams = args.cams,with_imgs='obs' not in args.model, with_rtg= not args.with_no_rtg)
        stats_table = train_dataset.get_stats()
        model.model.set_dataset_specs(train_dataset.data_specs)
        #test dataset class
        print('dataset length ',len(train_dataset) ,'test with idx ',len(train_dataset)-1)
        train_dataset[len(train_dataset)-1]

        wandb_logger.log_table(key=f"Dataset Success Rate",  columns=['Task name','Success Rate'],data=stats_table)
        if args.seq_len>1:
            p_sample = train_dataset.data_specs['p_sample']
            sampler = WeightedRandomSampler(torch.Tensor(p_sample),len(p_sample),replacement=True)
            shaffle = False
            print('samples length ',len(sampler))

    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=shaffle,num_workers = args.num_workers,pin_memory=True,sampler=sampler,drop_last=True)


    trainer = Trainer(default_root_dir=checkpoints_dir,callbacks=[lr_logger_callback,checkpoint_callback],logger = wandb_logger,max_epochs=args.num_epochs,strategy='ddp_find_unused_parameters_true',devices=args.n_gpus)#,reload_dataloaders_every_n_epochs=args.generate_data_every,use_distributed_sampler=False)
    trainer.fit(model,train_dataloader,ckpt_path= args.load_checkpoint_path)



if __name__ == "__main__":
    main()
    