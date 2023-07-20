#pip3 install torch torchvision torchaudio
#pip install transformers
#pip install ftfy regex tqdm
#pip install git+https://github.com/openai/CLIP.git


from torchvision import transforms
import itertools

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import clip
from train_utils.model import Transformer,Policy,AvgMeter,get_lr
from train_utils.datasets import Metaworld_Dataset_live
import wandb
import numpy as np
class CFG():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_train_per = 0.8
    lr   = 1e-4
    weight_decay = 1e-3
    patience = 10
    factor = 0.8
    episodes_per_model = 20
    epochs = 200
    epochs_per_model = 10
    batch_size = 10
    workers = 12
    instructs_file_dir = 'instructions/button_press.json'
    min_expert_prob = 0.2
    max_expert_prob = 1.0
    steps_sampling_ratio = 0.5
    


def train_epoch(model, train_loader, optimizer, lr_scheduler, step,criterion):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))

    for batch in tqdm_object:
        batch['embeddings'] = batch['embeddings'].to(CFG.device)
        batch['expert_a']   = batch['expert_a'].to(CFG.device)
        batch['hand_pos']   = batch['hand_pos'].to(CFG.device)
        
        logits = model(batch)
       
        loss = criterion(logits.float(),batch['expert_a'].float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["embeddings"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

    return loss_meter


def valid_epoch(model, valid_loader,criterion):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch['embeddings']   = batch['embeddings'].to(CFG.device)
        batch['expert_a'] = batch['expert_a'].to(CFG.device)
        batch['hand_pos'] = batch['hand_pos'].to(CFG.device)
        logits = model(batch)
       
        loss = criterion(logits.float(),batch['expert_a'].float())
        count = batch["embeddings"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(valid_loss=loss.mean())
    
    return loss_meter 
from torchvision import transforms

def main():

    wandb.init(project='instruct_RL')

    seq_length = 385 #(5 imgs + 1 text encoded to * 512) = 384*8  + 1*8 pos emp
    emp_length = 8
    
    
    policy_head = Transformer(
    dim_model=8,
    num_heads=8,
    num_encoder_layers=4,
    dropout_p=0.1,
    seq_length = seq_length,
    emp_length = emp_length,
    num_actions= 4,
    variations_per_action = 3,
    device=CFG.device
    ).to(CFG.device) 
    clip_model , preprocess = clip.load("ViT-B/32", device=CFG.device)
    
    policy = Policy(language_img_model=clip_model,
                    policy_head=policy_head,
                    seq_length=seq_length,
                    emp_length=emp_length,
                    device=CFG.device
                    )
    #policy.policy_head.load_state_dict(torch.load('last_1.pt'))


    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    params = [
        {"params": policy.policy_head.parameters(), "lr": CFG.lr},
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=CFG.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"
    models_number = (CFG.epochs//CFG.epochs_per_model)+20
    for model_num in range(20,models_number):
        print("Generating new Data")
        prob_expert_generate =   CFG.max_expert_prob  - (model_num/models_number *(CFG.max_expert_prob-CFG.min_expert_prob))   # starts from max to min value
        
        policy.eval()

        dataset = Metaworld_Dataset_live(policy,preprocess,clip,prob_expert_generate,CFG)
        dataset_length = len(dataset)

        wandb.log({'success Rate'       : (dataset.dones/CFG.episodes_per_model)*100 , 'model_num': model_num})
        wandb.log({'Expert actions Rate': prob_expert_generate                       , 'model_num': model_num})
        wandb.log({'dataset length'     : dataset_length                             , 'model_num': model_num})
        wandb.log({'lr'                 : get_lr(optimizer)                          , 'model_num': model_num})


        trainset_length = int(dataset_length * CFG.dataset_train_per)
        print('dataset len:',dataset_length,' trainset len:',trainset_length)
        train_set, val_valid = torch.utils.data.random_split(dataset, [trainset_length, dataset_length - trainset_length])

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=CFG.batch_size,shuffle=True ,num_workers = CFG.workers)
        valid_loader = torch.utils.data.DataLoader(val_valid, batch_size=CFG.batch_size,shuffle=False,num_workers = CFG.workers)
        
        best_loss = float('inf')

        print("Training the model")
        total_valid_loss = 0
        for epoch in range(CFG.epochs_per_model):
            print("Model num {} Epoch: {}".format(model_num,epoch + 1))
            policy.policy_head.train()
            train_loss  = train_epoch(policy, train_loader, optimizer, lr_scheduler, step,criterion)
            wandb.log({'train_loss' : train_loss.avg, 'epoch':(model_num*CFG.epochs_per_model)+epoch, 'model_num': model_num})

            torch.save(policy.policy_head.state_dict(), "last_{}.pt".format(model_num))

            policy.eval()
            with torch.no_grad():
                valid_loss  = valid_epoch(policy, valid_loader,criterion)
                wandb.log({'valid_loss' : valid_loss.avg, 'epoch':(model_num*CFG.epochs_per_model)+epoch, 'model_num': model_num})

                #wandb.log({"valid_loss" : loss_item})
                #wandb.log({"valid_dones": dones})

                if valid_loss.avg < best_loss:
                    best_loss = valid_loss.avg
                    torch.save(policy.policy_head.state_dict(), "best_{}.pt".format(model_num))
                    print("Saved Best Model!")
                total_valid_loss += valid_loss.avg
                
        lr_scheduler.step(total_valid_loss/CFG.epochs_per_model)


main()