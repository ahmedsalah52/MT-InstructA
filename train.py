from model.model import *
from model.clip import CLIPModel 
from transformers import DistilBertTokenizer
from torchvision import transforms
from model.datasets import Metaworld_Dataset
import itertools
from tqdm.autonotebook import tqdm

import torch
from torch import nn
import torch.nn.functional as F
import timm


class CFG:
    debug = False
    batch_size = 32
    num_workers = 4
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    policy_head_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('trainig on ',device)
    model_name = 'resnet50'
    image_embedding = 2048
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200
    dataset_train_per = 0.8
    pretrained = True # for both image encoder and text encoder
    trainable = False # for both image encoder and text encoder
    temperature = 1.0

    # image size
    size = 224

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256 
    dropout = 0.1



def train_epoch(model, train_loader, optimizer, lr_scheduler, step,criterion):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        logits = model(batch)
       
        loss = 0
        for i in range(4):
            loss += criterion(logits[i],batch['action'][:,i].to(torch.int64))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter



def main():
    batch_size = 64
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)

    images_transforms  = transforms.Compose([transforms.ToTensor(),transforms.Resize((CFG.size,CFG.size))])
    actions_transforms = transforms.Compose([transforms.ToTensor()])
    dataset = Metaworld_Dataset('/media/ahmed/HDD/WorkSpace/master_thesis/repo/datasets/metaworld/single_env.json','/media/ahmed/HDD/WorkSpace/master_thesis/repo/datasets/metaworld/',images_transforms,actions_transforms,tokenizer)
    
    dataset_length = len(dataset)
    train_set, val_valid = torch.utils.data.random_split(dataset, [int(dataset_length*CFG.dataset_train_per), int(dataset_length*(1-CFG.dataset_train_per))])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True)
    valid_loader = torch.utils.data.DataLoader(val_valid, batch_size=batch_size,shuffle=False)



    criterion = nn.CrossEntropyLoss()
    language_text_model = CLIPModel().to(CFG.device)
    policy_head = Transformer( dim_model=8, num_heads=2, num_encoder_layers=3, dropout_p=0.1).to(CFG.device) 
    policy = Policy(language_text_model,policy_head)


    params = [
        {"params": language_text_model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
        {"params": language_text_model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
        {"params": itertools.chain(
            language_text_model.image_projection.parameters(), language_text_model.text_projection.parameters()
        ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay},
        {"params": policy_head.parameters(), "lr": CFG.policy_head_lr},

    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        policy.train()
        train_loss = train_epoch(policy, train_loader, optimizer, lr_scheduler, step,criterion)
        print(train_loss)
        #model.eval()
        #with torch.no_grad():
        #    valid_loss = valid_epoch(model, valid_loader)
        
        #if valid_loss.avg < best_loss:
        #    best_loss = valid_loss.avg
        #    torch.save(model.state_dict(), "best.pt")
        #    print("Saved Best Model!")
        
        #lr_scheduler.step(valid_loss.avg)