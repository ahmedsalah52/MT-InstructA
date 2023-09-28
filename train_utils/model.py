import lightning as L

import torch.nn as nn
import torch
import math
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
import timm
import numpy as np


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    ret = []
    for param_group in optimizer.param_groups:
        ret.append(param_group["lr"])
    return ret
    

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name='resnet50', pretrained=True, trainable=True):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
    
class TextEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", pretrained=True, trainable=True,command_max_length=20):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.command_max_length = command_max_length
    
    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]
    
class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim = 2048,
        projection_dim=512,
        dropout=0.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x  
    

class ClIP(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.image_encoder    = ImageEncoder(model_name=args.image_model_name, pretrained=args.image_model_pretrained, trainable=args.image_model_trainable)
        self.text_encoder     = TextEncoder(model_name=args.text_model_name, pretrained=args.text_model_pretrained, trainable=args.text_model_trainable,command_max_length=args.text_model_max_length)
        self.image_projection = ProjectionHead(embedding_dim=2048)
        self.text_projection  = ProjectionHead(embedding_dim=768)
        self.pos_emp = nn.Linear(3,512)
        self.head = nn.Sequential(nn.Flatten(),
                                    nn.Linear(7*512, 512),
                                     nn.ReLU(),
                                     nn.Linear(512,512),
                                     nn.ReLU(),
                                     nn.Linear(512,4))
            
                             
    def forward(self,batch):
        # Getting Image and Text Features
        text_batch  = self.text_encoder.tokenizer(batch['command'], padding=True, truncation=True, max_length=self.text_encoder.command_max_length)

        batch_size,cams,ch,h,w  = batch['image'].shape
        batch["image"] = torch.flatten(batch["image"], start_dim=0, end_dim=1)

        image_features = self.image_encoder(batch["image"])
        

        text_features = self.text_encoder(
            input_ids=text_batch["input_ids"], attention_mask=text_batch["attention_mask"]
        )

        image_features = torch.unflatten(image_features,dim = 0,sizes=(batch_size,cams))
        
        image_features = self.image_projection(image_features)
        text_features  = self.text_projection(text_features)
        pos_embeddings = self.pos_emp(batch['hand_pos'])
        text_images_embeddings = torch.cat([image_features,text_features[:,None,:],pos_embeddings[:,None,:]],dim=1)

        logits = self.head(text_images_embeddings)

        return logits



class base_model(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        models = {'clip':ClIP}
        loss_funs = {'cross_entropy':nn.CrossEntropyLoss(),
                     'mse':nn.MSELoss()}
        self.model = models[args.model_name](args)
        self.loss_fun = loss_funs[args.loss_fun]
        self.opt = torch.optim.Adam(
                [
                    {"params": self.model.image_encoder.parameters()   , "lr": args.img_model_lr},
                    {"params": self.model.image_projection.parameters(), "lr": args.img_model_lr},
                    {"params": self.model.text_encoder.parameters()    , "lr": args.txt_model_lr},
                    {"params": self.model.text_projection.parameters() , "lr": args.txt_model_lr},
                    {"params": self.model.pos_emp.parameters()},
                    {"params": self.model.head.parameters()},
                ],
                lr=args.lr,
            )
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)

        
        loss = self.loss_fun(logits, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fun(logits, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return self.opt
