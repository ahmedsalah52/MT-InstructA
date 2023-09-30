import pytorch_lightning as pl

import torch.nn as nn
import torch
import math
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
import timm
import numpy as np
import wandb
import random
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
        self.pos_emp = nn.Linear(8,512)
        self.head = nn.Sequential(nn.Flatten(),
                                    nn.Linear(7*512, 512),
                                     nn.ReLU(),
                                     nn.Linear(512,512),
                                     nn.ReLU(),
                                     nn.Linear(512,4))
            
                             
    def forward(self,batch):
        # Getting Image and Text Features
        
        batch_size,cams,ch,h,w  = batch['images'].shape
        batch["images"] = torch.flatten(batch["images"], start_dim=0, end_dim=1)

        image_features = self.image_encoder(batch["images"])
        image_features = torch.unflatten(image_features,dim = 0,sizes=(batch_size,cams))


        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )

        
        image_features = self.image_projection(image_features)
        text_features  = self.text_projection(text_features)
        pos_embeddings = self.pos_emp(batch['hand_pos'])
       
        text_images_embeddings = torch.cat([image_features,text_features[:,None,:],pos_embeddings[:,None,:]],dim=1)

        logits = self.head(text_images_embeddings)

        return logits



class base_model(pl.LightningModule):
    def __init__(self,args,generator,env,wandb_logger,seed):
        super().__init__()
        torch.manual_seed(seed)  

        self.generator = generator
        self.generate_data_every = args.generate_data_every
        self.evaluate_every = args.evaluate_every
        self.evaluation_episodes = args.evaluation_episodes
        self.tasks = args.tasks
        self.env = env
        self.wandb_logger = wandb_logger
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
        
        text_batch = self.model.text_encoder.tokenizer(batch['instruction'], padding=True, truncation=True, max_length=self.model.text_encoder.command_max_length)
        text_batch = {k : torch.tensor(v) for k,v in text_batch.items()}
        

        batch = {**batch,**text_batch}
        batch = {k : v.to(self.device) for k,v in batch.items() if k != 'instruction'}
        
        logits = self.model(batch)

        y = batch['action']
        loss = self.loss_fun(logits, y)
        self.log("train_loss", loss,sync_dist=True)
        return loss
    
  
    
    def validation_step(self, batch, batch_idx):
        text_batch = self.model.text_encoder.tokenizer(batch['instruction'], padding=True, truncation=True, max_length=self.model.text_encoder.command_max_length)
        text_batch = {k : torch.tensor(v) for k,v in text_batch.items()}
        
        
        batch = {**batch,**text_batch}
        batch = {k : v.to(self.device) for k,v in batch.items() if k != 'instruction'}
        
        logits = self.model(batch)

        y = batch['action']

        loss = self.loss_fun(logits, y)
        self.log("val_loss", loss,sync_dist=True)
        return loss
    

    def train_dataloader(self):
        if (self.current_epoch % self.generate_data_every == 0):
            print(f"epoch {self.current_epoch} training data generation on device {self.device}")
            return self.generator.get_train_dataloader(self.device)
        else:
            return self.train_dataloader

    def val_dataloader(self):
        print(f"epoch {self.current_epoch} validation data generation on device {self.device}")
        return  self.generator.get_valid_dataloader(self.device)
        
    #def on_validation_epoch_start(self):
        


    def on_train_epoch_start(self):
        if (self.current_epoch % self.evaluate_every == 0):
            print(f"epoch {self.current_epoch}  evaluation on device {self.device}")
            total_success = 0
            total_vids =[]
            for task in self.tasks:
                videos=[]
                for pos in [0,1,2]:
                    env = self.env(task,pos,save_images=True,wandb_render = False,wandb_log = False)
                    #for i in range(self.evaluation_episodes):
                    obs , info = env.reset()
                    instruction = random.choice(self.generator.tasks_commands[task])
                    text_batch = self.model.text_encoder.tokenizer(instruction, padding=True, truncation=True, max_length=self.model.text_encoder.command_max_length)
                    rendered_seq = []
                    while 1:

                        step_input = {k : torch.tensor(v).unsqueeze(0).to(self.device) for k,v in text_batch.items()}
                        images = [self.generator.preprocess(img) for img in info['images']]
                        step_input['images']   = torch.stack(images).unsqueeze(0).to(self.device)
                        step_input['hand_pos'] = torch.tensor(np.concatenate((obs[0:4],obs[18:22]),axis =0)).to(torch.float32).unsqueeze(0).to(self.device)
                        a = self.model(step_input)
                        obs, reward, done,success, info = env.step(a.detach().cpu().numpy()[0]) 
                        total_success+=success
                        rendered_seq.append(env.get_visual_obs_log())
                        if (success or done): break 

                    rendered_seq = np.array(rendered_seq, dtype=np.uint8)
                    rendered_seq = rendered_seq.transpose(0,3, 1, 2)
                    videos.append(wandb.Video(rendered_seq, fps=30))   
                total_vids.append(videos[:])    
            self.wandb_logger.log_table(key="videos",  columns=['Right','Mid','Left'],data=total_vids)
            
            
            #if self.end_episode:
                
            self.log("success_rate", float(total_success)/(len(self.tasks)*3*self.evaluation_episodes))

    def configure_optimizers(self):
        return self.opt
