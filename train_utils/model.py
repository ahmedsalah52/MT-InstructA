import pytorch_lightning as pl

import torch.nn as nn
import torch
import math
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
import timm
import numpy as np
import wandb
import random
import clip
from torchvision import transforms
from PIL import Image 



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
        self.head    = nn.Sequential(nn.Flatten(),
                                    nn.Linear(7*512, 512),
                                     nn.ReLU(),
                                     nn.Linear(512,4))
        self.preprocess_image =  transforms.Compose([
            transforms.ToTensor(), 
            #transforms.Resize((224,224)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])      
    def forward(self,batch):
        # Getting Image and Text Features
       
        
        batch_size,cams,ch,h,w  = batch['images'].shape
        batch["images"] = torch.flatten(batch["images"], start_dim=0, end_dim=1)

        image_features = self.image_encoder(batch["images"])
        image_features = torch.unflatten(image_features,dim = 0,sizes=(batch_size,cams))

        text_batch = self.text_encoder.tokenizer(batch['instruction'], padding=True, truncation=True, max_length=self.text_encoder.command_max_length)
        text_batch = {k : torch.tensor(v).to(batch['hand_pos'].device) for k,v in text_batch.items()}
        
        text_features = self.text_encoder(
            input_ids=text_batch["input_ids"], attention_mask=text_batch["attention_mask"]
        )
        
        
        image_features = self.image_projection(image_features)
        text_features  = self.text_projection(text_features)
        pos_embeddings = self.pos_emp(batch['hand_pos'])
       
        text_images_embeddings = torch.cat([image_features,text_features[:,None,:],pos_embeddings[:,None,:]],dim=1)

        logits = self.head(text_images_embeddings)

        return logits

    def get_opt(self,args):
        return torch.optim.Adam(
                [
                    {"params": self.image_encoder.parameters()   , "lr": args.img_model_lr},
                    {"params": self.image_projection.parameters(), "lr": args.img_model_lr},
                    {"params": self.text_encoder.parameters()    , "lr": args.txt_model_lr},
                    {"params": self.text_projection.parameters() , "lr": args.txt_model_lr},
                    {"params": self.pos_emp.parameters()},
                    {"params": self.head.parameters()},
                ],
                lr=args.lr,
            )
    
class Open_AI_CLIP(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.model, self.preprocess_image = clip.load(args.op_image_model_name,jit=False)
        self.model = self.model.float()
        self.pos_emp = nn.Linear(8,512)
        self.head = nn.Sequential(nn.Flatten(),
                                     nn.Linear(7*512, 512),
                                     nn.LayerNorm(512),
                                     nn.ReLU(),
                                     nn.Linear(512,4))
        #self.grad_clip = nn.utils.clip_grad_norm_(self.parameters(), 0.5)
    def forward(self,batch):
        batch_size,cams,ch,h,w  = batch['images'].shape


        batch["images"] = torch.flatten(batch["images"], start_dim=0, end_dim=1)
        image_features = self.model.encode_image(batch['images'])
        image_features = torch.unflatten(image_features,dim = 0,sizes=(batch_size,cams))


        text = clip.tokenize(batch['instruction']).to(batch['images'].device)
        text_features  = self.model.encode_text(text)
        pos_embeddings = self.pos_emp(batch['hand_pos'])
        text_images_embeddings = torch.cat([image_features,text_features[:,None,:],pos_embeddings[:,None,:]],dim=1)

        logits = self.head(text_images_embeddings)
        return logits
    
    def get_opt(self,args):
        return torch.optim.Adam(
                [
                    {"params": self.model.parameters()   },
                    {"params": self.pos_emp.parameters()},
                    {"params": self.head.parameters()},
                ],
                lr=args.lr,
            )

class base_model(pl.LightningModule):
    def __init__(self,args,tasks_commands,env,wandb_logger,seed):
        super().__init__()
        self.tasks_commands = tasks_commands
        self.generate_data_every = args.generate_data_every
        self.evaluate_every = args.evaluate_every
        self.evaluation_episodes = args.evaluation_episodes
        self.tasks = args.tasks
        self.batch_size = args.batch_size
        self.env = env
        self.wandb_logger = wandb_logger
        loss_funs = {'cross_entropy':nn.CrossEntropyLoss(),
                     'mse':nn.MSELoss()}
        
        models = {'simple_clip':ClIP,'open_ai_clip':Open_AI_CLIP}
        self.loss_fun = loss_funs[args.loss_fun]
        self.model = models[args.model_name](args)
        self.opt = self.model.get_opt(args)
        self.preprocess = self.model.preprocess_image

    def training_step(self, batch, batch_idx):
        
        

        batch = {k : v.to(self.device) if k != 'instruction' else v  for k,v in batch.items()}
        
        logits = self.model(batch)

        y = batch['action']
        loss = self.loss_fun(logits, y)
        self.log("train_loss", loss,sync_dist=True,batch_size=self.batch_size,prog_bar=True)
        return loss
    
  
    
    def validation_step(self, batch, batch_idx):
        
        
        batch = {k : v.to(self.device) if k != 'instruction' else v  for k,v in batch.items()}
        
        logits = self.model(batch)

        y = batch['action']

        loss = self.loss_fun(logits, y)
        self.log("val_loss", loss,sync_dist=True,batch_size=self.batch_size)
        return loss
    

    def on_train_epoch_start(self):
        if (self.current_epoch % self.evaluate_every == 0):
            print(f"epoch {self.current_epoch}  evaluation on device {self.device}")
            total_success = 0
            total_vids =[]
            for task in self.tasks:
                videos=[]
                for pos in [0,1,2]:
                    env = self.env(task,pos,save_images=True,wandb_render = False,wandb_log = False,general_model=True)
                    #for i in range(self.evaluation_episodes):
                    obs , info = env.reset()
                    instruction = random.choice(self.tasks_commands[task])
                    rendered_seq = []
                    while 1:
                        
                        step_input = {'instruction':[instruction]}
                        images = [self.model.preprocess_image(Image.fromarray(np.uint8(img))) for img in info['images']]
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
                total_vids.append(list(reversed(videos)))    
            self.wandb_logger.log_table(key=f"videos {self.current_epoch}",  columns=['Left','Mid','Right'],data=total_vids,step=self.current_epoch)
                        
            self.log("success_rate", float(total_success)/(len(self.tasks)*3*self.evaluation_episodes),sync_dist=True,batch_size=self.batch_size) # type: ignore

    def configure_optimizers(self):
        return self.opt

