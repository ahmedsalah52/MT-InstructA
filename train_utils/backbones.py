import torch.nn as nn
import torch
from torch import functional as F

from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
import timm
import clip
from torchvision import transforms



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
        self.args = args
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

    def get_opt(self):
        return torch.optim.Adam(
                [
                    {"params": self.image_encoder.parameters()   , "lr": args.img_model_lr},
                    {"params": self.image_projection.parameters(), "lr": args.img_model_lr},
                    {"params": self.text_encoder.parameters()    , "lr": args.txt_model_lr},
                    {"params": self.text_projection.parameters() , "lr": args.txt_model_lr},
                    {"params": self.pos_emp.parameters()},
                    {"params": self.head.parameters()},
                ],
                lr=self.args.lr,
            )
    


class Open_AI_CLIP(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.model, self.preprocess_image = clip.load(args.op_image_model_name,jit=False)
        self.model = self.model.float()
        self.pos_emp = nn.Linear(8,args.pos_emp)
        self.flatten = nn.Flatten()
        
        #self.grad_clip = nn.utils.clip_grad_norm_(self.parameters(), 0.5)
    def forward(self,batch,cat=True):
        batch_size,cams,ch,h,w  = batch['images'].shape


        batch["images"] = torch.flatten(batch["images"], start_dim=0, end_dim=1)
        image_features = self.model.encode_image(batch['images'])
        image_features = torch.unflatten(image_features,dim = 0,sizes=(batch_size,cams))
        batch["images"] = torch.unflatten(batch["images"],dim = 0,sizes=(batch_size,cams))


        text = clip.tokenize(batch['instruction']).to(batch['images'].device)
        text_features  = self.model.encode_text(text)
        pos_embeddings = self.pos_emp(batch['hand_pos'])
        
        if not cat: 
            return image_features,text_features,pos_embeddings
        text_images_embeddings = torch.cat([image_features,text_features[:,None,:]],dim=1)
        text_images_embeddings = self.flatten(text_images_embeddings)
        return torch.cat([text_images_embeddings,pos_embeddings],dim=1)
    
    def get_opt_params(self):
        return  [
            {"params": self.model.parameters(),"lr": self.args.clip_lr},
            {"params": self.pos_emp.parameters()}
             ]