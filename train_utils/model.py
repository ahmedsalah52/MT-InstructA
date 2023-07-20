import torch.nn as nn
import torch
import math


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

class Transformer(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    # Constructor
    def __init__(
        self,
        dim_model,
        num_heads,
        num_encoder_layers,
        dropout_p,
        seq_length,
        emp_length,
        num_actions,
        variations_per_action,
        device
    ):
        super().__init__()
        self.device = device
        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=seq_length
        )
        #self.embedding = nn.Embedding(8, dim_model)
       
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads,dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.flatten = nn.Flatten(1)
        #self.outs    = [nn.Linear(seq_length*emp_length, variations_per_action).to(self.device) for i in range(num_actions)]
        self.outs    = nn.Sequential(nn.Linear(seq_length*emp_length, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024,512),
                                     nn.ReLU(),
                                     nn.Linear(512,256),
                                     nn.ReLU(),
                                     nn.Linear(256,num_actions),
                                     nn.Sigmoid()
        ).to(device)

        self.pos_emp = nn.Linear(3,8).to(device)

 

    def forward(self, src,hand_pos):
        pos_embeddings  = self.pos_emp(hand_pos)
        src = torch.cat([src,pos_embeddings[:,None,:]],dim=1)

        src = self.positional_encoder(src)
      
        src = src.permute(1,0,2)
        
        transformer_out = self.transformer(src)
        transformer_out = transformer_out.permute(1,0,2)
        transformer_out = self.flatten(transformer_out)
        
        
        out = self.outs(transformer_out)

        #rets = torch.cat(rets,1)

        return out


class Policy(nn.Module):
    def __init__(self,language_img_model,policy_head,seq_length,emp_length,device):
        super().__init__()
        self.device = device
        self.language_img_model = language_img_model
        self.policy_head = policy_head
        self.seq_length = seq_length
        self.emp_length = emp_length
        
        self.language_img_model.eval()

       
    def forward(self,batch):
        if 'embeddings' in batch.keys():
            text_images_embeddings = batch['embeddings']
        else:
            batch_size,cams,ch,h,w  = batch['image'].shape
            batch["image"] = torch.flatten(batch["image"], start_dim=0, end_dim=1)
            image_features = self.language_img_model.encode_image(batch["image"])
            text_features = self.language_img_model.encode_text(batch["caption"])
            
            image_features = torch.unflatten(image_features,dim = 0,sizes=(batch_size,cams))

            text_images_embeddings = torch.cat([image_features,text_features[:,None,:]],dim=1)
            text_images_embeddings = text_images_embeddings.flatten(1)
            text_images_embeddings = text_images_embeddings.unflatten(-1,(self.seq_length-1,self.emp_length)) # batch ,384 , 8
        
       

        logits = self.policy_head(text_images_embeddings,batch['hand_pos'])


        if 'embeddings' not in batch.keys():
            return logits , text_images_embeddings.detach().cpu().squeeze(0)
        else:
            return logits
    

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
    for param_group in optimizer.param_groups:
        return param_group["lr"]